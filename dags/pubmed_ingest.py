"""
Airflow DAG: PubMed Abstract Ingestion

Runs daily. For each configured MeSH category:
1. Check ingestion_state for last fetched date
2. Query PubMed for new abstracts since that date
3. Fetch full metadata and upsert into Postgres
4. Update ingestion state

On first run (or with backfill=True), fetches the last 5 years of abstracts.
"""

import json
import logging
from datetime import datetime, timedelta, date

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.providers.postgres.hooks.postgres import PostgresHook

from src.ingestion.pubmed_client import PubMedClient, CATEGORIES

logger = logging.getLogger(__name__)

POSTGRES_CONN_ID = "pubmed_postgres"
PUBMED_API_KEY = Variable.get("pubmed_api_key", default_var=None)
BACKFILL_YEARS = 5
MAX_RESULTS_PER_RUN = 10_000

default_args = {
    "owner": "pubmed-ml-platform",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="pubmed_ingest",
    default_args=default_args,
    description="Ingest PubMed abstracts by MeSH category",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ingestion", "pubmed"],
    params={"backfill": False},
) as dag:

    @task
    def get_ingestion_state(category: str, **context) -> dict:
        """Get the last fetched date for a category, or set backfill start."""
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        result = hook.get_first(
            "SELECT last_fetched_date FROM ingestion_state WHERE category = %s",
            parameters=(category,),
        )

        is_backfill = context["params"].get("backfill", False)

        if result and not is_backfill:
            return {"category": category, "min_date": result[0].isoformat()}
        else:
            backfill_start = date.today() - timedelta(days=365 * BACKFILL_YEARS)
            return {"category": category, "min_date": backfill_start.isoformat()}

    @task
    def fetch_abstracts(state: dict) -> dict:
        """Search PubMed and fetch abstracts for a category."""
        category = state["category"]
        min_date = date.fromisoformat(state["min_date"])
        query = CATEGORIES[category]

        client = PubMedClient(api_key=PUBMED_API_KEY)

        # Search for PMIDs
        all_pmids = []
        retstart = 0
        batch_size = 500

        while retstart == 0 or retstart < total:
            pmids, total = client.search(
                query=query,
                min_date=min_date,
                max_date=date.today(),
                retmax=batch_size,
                retstart=retstart,
            )
            all_pmids.extend(pmids)
            retstart += batch_size

            if len(all_pmids) >= MAX_RESULTS_PER_RUN:
                logger.warning(
                    f"Hit max results ({MAX_RESULTS_PER_RUN}) for {category}, "
                    f"total available: {total}"
                )
                all_pmids = all_pmids[:MAX_RESULTS_PER_RUN]
                break

        if not all_pmids:
            logger.info(f"No new abstracts for {category} since {min_date}")
            return {"category": category, "articles": [], "count": 0}

        # Fetch full metadata
        articles = client.fetch_articles(all_pmids)

        # Serialize for XCom
        serialized = [
            {
                "pmid": a.pmid,
                "title": a.title,
                "abstract": a.abstract,
                "authors": a.authors,
                "journal": a.journal,
                "pub_date": a.pub_date.isoformat() if a.pub_date else None,
                "mesh_terms": a.mesh_terms,
                "keywords": a.keywords,
                "doi": a.doi,
            }
            for a in articles
        ]

        logger.info(f"Fetched {len(serialized)} articles for {category}")
        return {"category": category, "articles": serialized, "count": len(serialized)}

    @task
    def load_to_postgres(fetch_result: dict) -> dict:
        """Upsert articles into Postgres and update ingestion state."""
        category = fetch_result["category"]
        articles = fetch_result["articles"]

        if not articles:
            return {"category": category, "loaded": 0}

        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

        insert_sql = """
            INSERT INTO papers (pmid, title, abstract, authors, journal, pub_date,
                                mesh_terms, keywords, doi, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (pmid) DO UPDATE SET
                title = EXCLUDED.title,
                abstract = EXCLUDED.abstract,
                authors = EXCLUDED.authors,
                journal = EXCLUDED.journal,
                pub_date = EXCLUDED.pub_date,
                mesh_terms = EXCLUDED.mesh_terms,
                keywords = EXCLUDED.keywords,
                doi = EXCLUDED.doi,
                updated_at = NOW()
        """

        rows = [
            (
                a["pmid"],
                a["title"],
                a["abstract"],
                json.dumps(a["authors"]),
                a["journal"],
                a["pub_date"],
                json.dumps(a["mesh_terms"]),
                json.dumps(a["keywords"]),
                a["doi"],
            )
            for a in articles
        ]

        conn = hook.get_conn()
        cur = conn.cursor()
        for row in rows:
            cur.execute(insert_sql, row)
        conn.commit()
        cur.close()
        conn.close()

        # Update ingestion state
        state_sql = """
            INSERT INTO ingestion_state (category, last_fetched_date, total_fetched, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (category) DO UPDATE SET
                last_fetched_date = EXCLUDED.last_fetched_date,
                total_fetched = ingestion_state.total_fetched + EXCLUDED.total_fetched,
                updated_at = NOW()
        """
        hook.run(state_sql, parameters=(category, date.today().isoformat(), len(articles)))

        logger.info(f"Loaded {len(articles)} articles for {category}")
        return {"category": category, "loaded": len(articles)}

    @task
    def log_summary(results: list[dict]):
        """Log a summary of the ingestion run."""
        total = sum(r["loaded"] for r in results)
        by_category = {r["category"]: r["loaded"] for r in results}
        logger.info(f"Ingestion complete. Total: {total}. By category: {by_category}")

    # Build the DAG dynamically for each category
    all_results = []
    for category_name in CATEGORIES:
        state = get_ingestion_state(category_name)
        fetched = fetch_abstracts(state)
        loaded = load_to_postgres(fetched)
        all_results.append(loaded)

    log_summary(all_results)
