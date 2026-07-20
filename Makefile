.PHONY: up down test lint ingest embed compare evaluate eval-gate loadtest logs

MIN_NDCG ?= 0.80

up:
	docker compose up -d

down:
	docker compose down

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/ dags/ loadtest/

ingest:
	docker compose exec api python -c "\
		import json; \
		import psycopg2; \
		from datetime import date, timedelta; \
		from src.ingestion.pubmed_client import PubMedClient, CATEGORIES; \
		client = PubMedClient(requests_per_second=2.0); \
		conn = psycopg2.connect('postgresql://pubmed:pubmed@postgres:5432/pubmed'); \
		cur = conn.cursor(); \
		for cat, q in CATEGORIES.items(): \
			print(f'--- {cat} ---'); \
			pmids, total = client.search(q, min_date=date.today()-timedelta(days=365), retmax=500); \
			articles = client.fetch_articles(pmids); \
			for a in articles: \
				cur.execute('INSERT INTO papers (pmid,title,abstract,authors,journal,pub_date,mesh_terms,keywords,doi,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW()) ON CONFLICT (pmid) DO NOTHING', (a.pmid,a.title,a.abstract,json.dumps(a.authors),a.journal,a.pub_date,json.dumps(a.mesh_terms),json.dumps(a.keywords),a.doi)); \
			conn.commit(); \
			print(f'  loaded {len(articles)}'); \
		cur.execute('SELECT COUNT(*) FROM papers'); \
		print(f'Total: {cur.fetchone()[0]}'); \
		conn.close(); \
	"

embed:
	docker compose exec api python -m src.embeddings.embed_pipeline \
		--model minilm \
		--db-url postgresql://pubmed:pubmed@postgres:5432/pubmed \
		--mlflow-uri http://mlflow:5000

compare:
	docker compose exec api python -m src.embeddings.embed_pipeline \
		--compare \
		--db-url postgresql://pubmed:pubmed@postgres:5432/pubmed \
		--mlflow-uri http://mlflow:5000

evaluate:
	docker compose exec api python -m src.embeddings.evaluate \
		--compare \
		--db-url postgresql://pubmed:pubmed@postgres:5432/pubmed \
		--mlflow-uri http://mlflow:5000

eval-gate:
	docker compose exec api python -m src.embeddings.evaluate \
		--model minilm \
		--db-url postgresql://pubmed:pubmed@postgres:5432/pubmed \
		--min-ndcg $(MIN_NDCG)

loadtest:
	locust -f loadtest/locustfile.py --headless \
		--host http://localhost:8000 \
		--users 20 --spawn-rate 5 --run-time 60s \
		--only-summary

logs:
	docker compose logs -f --tail=50
