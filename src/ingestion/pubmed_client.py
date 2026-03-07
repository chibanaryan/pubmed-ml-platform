"""
PubMed E-utilities API client.

Uses the NCBI E-utilities API to search and fetch biomedical abstracts.
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/

Rate limits: 3 requests/second without API key, 10/second with one.
Get a free key at https://www.ncbi.nlm.nih.gov/account/settings/
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import date
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# MeSH terms for target categories
CATEGORIES = {
    "nutrition": '"Nutritional Sciences"[MeSH] OR "Diet"[MeSH] OR "Dietary Supplements"[MeSH]',
    "exercise": '"Exercise"[MeSH] OR "Physical Fitness"[MeSH] OR "Sports"[MeSH]',
    "psychology": '"Psychology"[MeSH] OR "Behavioral Sciences"[MeSH] OR "Cognitive Science"[MeSH]',
    "habits": '"Habits"[MeSH] OR "Behavior, Addictive"[MeSH] OR "Health Behavior"[MeSH]',
    "ethics": '"Bioethics"[MeSH] OR "Ethics"[MeSH] OR "Moral Development"[MeSH]',
}


@dataclass
class PubMedArticle:
    pmid: int
    title: str
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    journal: str | None = None
    pub_date: date | None = None
    mesh_terms: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    doi: str | None = None


class PubMedClient:
    def __init__(self, api_key: str | None = None, requests_per_second: float = 3.0):
        self.api_key = api_key
        self.min_interval = 1.0 / (10.0 if api_key else requests_per_second)
        self.last_request_time = 0.0
        self.session = requests.Session()

    def _throttle(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def _get(self, endpoint: str, params: dict, max_retries: int = 3) -> requests.Response:
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{BASE_URL}/{endpoint}"

        for attempt in range(max_retries):
            self._throttle()
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt * 2
                logger.warning(f"Rate limited (429), retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp

        resp.raise_for_status()
        return resp

    def search(
        self,
        query: str,
        min_date: date | None = None,
        max_date: date | None = None,
        retmax: int = 500,
        retstart: int = 0,
    ) -> tuple[list[int], int]:
        """
        Search PubMed and return (list_of_pmids, total_count).
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retstart": retstart,
            "retmode": "json",
            "sort": "pub_date",
        }
        if min_date:
            params["mindate"] = min_date.strftime("%Y/%m/%d")
            params["datetype"] = "pdat"
        if max_date:
            params["maxdate"] = max_date.strftime("%Y/%m/%d")
            params["datetype"] = "pdat"

        resp = self._get("esearch.fcgi", params)
        data = resp.json()["esearchresult"]
        pmids = [int(pid) for pid in data.get("idlist", [])]
        total = int(data.get("count", 0))
        logger.info(f"Search returned {total} total results, fetched {len(pmids)} starting at {retstart}")
        return pmids, total

    def fetch_articles(self, pmids: list[int]) -> list[PubMedArticle]:
        """
        Fetch full article metadata for a list of PMIDs.
        Handles batches of up to 200 at a time.
        """
        articles = []
        batch_size = 200

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(str(p) for p in batch),
                "retmode": "xml",
            }
            resp = self._get("efetch.fcgi", params)
            articles.extend(self._parse_xml(resp.text))
            logger.info(f"Fetched batch {i // batch_size + 1}: {len(batch)} articles")

        return articles

    def _parse_xml(self, xml_text: str) -> list[PubMedArticle]:
        root = ElementTree.fromstring(xml_text)
        articles = []

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                articles.append(self._parse_article(article_elem))
            except Exception as e:
                pmid = article_elem.findtext(".//PMID", default="unknown")
                logger.warning(f"Failed to parse PMID {pmid}: {e}")

        return articles

    def _parse_article(self, elem) -> PubMedArticle:
        pmid = int(elem.findtext(".//PMID"))

        # Title
        title = elem.findtext(".//ArticleTitle", default="")

        # Abstract — may have multiple AbstractText elements (structured abstract)
        abstract_parts = []
        for at in elem.findall(".//AbstractText"):
            label = at.get("Label", "")
            text = "".join(at.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts) if abstract_parts else None

        # Authors
        authors = []
        for author in elem.findall(".//Author"):
            last = author.findtext("LastName", default="")
            first = author.findtext("ForeName", default="")
            if last:
                authors.append(f"{last}, {first}".strip(", "))

        # Journal
        journal = elem.findtext(".//Journal/Title")

        # Publication date
        pub_date = self._parse_pub_date(elem)

        # MeSH terms
        mesh_terms = [
            dh.findtext("DescriptorName", default="")
            for dh in elem.findall(".//MeshHeading")
            if dh.findtext("DescriptorName")
        ]

        # Keywords
        keywords = [
            kw.text for kw in elem.findall(".//Keyword") if kw.text
        ]

        # DOI
        doi = None
        for aid in elem.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text
                break

        return PubMedArticle(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            pub_date=pub_date,
            mesh_terms=mesh_terms,
            keywords=keywords,
            doi=doi,
        )

    def _parse_pub_date(self, elem) -> date | None:
        pd_elem = elem.find(".//PubDate")
        if pd_elem is None:
            return None
        year = pd_elem.findtext("Year")
        month = pd_elem.findtext("Month", default="01")
        day = pd_elem.findtext("Day", default="01")
        if not year:
            return None
        # Month might be text like "Jan"
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        try:
            month_int = int(month)
        except ValueError:
            month_int = month_map.get(month.lower()[:3], 1)
        try:
            return date(int(year), month_int, int(day))
        except (ValueError, TypeError):
            return date(int(year), 1, 1)
