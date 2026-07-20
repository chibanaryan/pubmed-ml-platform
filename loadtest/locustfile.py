"""
Load test for the PubMed search API.

Run headless against the local stack (make up first):
    make loadtest
or interactively:
    locust -f loadtest/locustfile.py --host http://localhost:8000

The first /search request triggers lazy model loading, so warm the API
(one manual search) before trusting latency numbers.
"""

import random

from locust import HttpUser, between, task

QUERIES = [
    "effects of creatine supplementation on muscle recovery",
    "psychological effects of quitting alcohol",
    "benefits of high intensity interval training",
    "vegetarian diet and protein intake",
    "ethics of artificial intelligence in healthcare",
    "impact of sleep deprivation on cognitive performance",
    "gut microbiome and mental health connection",
    "resistance training for older adults",
]

# PMIDs are populated lazily from search results so /paper and /similar
# exercise real ids instead of 404s.
_seen_pmids: list[int] = []


class SearchUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task(6)
    def search(self):
        resp = self.client.post(
            "/search",
            json={"query": random.choice(QUERIES), "top_k": 10},
        )
        if resp.ok:
            for r in resp.json().get("results", [])[:3]:
                if len(_seen_pmids) < 500:
                    _seen_pmids.append(r["pmid"])

    @task(2)
    def get_paper(self):
        if not _seen_pmids:
            return
        pmid = random.choice(_seen_pmids)
        self.client.get(f"/paper/{pmid}", name="/paper/{pmid}")

    @task(1)
    def find_similar(self):
        if not _seen_pmids:
            return
        pmid = random.choice(_seen_pmids)
        self.client.get(f"/similar/{pmid}?top_k=5", name="/similar/{pmid}")

    @task(1)
    def health(self):
        self.client.get("/health")
