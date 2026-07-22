# TODO

Completed work lives in `DEVLOG.md` (chronological, with results and gotchas) and
`docs/portfolio-dossier.md` (thematic summary). This file is only what's next.

## Next

- [ ] **Run Airflow somewhere other than a laptop.** The DAG ingests and embeds correctly, but
      only advances when the local stack is up, so the hosted corpus is a fixed snapshot.
      Set `max_db_bytes` when pointing it at Neon (131MB headroom, ~13K papers).
- [ ] **Deploy the k8s manifests to a real cluster.** `k8s/` has never been applied. kind or k3s
      first, then HPA, Ingress, and either Helm or Kustomize.
- [ ] **Infrastructure as code.** Render service and Neon project are click/API-provisioned;
      no Terraform anywhere.
- [ ] **One GPU / distributed training run.** Every model so far was trained on a laptop.

## Worth doing, smaller

- [ ] Split query encoding into its own service. It shares a process with request serving,
      which is what puts the p99 at 460ms under load.
- [ ] Bulk-load embeddings with `COPY` instead of row-by-row inserts (23 min → seconds
      when re-embedding a full corpus over the network).
- [ ] Try 2 req/s against PubMed. It 429s steadily at its documented 3 req/s, so backing off
      may finish faster by avoiding 2–8s retry waits.
- [ ] Hand-label ~50 query/document pairs. The MeSH proxy is good enough to pick a model,
      not to judge re-ranking quality.
- [ ] Secrets hygiene: `k8s/postgres.yaml` carries plaintext dev credentials, and compose
      defaults to `pubmed:pubmed`.

## Deliberately not doing

- **Streaming ingestion.** Daily batch matches how PubMed publishes; Kafka would be costume.
- **A dedicated vector database.** pgvector holds at this scale; revisit past ~1M vectors.
- **Feature store.** No feature reuse across models to justify one.
