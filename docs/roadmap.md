# Production Roadmap: From 150 Chunks to 10 Million

## 1. Scale the retrieval path

At 10M chunks with 1,536-dimensional float32 vectors, raw vectors alone are
about 61 GB. HNSW graph overhead typically adds 1.5-2x, so a single
PostgreSQL instance is not enough without capacity planning.

Plan:

- Store `halfvec` or quantized vectors where recall permits; this roughly
  halves raw vector memory.
- Partition by tenant, product, and effective year; keep hot policy families
  in a smaller active partition.
- Use read replicas for query traffic and a write path for ingestion.
- Build HNSW/IVF indexes offline with controlled `maintenance_work_mem`,
  then publish through a blue/green index swap.
- Run batch embedding and incremental re-embedding only for changed documents.
- Add a candidate-generation service in front of the database when a single
  cluster can no longer meet latency or recall targets.

Measure recall against exact search on a fixed sample before accepting any ANN
parameter change. Track index build time, memory, P50/P99 latency and recall
together.

## 2. Freshness and caching

Policies need explicit `effective_from`, `effective_to`, `supersedes`, `status`
and `version` fields. A retrieval request should filter on the effective window
and prefer the authoritative version, while historical queries can opt into
older versions.

- Ingest changes through a queue or outbox table; do not rebuild the whole
  corpus for one policy edit.
- Cache exact normalized query + filters + index version + model version in
  Redis with a short TTL.
- Invalidate on policy publication events; use a versioned cache key so a
  stale answer cannot survive a policy change.
- Cache embeddings for frequent queries and precompute common FAQ results.
- Treat semantic caching as a separate, higher-risk feature: in banking, an
  exact cache is easier to audit than a fuzzy one.

## 3. Governance, PII and access control

The knowledge base contains policy text, but customer questions may contain
account numbers, card numbers, addresses and other sensitive data.

- Redact or tokenize PII before embedding, logging and third-party API calls.
- Store document-level ACLs and enforce them in SQL with row-level security;
  retrieval must never cross tenant or role boundaries.
- Keep an audit trail of query, retrieved document IDs, model version and
  policy version, with PII removed.
- Encrypt in transit and at rest; manage keys with KMS and rotate them.
- Require a DPA and data-residency review before using external embedding or
  reranking providers.
- Treat retrieved text as untrusted evidence, not as instructions; the answer
  layer should cite documents and reject unsupported claims.
- Monitor for prompt injection, data exfiltration and unusual retrieval
  patterns.

## 4. Service-level targets

For a customer-support Co-Pilot, separate the targets:

- Vector candidate retrieval: P99 under 100 ms.
- Hybrid + rerank: report end-to-end P50/P99 honestly; a Cross-Encoder will
  usually dominate this stage.
- Answer generation: stream first token quickly, but never trade citation
  correctness for latency.

Add dashboards for recall drift after policy updates, empty-result rate,
fallback-filter rate, reranker disagreement and per-product retrieval quality.
