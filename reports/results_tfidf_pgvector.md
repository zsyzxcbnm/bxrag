# Apex RAG Retrieval Evaluation

Generated at `2026-10-20T09:28:47.044196+00:00`.

## Setup

- Queries: **30**
- Storage backend: `postgres`
- Embedding: `local-tfidf-svd` (128 dimensions)
- Embedding provider: `local`
- Embedding device: `n/a`
- Reranker: `heuristic`
- RRF `k`: **60**
- RRF weights: dense **0.5**, sparse **0.5**
- Rerank candidates: **20**
- Results fetched per query: **10**
- Metadata rules: **disabled**

The headline run disables metadata rules: card-tier/year filters and metadata heuristic bonuses are disabled. Metadata rules are opt-in and evaluated separately in `ablation_metadata.md`. The configured embedding and reranker providers are
recorded above so the run is reproducible. The OpenAI embedding and
Sentence-Transformers Cross-Encoder providers can be enabled through `.env`.

## Results

| Pipeline | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR | nDCG@5 | nDCG@10 | P50 (ms) | P99 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.694 | 0.917 | 0.950 | 0.989 | 0.260 | 0.894 | 0.884 | 0.902 | 2.21 | 7.34 |
| Sparse only | 0.750 | 0.956 | 1.000 | 1.000 | 0.287 | 0.950 | 0.956 | 0.956 | 5.79 | 8.28 |
| Hybrid (RRF) | 0.750 | 0.944 | 0.989 | 1.000 | 0.280 | 0.950 | 0.939 | 0.944 | 7.25 | 11.29 |
| Hybrid + rerank | 0.772 | 0.917 | 0.989 | 1.000 | 0.280 | 0.950 | 0.943 | 0.948 | 8.74 | 12.45 |

The table's P50/P99 columns measure the retrieval path (query embedding excluded):
vector search, sparse search, fusion and reranking. Query-embedding latency is
reported separately because it depends on the model and device:

| Pipeline | Embedding P50 (ms) | Embedding P99 (ms) |
|---|---:|---:|
| Dense only | 2.49 | 3.28 |
| Sparse only | 0.00 | 0.00 |
| Hybrid (RRF) | 2.60 | 3.34 |
| Hybrid + rerank | 2.61 | 3.30 |

Approximate end-to-end latency (retrieval + embedding percentiles; this is a
sum of percentiles, not a joint distribution):

| Pipeline | Approx. end-to-end P50 (ms) | Approx. end-to-end P99 (ms) |
|---|---:|---:|
| Dense only | 4.70 | 10.62 |
| Sparse only | 5.79 | 8.28 |
| Hybrid (RRF) | 9.86 | 14.63 |
| Hybrid + rerank | 11.36 | 15.75 |

![Retrieval quality](figures/metrics_comparison.png)

![Latency](figures/latency_percentiles.png)

## What the comparison shows

- Dense-only is the required baseline; sparse-only isolates the value of exact
  lexical matching on policy numbers, product names and time limits.
- Hybrid retrieval uses Reciprocal Rank Fusion over dense and PostgreSQL
  full-text candidates.
- Sparse query terms are OR-ed before `ts_rank_cd`; strict AND semantics
  produced near-zero recall on long natural-language questions.
- Hybrid + rerank applies a second-stage scorer to the top candidates and
  returns the requested number of results.
- Recall and nDCG are reported at both 5 and 10. The evaluation fetches
  **10** results, so `Recall@10` is computed over ten actual documents.
  `@5` reflects the evidence window used by the demo, while `@10` measures
  wider candidate recall. On this small corpus `@10` often saturates quickly.

## Index verification

The PostgreSQL schema contains `idx_chunks_embedding_hnsw` (HNSW, cosine) and
`idx_chunks_fts` (GIN). Dense retrieval is split into an HNSW candidate query
and a separate document-metadata fetch; this prevents the planner from doing a
full scan plus sort just to join product metadata. The HNSW sweep in
`index_tuning.md` reports Recall@10 **1.000** for every tested configuration,
with candidate-search P50 around **2.5 ms** and P99 around **7-8 ms**. All
tested `m`, `ef_construction` and `ef_search` values remain far below 100 ms;
`m=16`, `ef_construction=64`, `ef_search=80` is the default.

## Failure analysis

Queries where hybrid + rerank moved the first relevant document upward:

- Q-006: rank 2 -> 1 — When will I receive my provisional credit during a transaction dispute?
- Q-012: rank 3 -> 1 — What is the deductible and coverage limit for cell phone protection on Apex Platinum?
- Q-016: rank 2 -> 1 — What is the fee for adding authorized users to an Apex Platinum card?

Queries where hybrid + rerank moved the first relevant document downward:

- None in this run.

Hybrid + rerank moved the first relevant document upward in 3 queries and downward in 0. These are per-query diagnostics; the aggregate table above is the primary comparison.

Card-tier confusion remains a plausible error source because the same policy exists for several products with slightly different numbers. It does not follow that metadata rules should be enabled by default: the separate metadata ablation is the appropriate test, and metadata rules remain disabled in the headline run.

A second issue is label exhaustiveness. For generic questions the judgment often lists only one or two card tiers even when another tier's policy contains the same fact. For example, DOC-0023 for Q-005 and DOC-0053 for Q-011 are semantically valid but not labeled relevant. Strict Recall/MRR therefore understates retrieval quality; it should not be tuned away by overfitting to the 30 labeled queries.

## Reproducibility and limitations

- All labels come from `data/eval_queries.json`; no judge or generated answer
  is used for the retrieval metrics.
- This is a small synthetic benchmark. Hyperparameters are global, not tuned
  per query.
- Dense quality depends on the selected embedding model. The local LSA
  embedding is an offline fallback; Qwen3/OpenAI embeddings can be selected
  through the provider configuration. BGE-large uses its retrieval query prefix.
- End-to-end latency includes query embedding and reranking; database-only
  latency is reported separately in the raw JSON.
- The evaluation requests **10** ranked documents and derives @5 and @10
  metrics from that same ranking.
