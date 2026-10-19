# Apex RAG Retrieval Evaluation

Generated at `2026-10-19T11:36:35.671015+00:00`.

## Setup

- Queries: **30**
- Storage backend: `postgres`
- Embedding: `local-tfidf-svd` (128 dimensions)
- Embedding provider: `local`
- Reranker: `heuristic`
- RRF `k`: **60**
- RRF weights: dense **0.4**, sparse **0.6**
- Rerank candidates: **20**

The local TF-IDF + SVD embedding and metadata-aware heuristic reranker make the
evaluation reproducible without network access or API keys. The OpenAI
embedding and Sentence-Transformers Cross-Encoder providers can be enabled
through `.env`.

## Results

| Pipeline | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR | nDCG@5 | P50 (ms) | P99 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.761 | 0.900 | 0.933 | 0.260 | 0.917 | 0.902 | 4.48 | 9.29 |
| Sparse only | 0.750 | 0.922 | 0.967 | 0.280 | 0.933 | 0.935 | 5.39 | 7.14 |
| Hybrid (RRF) | 0.783 | 0.911 | 0.967 | 0.280 | 0.950 | 0.941 | 7.70 | 11.62 |
| Hybrid + rerank | 0.783 | 0.944 | 0.967 | 0.280 | 0.950 | 0.950 | 9.12 | 11.91 |

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
  returns the final top 5.
- Recall@10 is deliberately not the headline metric: with 150 short documents
  and multi-label relevance it saturates quickly. Recall@1, MRR and nDCG@5 are
  the discriminating metrics.

## Index verification

The PostgreSQL schema contains `idx_chunks_embedding_hnsw` (HNSW, cosine) and
`idx_chunks_fts` (GIN). At only 150 chunks the planner correctly prefers a
sequential scan; forcing `enable_seqscan=off` produces an HNSW index scan with
about 0.3 ms of scan time, while JIT compilation dominates that one-off plan.
JIT is disabled on the reusable evaluation connection so it does not inflate
the reported small-query latency. At larger corpus sizes the same indexes are
the relevant path; this is why the roadmap treats index tuning as a
scale-dependent exercise rather than a 150-row benchmark.

## Failure analysis

Queries where hybrid + rerank moved the first relevant document upward:

- Q-006: rank 2 -> 1 — When will I receive my provisional credit during a transaction dispute?
- Q-011: rank 2 -> 1 — Do Apex Reward points expire if I keep my account active?

Queries where hybrid + rerank moved the first relevant document downward:

- None in this run.

The dominant error mode in this dataset is card-tier confusion: the same policy exists for several products with slightly different numbers. Metadata-aware reranking is therefore more valuable than generic semantic similarity.

A second issue is label exhaustiveness. For generic questions the judgment often lists only one or two card tiers even when another tier's policy contains the same fact. For example, DOC-0023 for Q-005 and DOC-0053 for Q-011 are semantically valid but not labeled relevant. Strict Recall/MRR therefore understates retrieval quality; it should not be tuned away by overfitting to the 30 labeled queries.

## Reproducibility and limitations

- All labels come from `data/eval_queries.json`; no judge or generated answer
  is used for the retrieval metrics.
- This is a small synthetic benchmark. Hyperparameters are global, not tuned
  per query.
- The local LSA embedding is an offline fallback, not a production embedding
  model. The same interface accepts OpenAI or BGE embeddings.
- End-to-end latency includes query embedding and reranking; database-only
  latency is reported separately in the raw JSON.
