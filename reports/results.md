# Apex RAG Retrieval Evaluation

Generated at `2026-10-19T12:08:10.860736+00:00`.

## Setup

- Queries: **30**
- Storage backend: `postgres`
- Embedding: `G:\博贤\data\models\Qwen3-Embedding-0.6B` (1024 dimensions)
- Embedding provider: `sentence-transformers`
- Embedding device: `cpu`
- Reranker: `heuristic`
- RRF `k`: **60**
- RRF weights: dense **0.7**, sparse **0.3**
- Rerank candidates: **20**

The configured embedding provider and metadata-aware heuristic reranker are
recorded above so the run is reproducible. The OpenAI embedding and
Sentence-Transformers Cross-Encoder providers can be enabled through `.env`.

## Results

| Pipeline | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR | nDCG@5 | P50 (ms) | P99 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.817 | 0.900 | 0.956 | 0.273 | 0.967 | 0.944 | 44.23 | 53.89 |
| Sparse only | 0.750 | 0.922 | 0.967 | 0.280 | 0.933 | 0.935 | 2.58 | 4.35 |
| Hybrid (RRF) | 0.817 | 0.922 | 0.967 | 0.280 | 0.967 | 0.955 | 47.90 | 55.10 |
| Hybrid + rerank | 0.817 | 0.933 | 0.967 | 0.280 | 0.967 | 0.961 | 51.04 | 53.76 |

The table's P50/P99 columns measure the retrieval path (query embedding excluded):
vector search, sparse search, fusion and reranking. Query-embedding latency is
reported separately because it depends on the model and device:

| Pipeline | Embedding P50 (ms) | Embedding P99 (ms) |
|---|---:|---:|
| Dense only | 428.69 | 491.10 |
| Sparse only | 0.00 | 0.00 |
| Hybrid (RRF) | 397.74 | 479.81 |
| Hybrid + rerank | 393.78 | 460.97 |

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

- None in this run.

Queries where hybrid + rerank moved the first relevant document downward:

- None in this run.

The dominant error mode in this dataset is card-tier confusion: the same policy exists for several products with slightly different numbers. Metadata-aware reranking is therefore more valuable than generic semantic similarity.

A second issue is label exhaustiveness. For generic questions the judgment often lists only one or two card tiers even when another tier's policy contains the same fact. For example, DOC-0023 for Q-005 and DOC-0053 for Q-011 are semantically valid but not labeled relevant. Strict Recall/MRR therefore understates retrieval quality; it should not be tuned away by overfitting to the 30 labeled queries.

## Reproducibility and limitations

- All labels come from `data/eval_queries.json`; no judge or generated answer
  is used for the retrieval metrics.
- This is a small synthetic benchmark. Hyperparameters are global, not tuned
  per query.
- Dense quality depends on the selected embedding model. The local LSA
  embedding is an offline fallback; Qwen3/OpenAI embeddings can be selected
  through the provider configuration.
- End-to-end latency includes query embedding and reranking; database-only
  latency is reported separately in the raw JSON.
