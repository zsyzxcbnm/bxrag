# Apex RAG Retrieval Evaluation

Generated at `2026-10-19T11:05:57.465439+00:00`.

## Setup

- Queries: **30**
- Embedding: `local-tfidf-svd` (128 dimensions)
- Embedding provider: `local`
- Reranker: `heuristic`
- RRF `k`: **60**
- RRF weights: dense **0.4**, sparse **0.6**
- Rerank candidates: **20**

The default offline configuration uses a local TF-IDF + SVD embedding and a
metadata-aware heuristic reranker so the evaluation is reproducible without
network access or API keys. The OpenAI embedding and Sentence-Transformers
Cross-Encoder providers can be enabled through `.env`.

## Results

| Pipeline | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR | nDCG@5 | P50 (ms) | P99 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.761 | 0.900 | 0.933 | 0.260 | 0.917 | 0.902 | 2.62 | 3.92 |
| Sparse only | 0.794 | 0.900 | 0.933 | 0.260 | 0.933 | 0.917 | 2.98 | 4.22 |
| Hybrid (RRF) | 0.794 | 0.889 | 0.933 | 0.260 | 0.933 | 0.916 | 3.52 | 5.98 |
| Hybrid + rerank | 0.817 | 0.911 | 0.933 | 0.260 | 0.967 | 0.934 | 4.89 | 6.43 |

![Retrieval quality](figures/metrics_comparison.png)

![Latency](figures/latency_percentiles.png)

## What the comparison shows

- Dense-only is the required baseline; sparse-only isolates the value of exact
  lexical matching on policy numbers, product names and time limits.
- Hybrid retrieval uses Reciprocal Rank Fusion over dense and PostgreSQL
  full-text candidates.
- Hybrid + rerank applies a second-stage scorer to the top candidates and
  returns the final top 5.
- Recall@10 is deliberately not the headline metric: with 150 short documents
  and multi-label relevance it saturates quickly. Recall@1, MRR and nDCG@5 are
  the discriminating metrics.

## Failure analysis

Queries where hybrid + rerank moved the first relevant document upward:

- Q-006: rank 2 -> 1 — When will I receive my provisional credit during a transaction dispute?
- Q-011: rank 2 -> 1 — Do Apex Reward points expire if I keep my account active?
- Q-016: rank 2 -> 1 — What is the fee for adding authorized users to an Apex Platinum card?

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
