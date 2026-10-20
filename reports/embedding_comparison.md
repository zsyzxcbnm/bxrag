# Embedding Comparison

All runs use the same 150 documents, 30 labeled queries, PostgreSQL + pgvector,
0.5/0.5 RRF weights, metadata rules disabled, and ten retrieved documents per
query. Only the embedding model changes.

| Embedding | Dense R@1 | Dense MRR | Hybrid R@1 | Hybrid MRR | Hybrid nDCG@5 | Rerank R@1 | Rerank MRR | Rerank nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + SVD | 0.694 | 0.894 | 0.750 | 0.950 | 0.939 | 0.772 | 0.950 | 0.943 |
| BGE-large-en-v1.5 | 0.806 | 0.956 | **0.850** | **1.000** | **0.994** | **0.850** | **1.000** | 0.991 |
| Qwen3-Embedding-0.6B | **0.817** | **0.983** | 0.817 | 0.983 | 0.981 | **0.850** | **1.000** | 0.991 |

## Conclusion

Both neural embeddings beat the TF-IDF baseline. BGE-large is strongest on the
default rules-free Hybrid RRF path (nDCG@5 0.994 vs 0.981 for Qwen3). Qwen3
has the best dense-only ranking. After the optional heuristic reranker, the two
neural models tie on Recall@1, MRR and nDCG@5 (both 0.991 nDCG@5).

Latency on the RTX 3050 Ti:

| Embedding | Query embedding P50 | Hybrid retrieval P50 | Approx. end-to-end P50 |
|---|---:|---:|---:|
| TF-IDF + SVD | 2.61 ms | 7.25 ms | 9.86 ms |
| BGE-large CUDA | 27.34 ms | 8.10 ms | 35.44 ms |
| Qwen3-0.6B CUDA | 69.57 ms | 9.93 ms | 79.50 ms |

BGE-large is the recommended neural default for this English corpus: it has
the best default retrieval quality and materially lower embedding latency than
Qwen3 on the same GPU.

## Reproduce

```powershell
python -m src.cli reset-db --yes
python -m src.cli ingest
python -m src.cli eval --dense-weight 0.5 --sparse-weight 0.5
```

Raw results:

- `reports/results_tfidf_pgvector.md`
- `reports/results_bge_large_pgvector.md`
- `reports/results_qwen3_pgvector.md`
