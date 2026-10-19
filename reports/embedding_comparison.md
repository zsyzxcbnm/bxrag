# Embedding Comparison: TF-IDF + SVD vs Qwen3-Embedding-0.6B

Both runs use the same 150 documents, 30 labeled queries, PostgreSQL +
pgvector, weighted RRF and the heuristic reranker. Only the embedding model
changes.

| Pipeline | TF-IDF Recall@1 | Qwen3 Recall@1 | TF-IDF MRR | Qwen3 MRR | TF-IDF nDCG@5 | Qwen3 nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| Dense only | 0.761 | **0.817** | 0.917 | **0.967** | 0.902 | **0.944** |
| Sparse only | 0.750 | 0.750 | 0.933 | 0.933 | 0.935 | 0.935 |
| Hybrid + RRF | 0.783 | **0.817** | 0.950 | **0.967** | 0.941 | **0.955** |
| Hybrid + rerank | 0.783 | **0.817** | 0.950 | **0.967** | 0.950 | **0.961** |

## Conclusion

Qwen3-Embedding-0.6B clearly improves the dense path:

- Dense Recall@1: 0.761 -> 0.817
- Dense MRR: 0.917 -> 0.967
- Dense nDCG@5: 0.902 -> 0.944
- Hybrid + rerank nDCG@5: 0.950 -> 0.961

The sparse path is unchanged because it uses PostgreSQL full-text search, not
the embedding model. With Qwen3, dense retrieval alone is already competitive
with the hybrid stack; the best result uses dense weight 0.7 and sparse weight
0.3.

## Cost

On the current CPU-only PyTorch environment:

- PostgreSQL retrieval path: about 8 ms P50 / 12 ms P99.
- Qwen3 query embedding: about 387 ms P50 / 460 ms P99.
- The first model load takes roughly 14 seconds.

For production, the embedding service should run on CUDA, ONNX Runtime or a
dedicated TEI service. The retrieval path itself remains well below 100 ms.

## Reproduce

```powershell
python -m src.cli reset-db --yes
python -m src.cli ingest
python -m src.cli eval --dense-weight 0.7 --sparse-weight 0.3
```

Raw results:

- `reports/results_tfidf_pgvector.md`
- `reports/results_qwen3_pgvector.md`
