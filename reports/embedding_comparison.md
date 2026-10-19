# Embedding Comparison

All runs use the same 150 documents, 30 labeled queries, PostgreSQL + pgvector,
weighted RRF and the heuristic reranker. Only the embedding model (and its
recommended fusion weights) changes.

| Pipeline | TF-IDF R@1 | BGE R@1 | Qwen3 R@1 | TF-IDF MRR | BGE MRR | Qwen3 MRR | TF-IDF nDCG@5 | BGE nDCG@5 | Qwen3 nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.761 | 0.772 | **0.817** | 0.917 | 0.922 | **0.967** | 0.902 | 0.914 | **0.944** |
| Sparse only | 0.750 | 0.750 | 0.750 | 0.933 | 0.933 | 0.933 | 0.935 | 0.935 | 0.935 |
| Hybrid + RRF | 0.783 | 0.817 | 0.817 | 0.950 | **0.967** | **0.967** | 0.941 | **0.960** | 0.955 |
| Hybrid + rerank | 0.783 | 0.817 | 0.817 | 0.950 | **0.967** | **0.967** | 0.950 | **0.964** | 0.961 |

## Conclusion

Both neural embeddings beat the TF-IDF baseline. BGE-large and Qwen3-0.6B are
effectively tied on Recall@1 and MRR after hybrid reranking; BGE has a small
edge on nDCG@5 (0.964 vs 0.961).

The important tradeoff is latency:

- BGE query embedding: about **175 ms** P50 on CPU.
- Qwen3 query embedding: about **387 ms** P50 / **460 ms** P99 on CPU.
- Both retrieval paths are about 8-12 ms P50/P99.

## Cost

On the current CPU-only PyTorch environment:

- PostgreSQL retrieval path: about 9 ms P50 / 13 ms P99.
- BGE query embedding: about 175 ms P50.
- Qwen3 query embedding: about 387 ms P50 / 460 ms P99.
- The first model load takes roughly 14 seconds.

For production, the embedding service should run on CUDA, ONNX Runtime or a
dedicated TEI service. The retrieval path itself remains well below 100 ms.

## Reproduce

```powershell
python -m src.cli reset-db --yes
python -m src.cli ingest
python -m src.cli eval --dense-weight 0.5 --sparse-weight 0.5  # BGE
python -m src.cli eval --dense-weight 0.7 --sparse-weight 0.3  # Qwen3
```

Raw results:

- `reports/results_tfidf_pgvector.md`
- `reports/results_bge_large_pgvector.md`
- `reports/results_qwen3_pgvector.md`
