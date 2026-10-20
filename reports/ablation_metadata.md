# Metadata Rule Ablation

This experiment separates three effects:

- `no_metadata_rrf`: pure dense + sparse hybrid retrieval and RRF, with no
  card-tier/year filters and no manual reranker.
- `no_metadata_cross_encoder`: the same retrieval, followed by the pure BGE
  Cross-Encoder without heuristic blending.
- `metadata_heuristic`: the project default, with card-tier/year filters and
  the metadata-aware heuristic reranker.

| Pipeline | Recall@1 | Recall@3 | MRR | nDCG@5 | Retrieval P50 | Retrieval P99 |
|---|---:|---:|---:|---:|---:|---:|
| no_metadata_rrf | 0.850 | 0.956 | 1.000 | 0.994 | 8.16 ms | 14.03 ms |
| no_metadata_cross_encoder | 0.817 | 0.944 | 0.983 | 0.979 | 168.57 ms | 180.42 ms |
| metadata_heuristic | 0.817 | 0.944 | 0.967 | 0.964 | 7.39 ms | 10.01 ms |

The metadata rules are dataset-informed, so the default score should not be
treated as a clean holdout. This table makes the optimism explicit and shows
the independent Cross-Encoder control.
