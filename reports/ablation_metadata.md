# Metadata Rule Ablation

This experiment separates three effects:

- `no_metadata_rrf`: pure dense + sparse hybrid retrieval and RRF, with no
  card-tier/year filters and no manual reranker.
- `no_metadata_cross_encoder`: the same retrieval, followed by the pure BGE
  Cross-Encoder without heuristic blending.
- `metadata_heuristic`: an opt-in configuration with card-tier/year filters
  and the metadata-aware heuristic reranker.

| Pipeline | Recall@1 | Recall@3 | Recall@10 | MRR | nDCG@5 | nDCG@10 | Retrieval P50 | Retrieval P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_metadata_rrf | 0.850 | 0.956 | 1.000 | 1.000 | 0.994 | 0.994 | 8.12 ms | 10.24 ms |
| no_metadata_cross_encoder | 0.817 | 0.944 | 1.000 | 0.983 | 0.979 | 0.979 | 176.02 ms | 218.71 ms |
| metadata_heuristic | 0.817 | 0.944 | 0.967 | 0.967 | 0.964 | 0.964 | 8.54 ms | 13.55 ms |

The metadata rules are dataset-informed and are not the project default. This
table makes their performance explicit and compares them with the rules-free
RRF and Cross-Encoder controls. Every row retrieves ten documents; @5 and @10
metrics are derived from the same ranking.
