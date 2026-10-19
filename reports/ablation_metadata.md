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
| no_metadata_rrf | 0.850 | 0.956 | 1.000 | 0.994 | 8.29 ms | 12.82 ms |
| no_metadata_cross_encoder | 0.817 | 0.944 | 0.983 | 0.979 | 170.62 ms | 182.04 ms |
| metadata_heuristic | 0.817 | 0.944 | 0.967 | 0.964 | 7.54 ms | 10.55 ms |

The metadata rules are dataset-informed, so the default score should not be
treated as a clean holdout. This table makes the optimism explicit and shows
the independent Cross-Encoder control.

## Conclusion

Pure RRF is the strongest pipeline on this 30-query benchmark: Recall@1 0.850,
MRR 1.000 and nDCG@5 0.994. The card-tier/year hard filters and the metadata
heuristic both remove useful candidates on this small, cross-tier corpus.

The pure BGE Cross-Encoder recovers ranking quality over the heuristic
(MRR 0.983 vs 0.967, nDCG@5 0.979 vs 0.964) but is much slower. It still does
not beat pure RRF.

Recommendation: treat metadata as a soft signal or a fallback filter, not a
default hard filter. Keep the heuristic and Cross-Encoder as optional re-rankers,
and report pure RRF as the clean, rules-free baseline.
