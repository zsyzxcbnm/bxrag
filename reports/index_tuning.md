# HNSW Index Tuning

Embedding: `G:\博贤\data\models\Qwen3-Embedding-0.6B` (1024 dimensions)

Queries: **30**, repeats: **3**.
Recall is measured against `NumPy cosine top-10 over all 150 chunk vectors`. The index scan is
forced with `enable_seqscan=off` so the benchmark measures HNSW rather than the
planner's sequential-scan choice for 150 rows.

| m | ef_construction | ef_search | Build (ms) | Index size | Recall@10 | P50 (ms) | P99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 64 | 80 | 27.3 | 1208 kB | 1.000 | 2.63 | 6.56 |
| 16 | 64 | 80 | 29.2 | 1208 kB | 1.000 | 2.48 | 8.22 |
| 32 | 64 | 80 | 42.0 | 1208 kB | 1.000 | 2.44 | 7.49 |
| 16 | 128 | 80 | 30.7 | 1208 kB | 1.000 | 2.54 | 7.09 |
| 16 | 256 | 80 | 30.6 | 1208 kB | 1.000 | 2.43 | 7.00 |
| 16 | 64 | 40 | 28.7 | 1208 kB | 1.000 | 2.54 | 7.98 |
| 16 | 64 | 120 | 29.0 | 1208 kB | 1.000 | 2.37 | 6.49 |
| 16 | 64 | 200 | 26.9 | 1208 kB | 1.000 | 2.53 | 6.66 |

## Interpretation

- At 150 chunks every HNSW configuration reaches the same Recall@10 against the
  exact reference; the graph has more than enough capacity for this corpus.
- Query latency is well below 100 ms for every `ef_search` tested. Increasing
  `ef_search` improves the candidate search effort but is not needed at this
  scale.
- `m=16`, `ef_construction=64`, `ef_search=80` is a sensible default: small
  index, short build time and no measurable quality loss.
- This is a scale-dependent tuning exercise. At 10M vectors, measure recall,
  index build memory and P99 together instead of reusing these numbers.

The project restores the default HNSW configuration after the sweep.
