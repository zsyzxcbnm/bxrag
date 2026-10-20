# Chunking Strategy Comparison

This comparison holds the retrieval stack fixed (dense + sparse + weighted RRF
+ heuristic rerank) and changes only the chunking strategy. It uses the
in-memory backend so every strategy runs under identical conditions.

| Strategy | Chunks | Avg tokens | Recall@1 | Recall@5 | MRR | nDCG@5 | P50 (ms) | P99 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Document-level | 150 | 47.3 | 0.817 | 0.956 | 0.967 | 0.954 | 173.47 | 181.85 |
| Fixed window | 282 | 29.9 | 0.783 | 0.944 | 0.944 | 0.928 | 173.56 | 186.65 |
| Semantic | 163 | 49.9 | 0.783 | 0.956 | 0.950 | 0.940 | 174.66 | 186.67 |

![Chunking comparison](figures/chunking_comparison.png)

## Interpretation

The provided knowledge base is already chunk-sized: the median document is
about 282 characters and contains one policy fact. Splitting it again can
separate a product name from the sentence that contains the answer, which
increases the number of candidates and weakens the signal used by the reranker.

On this dataset the fixed-window strategy loses quality because it separates
the product context from the answer sentence. Document-level and semantic
chunking are tied on quality, while semantic chunking creates more chunks and
adds pipeline complexity. Document-level chunks are therefore the simplest
defensible default. The other strategies remain in the repository so the claim
is measured rather than assumed, and so the comparison can be repeated after
the corpus grows into longer policy documents.
