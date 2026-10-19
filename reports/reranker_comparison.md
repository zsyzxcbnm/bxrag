# Reranker Comparison: Metadata Heuristic vs BGE Cross-Encoder

## Setup

- Embedding: BGE-large-en-v1.5, CUDA
- Rerank candidates: 20
- Queries: 30, repeats: 3
- Cross-Encoder: `bge-reranker-large`, CUDA fp16, blended 65% Cross-Encoder +
  35% metadata heuristic
- Metadata heuristic: card-tier, answer-type, year and business-context rules

| Reranker | Recall@1 | Recall@3 | MRR | nDCG@5 | Retrieval P50 | Retrieval P95 | Retrieval P99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Metadata heuristic | **0.817** | **0.944** | **0.967** | **0.964** | **7.68 ms** | **9.34 ms** | **10.36 ms** |
| BGE Cross-Encoder | 0.817 | 0.933 | 0.967 | 0.961 | 162.86 ms | 183.78 ms | 849.36 ms |

![Reranker comparison](figures/reranker_comparison_zh.png)

The Cross-Encoder P99 includes a cold-start spike; even after warm-up its P50 is
about 160 ms, roughly 20x the heuristic path.

## Conclusion

`bge-reranker-large` loads and runs correctly on the 4GB GPU in fp16 (about
1.1GB of model memory). It can produce semantic relevance scores, but on this
synthetic credit-card policy benchmark it does not beat the metadata-aware
heuristic:

- nDCG@5 drops from 0.964 to 0.961.
- Recall@3 drops from 0.944 to 0.933.
- Retrieval P50 rises from about 8 ms to about 163 ms.

The likely reason is that the benchmark's hardest cases are card-tier and
answer-type distinctions, not general semantic similarity. The heuristic
explicitly encodes those constraints.

The production default therefore remains the heuristic reranker. The
Cross-Encoder is kept as an optional provider for larger or more semantic
corpora, and as evidence that the pipeline can integrate a real second-stage
model.

## Reproduce

```powershell
$env:RERANKER_PROVIDER='cross-encoder'
$env:RERANKER_MODEL='G:\博贤\data\models\bge-reranker-large'
$env:RERANKER_DEVICE='cuda'
$env:RERANKER_FP16='true'

& 'G:\博贤\.venv-gpu\Scripts\python.exe' -m src.cli eval `
  --dense-weight 0.5 --sparse-weight 0.5 --repeats 3
```

