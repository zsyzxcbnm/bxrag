# Embedding 方案对比（中文版）

三组实验使用相同的 150 条文档、30 条查询、PostgreSQL/pgvector、加权 RRF 和
启发式重排序，只替换 Embedding 模型及推荐融合权重。

## 检索质量

| 管线 | 指标 | TF-IDF + SVD | BGE-large | Qwen3-0.6B |
|---|---|---:|---:|---:|
| 仅稠密检索 | Recall@1 | 0.761 | 0.772 | **0.817** |
| 仅稠密检索 | MRR | 0.917 | 0.922 | **0.967** |
| 仅稠密检索 | nDCG@5 | 0.902 | 0.914 | **0.944** |
| 混合检索+重排序 | Recall@1 | 0.783 | **0.817** | **0.817** |
| 混合检索+重排序 | MRR | 0.950 | **0.967** | **0.967** |
| 混合检索+重排序 | nDCG@5 | 0.950 | **0.964** | 0.961 |

![Embedding 对比](figures/embedding_comparison_zh.png)

## 延迟

| 方案 | Query Embedding P50 | 检索路径 P50 | 近似端到端 P50 |
|---|---:|---:|---:|
| TF-IDF + SVD | 约 2 ms | 约 9 ms | 约 11 ms |
| BGE-large CPU | 约 175 ms | 约 9 ms | 约 184 ms |
| BGE-large CUDA | **约 25 ms** | 约 8 ms | **约 32 ms** |
| Qwen3-0.6B CPU | 约 387 ms | 约 8 ms | 约 395 ms |

## 结论

- 两个神经 Embedding 都明显优于 TF-IDF 基线。
- BGE 和 Qwen3 在混合检索+重排序后的 Recall@1/MRR 基本打平。
- BGE 的 nDCG@5 略高（0.964 vs 0.961），并且 GPU 推理约快 15 倍（25ms vs 387ms）。
- Qwen3 的 Dense-only 更强，适合多语言、中文或更依赖语义相似度的场景。
- 当前英文信用卡政策库，推荐 **BGE-large + CUDA**：质量好、延迟低、显存占用约 1.3GB。

英文版见 [embedding_comparison.md](embedding_comparison.md)。

