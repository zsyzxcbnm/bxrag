# 重排序器对比：元数据启发式 vs BGE Cross-Encoder

## 实验配置

- Embedding：BGE-large-en-v1.5，CUDA
- 重排序候选：20
- 查询数：30，重复 3 次
- Cross-Encoder：`bge-reranker-large`，CUDA fp16，纯模型分数，不融合人工元数据
- 元数据启发式：卡种、答案类型、年份、企业/个人语境规则

| 重排序器 | Recall@1 | Recall@3 | MRR | nDCG@5 | 检索 P50 | 检索 P99 |
|---|---:|---:|---:|---:|---:|---:|
| 元数据启发式 | **0.817** | **0.944** | 0.967 | 0.964 | **8.54 ms** | **13.55 ms** |
| BGE Cross-Encoder（纯模型） | **0.817** | **0.944** | **0.983** | **0.979** | 176.02 ms | 218.71 ms |

![重排序器对比](figures/reranker_comparison_zh.png)

Cross-Encoder 的 P50 约 176ms，约为元数据启发式路径的 20 倍。

## 结论

`bge-reranker-large` 能在 4GB 显卡上以 fp16 正常加载运行，模型显存约 1.1GB。
使用纯模型分数时，它的排序质量优于元数据启发式：

- MRR 从 0.967 提升到 0.983。
- nDCG@5 从 0.964 提升到 0.979。
- 检索 P50 从约 8.5ms 升到约 176ms。

不过，无人工规则的纯 RRF 在这 30 条查询上仍然更强（Recall@1 0.850、
MRR 1.000、nDCG@5 0.994），见 `ablation_metadata.zh-CN.md`。

主要原因是卡种/年份硬过滤会移除有用候选，而 Cross-Encoder 可以更灵活地比较
query 和 passage。把 Cross-Encoder 与启发式按 65/35 简单融合后，nDCG@5 反而
降到 0.961，因此纯 Cross-Encoder 才是更合理的对照。

两个重排序器都不是默认路径。默认使用 rules-free Hybrid RRF，因为它在本次评测中
质量更高、延迟更低。Cross-Encoder 作为可选能力保留，适合更大规模、更偏语义的
语料，也证明流水线可以接入真正的第二阶段模型。

## 复现

```powershell
$env:RERANKER_PROVIDER='cross-encoder'
$env:RERANKER_MODEL='data\models\bge-reranker-large'
$env:RERANKER_DEVICE='cuda'
$env:RERANKER_FP16='true'

& '.\.venv\Scripts\python.exe' -m src.cli eval `
  --dense-weight 0.5 --sparse-weight 0.5 --repeats 3
```

