# 重排序器对比：元数据启发式 vs BGE Cross-Encoder

## 实验配置

- Embedding：BGE-large-en-v1.5，CUDA
- 重排序候选：20
- 查询数：30，重复 3 次
- Cross-Encoder：`bge-reranker-large`，CUDA fp16，分数按 65% Cross-Encoder +
  35% 元数据启发式融合
- 元数据启发式：卡种、答案类型、年份、企业/个人语境规则

| 重排序器 | Recall@1 | Recall@3 | MRR | nDCG@5 | 检索 P50 | 检索 P95 | 检索 P99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 元数据启发式 | **0.817** | **0.944** | **0.967** | **0.964** | **7.68 ms** | **9.34 ms** | **10.36 ms** |
| BGE Cross-Encoder | 0.817 | 0.933 | 0.967 | 0.961 | 162.86 ms | 183.78 ms | 849.36 ms |

![重排序器对比](figures/reranker_comparison_zh.png)

Cross-Encoder 的 P99 包含冷启动尖峰；即使预热后，P50 仍约 160ms，约为启发式
路径的 20 倍。

## 结论

`bge-reranker-large` 能在 4GB 显卡上以 fp16 正常加载运行，模型显存约 1.1GB。
它具备语义重排序能力，但在这份合成信用卡政策数据上没有超过元数据启发式：

- nDCG@5 从 0.964 降到 0.961。
- Recall@3 从 0.944 降到 0.933。
- 检索 P50 从约 8ms 升到约 163ms。

主要原因是这份数据集的难点是卡种和答案类型区分，而不是通用语义相似度；启发式
显式利用了这些约束。

因此默认仍使用启发式重排序。Cross-Encoder 作为可选能力保留，适合更大规模、
更偏语义的语料，也证明流水线可以接入真正的第二阶段模型。

## 复现

```powershell
$env:RERANKER_PROVIDER='cross-encoder'
$env:RERANKER_MODEL='G:\博贤\data\models\bge-reranker-large'
$env:RERANKER_DEVICE='cuda'
$env:RERANKER_FP16='true'

& 'G:\博贤\.venv-gpu\Scripts\python.exe' -m src.cli eval `
  --dense-weight 0.5 --sparse-weight 0.5 --repeats 3
```

