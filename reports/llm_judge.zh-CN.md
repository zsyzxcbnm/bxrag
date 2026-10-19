# LLM-as-a-Judge 评估（中文版）

本地 Judge 模型：`Qwen2.5-0.5B-Instruct`

## 汇总

- 查询数：**30**
- Context Relevance 平均分：**4.000 / 5**
- Faithfulness 平均分：**3.069 / 5**
- JSON 解析失败：**2 / 60 次 Judge 调用**

## 方法

对每条查询：

1. 用规则无关的 Hybrid RRF 取 Top-5 证据；
2. 让 Judge 模型评估证据是否足以回答问题；
3. 让 Judge 模型只基于证据生成简短答案；
4. 让 Judge 模型检查答案中的每个事实是否被证据支持。

Judge 与检索指标完全分开，不参与检索参数调优。

## 结论与限制

- 本地 0.5B Judge 能稳定输出结构化 JSON，60 次调用中 58 次成功解析。
- Context Relevance 几乎全部落在 4 分，区分度很低，说明小模型存在明显评分饱和度。
- Faithfulness 平均 3.07，部分答案包含过度概括或错误数字，例如把“60 天申请窗口”与
  “15 个账单周期”混在一起。
- 2 次 JSON 解析失败（Q-017、Q-026），已在报告中标记为 `ERR`，没有伪造分数。

这份结果适合作为**离线 Judge 流程的冒烟测试和工程验证**。正式上线应使用更大的本地模型
或经过校准的云端 Judge，并抽样与人工评分做一致性检查。

## 相关文件

- [llm_judge.md](llm_judge.md)
- [llm_judge.json](llm_judge.json)
