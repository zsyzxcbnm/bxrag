# LLM-as-a-Judge Evaluation

Judge model: `data\models\Qwen2.5-0.5B-Instruct`

## Summary

- Queries: **30**
- Context relevance (1-5): **4.000**
- Faithfulness (1-5): **3.069**
- JSON parse errors: **2**

## Per-query results

| Query | Context relevance | Faithfulness | Retrieved docs |
|---|---:|---:|---|
| Q-001 | 4 | 4 | DOC-0001, DOC-0005, DOC-0002, DOC-0003, DOC-0004 |
| Q-002 | 4 | 4 | DOC-0003, DOC-0014, DOC-0037, DOC-0033, DOC-0006 |
| Q-003 | 4 | 4 | DOC-0005, DOC-0001, DOC-0013, DOC-0009, DOC-0004 |
| Q-004 | 4 | 5 | DOC-0016, DOC-0018, DOC-0019, DOC-0017, DOC-0030 |
| Q-005 | 4 | 4 | DOC-0020, DOC-0023, DOC-0022, DOC-0021, DOC-0024 |
| Q-006 | 4 | 4 | DOC-0026, DOC-0024, DOC-0027, DOC-0020, DOC-0025 |
| Q-007 | 4 | 2 | DOC-0031, DOC-0032, DOC-0035, DOC-0039, DOC-0033 |
| Q-008 | 4 | 2 | DOC-0036, DOC-0035, DOC-0032, DOC-0037, DOC-0038 |
| Q-009 | 4 | 4 | DOC-0040, DOC-0041, DOC-0147, DOC-0039, DOC-0044 |
| Q-010 | 4 | 2 | DOC-0046, DOC-0049, DOC-0050, DOC-0035, DOC-0060 |
| Q-011 | 4 | 4 | DOC-0050, DOC-0052, DOC-0051, DOC-0053, DOC-0086 |
| Q-012 | 4 | 1 | DOC-0054, DOC-0057, DOC-0056, DOC-0101, DOC-0065 |
| Q-013 | 4 | 5 | DOC-0061, DOC-0062, DOC-0063, DOC-0054, DOC-0035 |
| Q-014 | 4 | 4 | DOC-0065, DOC-0066, DOC-0067, DOC-0106, DOC-0114 |
| Q-015 | 4 | 2 | DOC-0065, DOC-0114, DOC-0115, DOC-0066, DOC-0106 |
| Q-016 | 4 | 4 | DOC-0031, DOC-0076, DOC-0035, DOC-0077, DOC-0084 |
| Q-017 | ERR | 2 | DOC-0077, DOC-0033, DOC-0037, DOC-0044, DOC-0041 |
| Q-018 | 4 | 1 | DOC-0094, DOC-0095, DOC-0096, DOC-0091, DOC-0102 |
| Q-019 | 4 | 2 | DOC-0058, DOC-0059, DOC-0060, DOC-0031, DOC-0050 |
| Q-020 | 4 | 1 | DOC-0099, DOC-0091, DOC-0100, DOC-0097, DOC-0046 |
| Q-021 | 4 | 2 | DOC-0037, DOC-0035, DOC-0033, DOC-0036, DOC-0038 |
| Q-022 | 4 | 4 | DOC-0043, DOC-0035, DOC-0031, DOC-0001, DOC-0039 |
| Q-023 | 4 | 2 | DOC-0057, DOC-0030, DOC-0126, DOC-0056, DOC-0102 |
| Q-024 | 4 | 2 | DOC-0073, DOC-0114, DOC-0065, DOC-0031, DOC-0074 |
| Q-025 | 4 | 4 | DOC-0024, DOC-0027, DOC-0025, DOC-0020, DOC-0026 |
| Q-026 | 4 | ERR | DOC-0061, DOC-0043, DOC-0101, DOC-0062, DOC-0009 |
| Q-027 | 4 | 4 | DOC-0065, DOC-0106, DOC-0110, DOC-0024, DOC-0108 |
| Q-028 | 4 | 4 | DOC-0149, DOC-0033, DOC-0044, DOC-0041, DOC-0037 |
| Q-029 | 4 | 2 | DOC-0060, DOC-0032, DOC-0036, DOC-0040, DOC-0107 |
| Q-030 | 4 | 4 | DOC-0039, DOC-0138, DOC-0001, DOC-0009, DOC-0150 |

## Method

For each query, the evaluator:

1. retrieves the top-5 passages with the rules-free hybrid RRF pipeline;
2. asks the judge model to rate whether the passages answer the question;
3. asks the judge model to answer using only those passages;
4. asks the judge model whether every claim in that answer is supported.

The judge is intentionally separate from the retrieval metrics and is not used
to tune the retriever.
