# Apex RAG Retrieval Engine

A reproducible hybrid retrieval and evaluation harness for the Apex Card Services
credit-card support knowledge base. The project is intentionally retrieval-first:
the benchmark measures whether the right policy document is found, not whether a
chat model sounds convincing.

## For reviewers

Run commands from the repository root after cloning. The shortest verification
is:

```powershell
git clone https://github.com/zsyzxcbnm/bxrag.git
cd bxrag
powershell -ExecutionPolicy Bypass -File .\scripts\reproduce.ps1
```

See [REVIEW.md](REVIEW.md) for code checks, full BGE GPU, and Docker/API
verification.

Chinese documentation: [README.zh-CN.md](README.zh-CN.md) ·
[战略架构路线图](docs/roadmap.zh-CN.md) ·
[中文评测报告](reports/results.zh-CN.md) ·
[中文图表](reports/figures/metrics_comparison_zh.png).

## What is included

- PostgreSQL + `pgvector` storage with HNSW and full-text indexes.
- Document-level, fixed-window and sentence-aware chunking strategies.
- Dense retrieval, PostgreSQL `tsvector` retrieval, weighted Reciprocal Rank
  Fusion and an optional second-stage reranker.
- Offline and optional API-backed embedding/reranker providers.
- Recall@K, Precision@K, MRR, nDCG@K and P50/P95/P99 latency evaluation.
- Markdown report, CSV raw results and PNG charts.
- A memory backend that mirrors the database interface for offline validation.

Current PostgreSQL + pgvector run with local BGE-large-en-v1.5, ten results per
query, metadata rules disabled, and 0.5/0.5 dense/sparse RRF weights:

- Dense-only: Recall@1 **0.806**, MRR **0.956**.
- Default rules-free Hybrid RRF: Recall@1 **0.850**, MRR **1.000**, nDCG@5
  **0.994**, retrieval P50 **8.10 ms**, P99 **10.28 ms**.
- Optional Hybrid + heuristic rerank: Recall@1 **0.850**, MRR **1.000**, nDCG@5
  **0.991**, retrieval P50 **9.55 ms**, P99 **12.57 ms**.

GPU query embedding adds about **27-29 ms** P50. Approximate end-to-end latency
is **35.44 ms P50 / 41.20 ms P99** for the default Hybrid RRF path and
**38.71 ms P50 / 46.84 ms P99** for Hybrid + rerank. The HNSW sweep reports
2.5 ms P50 / 7-8 ms P99. See [reports/results.md](reports/results.md) and
[reports/chunking.md](reports/chunking.md).

Rules-free RRF is the default because the metadata rules are dataset-informed
and do not generalize cleanly: the ablation still puts RRF ahead of the
metadata heuristic (Recall@1 0.850 vs 0.817, MRR 1.000 vs 0.967). Metadata
filters and heuristic bonuses remain available as an opt-in comparison in
[reports/ablation_metadata.md](reports/ablation_metadata.md).

All rows below use the default 0.5/0.5 RRF weights, metadata rules disabled,
and the same ten-result evaluation. The neural rows run on CUDA (RTX 3050 Ti):

| Embedding | RRF dense/sparse | Dense R@1 | Dense MRR | Hybrid R@1 | Hybrid MRR | Hybrid nDCG@5 | Rerank R@1 | Rerank MRR | Rerank nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + SVD baseline | 0.5/0.5 | 0.694 | 0.894 | 0.750 | 0.950 | 0.939 | 0.772 | 0.950 | 0.943 |
| BGE-large-en-v1.5 | 0.5/0.5 | 0.806 | 0.956 | **0.850** | **1.000** | **0.994** | **0.850** | **1.000** | 0.991 |
| Qwen3-Embedding-0.6B | 0.5/0.5 | 0.817 | 0.983 | 0.817 | 0.983 | 0.981 | **0.850** | **1.000** | 0.991 |

The per-model reports record their exact provider, device and weights:
[TF-IDF](reports/results_tfidf_pgvector.md) ·
[BGE](reports/results_bge_large_pgvector.md) ·
[Qwen3](reports/results_qwen3_pgvector.md).

## Architecture

```text
data/kb_documents.json
        |
        v
  normalize + metadata
        |
        v
  document / fixed / sentence-aware chunks
        |
        v
  embedder (local LSA, BGE, Qwen3 or OpenAI)
        |
        v
  PostgreSQL + pgvector  <-->  PostgreSQL FTS (tsvector)
        |                              |
        +-------- dense Top-50 --------+
                       |
                       v
                  weighted RRF
                       |
             +---------+---------+
             |                   |
             v                   v
      Top-5 evidence     optional reranker
         (default)        Cross-Encoder
                              |
                              v
                       Top-5 evidence
```

## Quick start with PostgreSQL

Requirements: Python 3.11+, Docker Desktop (or another Docker daemon).

```powershell
docker compose up -d db
.\scripts\setup.ps1
.\.venv\Scripts\python.exe -m src.cli init-db
.\.venv\Scripts\python.exe -m src.cli ingest
.\.venv\Scripts\python.exe -m src.cli eval
```

Or, after the database is running:

```powershell
python -m src.cli all
```

The default connection string is in `.env.example`:

```text
postgresql://apex:apex@localhost:5432/apex
```

Copy it to `.env` only if you need to override the defaults.

## Offline quick start

If Docker is unavailable, the same retrieval and evaluation logic runs against
an in-memory backend with BM25-style sparse scoring:

```powershell
python -m src.cli all --backend memory
```

This path is useful for CI and for reviewing the metrics without starting a
database. The PostgreSQL path remains the production-oriented implementation.

## Configuration and secrets

- `.env.example` is a development template (local Docker credentials).
- `.env.production.example` is a production template with placeholders.
- A real `.env`, database password, API key, or private model path must never
  be committed. Inject them through the deployment platform or a secret
  manager.
- `pydantic-settings` loads configuration from environment variables and
  `.env`; paths and model locations are configurable rather than hard-coded in
  the Python source.
- The CLI reports expected application errors through a single boundary and
  exits non-zero without dumping a full traceback.

## One-command reproduction

### Offline: no Docker, no GPU, no downloaded embedding model

```powershell
.\scripts\reproduce.ps1
```

Equivalent Make target (if `make` is installed):

```text
make reproduce
```

This creates/uses `.venv`, runs the 150-document ingestion against the
in-memory BM25 + cosine backend, runs the four retrieval pipelines, runs the
chunking comparison, and writes Chinese charts to `reports/offline/`.

### Full PostgreSQL + BGE GPU run

```powershell
.\scripts\reproduce.ps1 -Profile bge-gpu
```

Equivalent Make target:

```text
make reproduce-gpu
```

The full static/test check is:

```text
make verify
```

The Docker deployment path is also wrapped:

```text
make docker-build
make docker-ingest
make docker-api
```

The optional judge model is downloaded with `make download-judge`.

This starts the pgvector container, rebuilds the 1024-dimensional BGE index,
runs evaluation, HNSW parameter sweep, metadata-rule ablation, and regenerates
the Chinese charts in `reports/`. Missing models are downloaded automatically
on the first run (about 5GB; network required). Docker and an NVIDIA GPU are
required for this profile.

Both commands use `.venv` and the project-local cache/tmp directories.
`requirements.txt` is a convenience wrapper around the canonical
`pyproject.toml` dependency metadata.

The models are intentionally not committed to Git. `docker compose up -d` only
starts PostgreSQL; it does not import data or run evaluation. The one-command
scripts are the correct entry points.

## Embedding and reranker providers

The default is deliberately offline and reproducible:

- Embedding: local TF-IDF + Truncated SVD, 128 dimensions.
- Retrieval: weighted RRF over dense and sparse candidates, with metadata
  rules disabled.
- Reranker: optional. The serving default is `hybrid` (RRF only);
  `hybrid_rerank` enables the heuristic or Cross-Encoder stage.

Metadata filters and heuristic bonuses are opt-in. Pass `use_metadata=true` to
the API, or `--use-metadata` to `python -m src.cli search`, together with
`mode=hybrid_rerank`. They are evaluated separately in the ablation because
they were designed against this 30-query benchmark.

To use OpenAI embeddings, install the optional dependency and set:

```text
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
OPENAI_API_KEY=...
```

The database vector dimension is fixed when the schema is created. Use
`python -m src.cli reset-db --yes` after changing embedding dimensions.

To use a Sentence-Transformers Cross-Encoder:

```text
RERANKER_PROVIDER=cross-encoder
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Install with `.\scripts\setup.ps1` first. If the model cannot be loaded,
the reranker falls back to the heuristic implementation instead of failing the
whole evaluation.

### Local BGE reranker

The provided `bge-reranker-large` is supported as a real Cross-Encoder:

```text
RERANKER_PROVIDER=cross-encoder
RERANKER_MODEL=data/models/bge-reranker-large
RERANKER_DEVICE=cuda
RERANKER_FP16=true
```

It runs on the 4GB GPU in fp16 (about 1.1GB), but on this synthetic policy
dataset it does not beat the default rules-free RRF: nDCG@5 is 0.979 vs 0.994
and P50 is about 176 ms vs 8 ms. It remains an optional provider. See
[reports/reranker_comparison.md](reports/reranker_comparison.md).

## BGE-large-en-v1.5

BGE-large is the recommended neural embedding for this English dataset:

```text
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=data/models/bge-large-en-v1.5
EMBEDDING_DIM=1024
EMBEDDING_DEVICE=cpu
```

The adapter automatically applies BGE's retrieval query prefix:
`Represent this sentence for searching relevant passages: `. Documents are
encoded without the prefix. Rebuild and evaluate with:

```powershell
python -m src.cli reset-db --yes
python -m src.cli ingest
python -m src.cli eval --dense-weight 0.5 --sparse-weight 0.5
```

See [reports/results_bge_large_pgvector.md](reports/results_bge_large_pgvector.md).

### GPU inference

All project dependencies are installed in the project-local environment
`.venv`; do not install packages into the system Anaconda base
environment. The environment includes CUDA PyTorch:

```powershell
$env:EMBEDDING_PROVIDER='sentence-transformers'
$env:EMBEDDING_MODEL='data\models\bge-large-en-v1.5'
$env:EMBEDDING_DIM='1024'
$env:EMBEDDING_DEVICE='cuda'
$env:KMP_DUPLICATE_LIB_OK='TRUE'

& '.\.venv\Scripts\python.exe' -m src.cli eval `
  --dense-weight 0.5 --sparse-weight 0.5 --repeats 3
```

On the RTX 3050 Ti, BGE query embedding is about **27-29 ms P50**. Approximate
end-to-end latency is **35.44 ms P50 / 41.20 ms P99** for Hybrid RRF and
**38.71 ms P50 / 46.84 ms P99** for Hybrid + rerank.

## Qwen3-Embedding-0.6B

Qwen3 remains available through the same provider interface:

```text
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=data/models/Qwen3-Embedding-0.6B
EMBEDDING_DIM=1024
EMBEDDING_DEVICE=cpu
```

The model directory is intentionally ignored by Git. Install the project-local
environment with `.\scripts\setup.ps1`, then rebuild the schema because the
dimension changes from 128 to 1024:

```powershell
python -m src.cli reset-db --yes
python -m src.cli ingest
python -m src.cli eval --dense-weight 0.5 --sparse-weight 0.5
```

On this Windows/Anaconda setup an OpenMP runtime conflict appears when PyTorch
and scikit-learn are loaded together; `KMP_DUPLICATE_LIB_OK=TRUE` is set by the
embedding adapter as a local workaround. A clean virtual environment is
preferable for deployment. The CPU embedding latency is much higher than the
PostgreSQL retrieval path; a CUDA/ONNX/TEI deployment is the next optimization.

## Commands

```text
python -m src.cli init-db [--reset]
python -m src.cli ingest [--strategy document|fixed|sentence|all] [--refit]
python -m src.cli index
python -m src.cli eval [--repeats 3] [--backend postgres|memory]
python -m src.cli compare-chunking
python -m src.cli benchmark-index
python -m src.cli ablation-metadata
python -m src.cli judge
python -m src.cli search "How long do I have to report an unauthorized charge?"
python -m src.cli stats
python -m src.cli reset-db --yes
```

## FastAPI / OpenAPI

The API is available through the `api` compose profile:

```powershell
docker compose --profile api up -d --build
```

The first deployment needs a one-off ingest so the API has data:

```powershell
docker compose --profile api build api
docker compose --profile api run --rm api python -m src.cli ingest
docker compose --profile api up -d api
```

Open:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`
- Health: `http://localhost:8000/healthz`

Example search:

```powershell
$body = @{ query = "balance transfer fee"; top_k = 5; mode = "hybrid"; use_metadata = $false } |
  ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:8000/api/v1/search -Method Post `
  -ContentType application/json -Body $body
```

The API loads the embedder once at startup, keeps the database connection
reusable, and supports `dense_only`, `sparse_only`, `hybrid`, and
`hybrid_rerank` modes. The compose profile mounts `./data/models` read-only and
uses CPU BGE by default; set `EMBEDDING_DEVICE=cuda` and configure Docker GPU
support for GPU inference.

## LLM-as-a-Judge

`python -m src.cli judge` retrieves the rules-free Top-5 context, generates a
cited answer with a local `Qwen2.5-0.5B-Instruct`, and asks the model to score
Context Relevance and Faithfulness. The local 0.5B judge is intentionally a
smoke-test judge; it saturates on context relevance and has two JSON parse
failures in the 30-query run. See [reports/llm_judge.md](reports/llm_judge.md)
and [reports/llm_judge.zh-CN.md](reports/llm_judge.zh-CN.md).

`python -m src.cli all` runs ingestion and evaluation in one process. Use
`--backend memory` for the offline path.

## Repository layout

```text
data/                       Provided benchmark data and generated artifacts
docs/roadmap.md             Production scaling, freshness and PII roadmap
reports/results.md          Human-readable evaluation report
reports/results.csv         Per-query metrics for further analysis
reports/figures/            Quality and latency charts
src/ingestion/              Loading, metadata normalization, chunking, embedding
src/retrieval/              Dense, sparse, RRF, reranking and pipeline
src/eval/                   Metrics, experiment runner and report generation
src/db.py                   PostgreSQL + pgvector gateway
src/db_memory.py            In-memory validation fallback
tests/                      Metric, fusion, query parsing and chunking tests
```

## Dataset observations

- 150 policy chunks across 10 categories and 4 card tiers.
- 30 labeled customer-support queries; 23 have one relevant document, 1 has
  two, and 6 have three.
- Documents are already short (median about 282 characters), so document-level
  chunks are the strongest default. The chunking module exists to test that
  claim rather than to split blindly.
- `policy_code` is not unique across products; never use it as the primary key.
- Category metadata contains some noise, so the pipeline does not hard-filter
  on category.
- Generic questions often have valid answers in an unlabeled card tier. Strict
  metrics therefore understate retrieval quality; the report calls this out.

## Evaluation notes

The four pipelines are:

1. Dense only.
2. Sparse only.
3. Hybrid with weighted RRF.
4. Hybrid + reranker.

The evaluation fetches **ten** ranked documents per query and computes Recall
and nDCG at both 5 and 10. Recall@1, MRR and nDCG@5 remain the most
discriminating metrics; Recall@10 often saturates on this small dataset but is
now measured over ten actual results. Latency is measured end-to-end;
database-only timing is kept in the raw JSON so the two are not conflated.

## Tests and checks

```powershell
pytest
ruff check src tests
mypy src
```

## Known limitations

- The default local LSA embedding is a reproducibility fallback, not a
  production-grade semantic model.
- The benchmark is small and synthetic; global parameters are used rather than
  per-query tuning.
- The PostgreSQL/pgvector path is the intended delivery path, but the offline
  backend exists because Docker may require administrator privileges on Windows.
- Answer generation and LLM-as-a-Judge are available through
  `src/eval/judge.py` and `reports/llm_judge.md`, but the 0.5B local judge is a
  smoke test, not a reliable production evaluator. Retrieval scores never use
  generated answers.

