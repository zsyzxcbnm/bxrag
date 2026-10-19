# Apex RAG Retrieval Engine

A reproducible hybrid retrieval and evaluation harness for the Apex Card Services
credit-card support knowledge base. The project is intentionally retrieval-first:
the benchmark measures whether the right policy document is found, not whether a
chat model sounds convincing.

## What is included

- PostgreSQL + `pgvector` storage with HNSW and full-text indexes.
- Document-level, fixed-window and semantic chunking strategies.
- Dense retrieval, PostgreSQL `tsvector` retrieval, weighted Reciprocal Rank
  Fusion and a second-stage reranker.
- Offline and optional API-backed embedding/reranker providers.
- Recall@K, Precision@K, MRR, nDCG@K and P50/P95/P99 latency evaluation.
- Markdown report, CSV raw results and PNG charts.
- A memory backend that mirrors the database interface for offline validation.

Current PostgreSQL + pgvector run with local BGE-large-en-v1.5: dense-only
Recall@1 **0.772** / MRR **0.922**; hybrid + rerank Recall@1 **0.817** / MRR
**0.967** / nDCG@5 **0.964**, retrieval P50 **7.7 ms** and P99 **10.4 ms**;
GPU query embedding adds about **25 ms** P50. Approximate end-to-end latency is
**32 ms P50 / 40 ms P99**. The HNSW sweep reports 2.5 ms P50 / 7-8 ms P99. See
[reports/results.md](reports/results.md) and
[reports/chunking.md](reports/chunking.md).

| Embedding | Dense Recall@1 | Dense MRR | Hybrid + rerank Recall@1 | Hybrid + rerank MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + SVD baseline | 0.761 | 0.917 | 0.783 | 0.950 | 0.950 |
| BGE-large-en-v1.5 | 0.772 | 0.922 | **0.817** | **0.967** | **0.964** |
| Qwen3-Embedding-0.6B | **0.817** | **0.967** | **0.817** | **0.967** | 0.961 |

## Architecture

```text
data/kb_documents.json
        |
        v
  normalize + metadata
        |
        v
  document / fixed / semantic chunks
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
                       v
              Cross-Encoder or heuristic
                       |
                       v
                    Top-5 evidence
```

## Quick start with PostgreSQL

Requirements: Python 3.11+, Docker Desktop (or another Docker daemon).

```powershell
docker compose up -d db
python -m pip install -e ".[dev]"
python -m src.cli init-db
python -m src.cli ingest
python -m src.cli eval
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

## Embedding and reranker providers

The default is deliberately offline and reproducible:

- Embedding: local TF-IDF + Truncated SVD, 128 dimensions.
- Reranker: metadata-aware heuristic reranker.

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

Install `pip install -e ".[cross-encoder]"` first. If the model cannot be loaded,
the reranker falls back to the heuristic implementation instead of failing the
whole evaluation.

## BGE-large-en-v1.5

BGE-large is the recommended English CPU default for this dataset:

```text
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=G:/博贤/data/models/bge-large-en-v1.5
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

The main Anaconda environment currently has CPU-only PyTorch. A separate CUDA
environment is available at `G:\博贤\.venv-gpu`:

```powershell
$env:EMBEDDING_PROVIDER='sentence-transformers'
$env:EMBEDDING_MODEL='G:\博贤\data\models\bge-large-en-v1.5'
$env:EMBEDDING_DIM='1024'
$env:EMBEDDING_DEVICE='cuda'
$env:KMP_DUPLICATE_LIB_OK='TRUE'

& 'G:\博贤\.venv-gpu\Scripts\python.exe' -m src.cli eval `
  --dense-weight 0.5 --sparse-weight 0.5 --repeats 3
```

On the RTX 3050 Ti, BGE query embedding is about **25 ms P50** and the complete
hybrid + rerank path is about **32 ms P50 / 40 ms P99**.

## Qwen3-Embedding-0.6B

Qwen3 remains available through the same provider interface:

```text
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=G:/博贤/data/models/Qwen3-Embedding-0.6B
EMBEDDING_DIM=1024
EMBEDDING_DEVICE=cpu
```

The model directory is intentionally ignored by Git. Install the optional
dependency with `pip install -e ".[cross-encoder]"`, then rebuild the schema
because the dimension changes from 128 to 1024:

```powershell
python -m src.cli reset-db --yes
python -m src.cli ingest
python -m src.cli eval --dense-weight 0.7 --sparse-weight 0.3
```

On this Windows/Anaconda setup an OpenMP runtime conflict appears when PyTorch
and scikit-learn are loaded together; `KMP_DUPLICATE_LIB_OK=TRUE` is set by the
embedding adapter as a local workaround. A clean virtual environment is
preferable for deployment. The CPU embedding latency is much higher than the
PostgreSQL retrieval path; a CUDA/ONNX/TEI deployment is the next optimization.

## Commands

```text
python -m src.cli init-db [--reset]
python -m src.cli ingest [--strategy document|fixed|semantic|all] [--refit]
python -m src.cli index
python -m src.cli eval [--repeats 3] [--backend postgres|memory]
python -m src.cli compare-chunking
python -m src.cli benchmark-index
python -m src.cli search "How long do I have to report an unauthorized charge?"
python -m src.cli stats
python -m src.cli reset-db --yes
```

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

The headline metrics are Recall@1, MRR and nDCG@5. Recall@10 saturates on this
small dataset. Latency is measured end-to-end; database-only timing is kept in
the raw JSON so the two are not conflated.

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
- No answer-generation or LLM-judge pipeline is included in this first
  closed-loop version.

