# Reviewer Verification Guide

Run every command from the repository root after cloning.

## Fast verification: no Docker, GPU, or model download

```powershell
git clone https://github.com/zsyzxcbnm/bxrag.git
cd bxrag
powershell -ExecutionPolicy Bypass -File .\scripts\reproduce.ps1
```

Expected output:

```text
Offline reproduction complete: reports/offline
```

Results are written to `reports/offline/`.

## Code quality checks

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m ruff check src tests scripts
& .\.venv\Scripts\python.exe -m mypy src
```

Expected: 16 tests pass, Ruff passes, mypy reports no issues.

## Full BGE GPU + pgvector verification

Prerequisites: Docker Desktop, NVIDIA GPU, and network access on the first run.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\reproduce.ps1 -Profile bge-gpu
```

The script creates `.venv` if needed, downloads missing BGE models, starts
pgvector, rebuilds the 1024-dimensional index, runs evaluation, HNSW tuning,
the metadata ablation, and regenerates the Chinese charts in `reports/`.

## Docker/API verification

```powershell
make docker-build
make docker-ingest
make docker-api
```

Without `make`:

```powershell
docker compose --profile api build api
docker compose --profile api run --rm api python -m src.cli ingest
docker compose --profile api up -d api
```

Then open `http://localhost:8000/docs`.

