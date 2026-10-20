# 评审验证指南

所有命令都在 `git clone` 后的仓库根目录执行。

## 快速验证：不需要 Docker、GPU 或模型下载

```powershell
git clone https://github.com/zsyzxcbnm/bxrag.git
cd bxrag
powershell -ExecutionPolicy Bypass -File .\scripts\reproduce.ps1
```

预期最后一行：

```text
Offline reproduction complete: reports/offline
```

结果在 `reports/offline/`。

## 代码质量检查

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m ruff check src tests scripts
& .\.venv\Scripts\python.exe -m mypy src
```

预期：22 个测试通过，Ruff 和 mypy 无错误。

## 完整 BGE GPU + pgvector 验证

前置条件：Docker Desktop、NVIDIA GPU，首次运行需要网络。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\reproduce.ps1 -Profile bge-gpu
```

脚本会自动创建 `.venv`、下载缺失模型、启动 pgvector、重建 1024 维索引、
执行评测、HNSW 调参、元数据消融，并在 `reports/` 生成中文图表。

## Docker/API 验证

```powershell
make docker-build
make docker-ingest
make docker-api
```

没有 `make` 时：

```powershell
docker compose --profile api build api
docker compose --profile api run --rm api python -m src.cli ingest
docker compose --profile api up -d api
```

然后访问 `http://localhost:8000/docs`。
