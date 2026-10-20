PYTHON ?= .venv/Scripts/python.exe

.PHONY: up down install setup init ingest index eval chunking index-bench ablation judge zh-charts all test lint typecheck verify download-models reset reproduce reproduce-gpu

up:
	docker compose up -d db

down:
	docker compose down

install:
	$(PYTHON) -m pip install -e ".[api,cross-encoder,dev,openai]"

setup:
	powershell -ExecutionPolicy Bypass -File scripts/setup.ps1

init:
	$(PYTHON) -m src.cli init-db

ingest:
	$(PYTHON) -m src.cli ingest

index:
	$(PYTHON) -m src.cli index

eval:
	$(PYTHON) -m src.cli eval

chunking:
	$(PYTHON) -m src.cli compare-chunking

index-bench:
	$(PYTHON) -m src.cli benchmark-index

ablation:
	$(PYTHON) -m src.cli ablation-metadata

judge:
	$(PYTHON) -m src.cli judge

download-models:
	$(PYTHON) scripts/download_models.py bge-embedding bge-reranker

zh-charts:
	$(PYTHON) scripts/generate_zh_charts.py

all: reproduce

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

typecheck:
	$(PYTHON) -m mypy src

verify: test lint typecheck

reset:
	$(PYTHON) -m src.cli reset-db --yes

reproduce:
	powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 -Profile offline

reproduce-gpu:
	powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 -Profile bge-gpu

