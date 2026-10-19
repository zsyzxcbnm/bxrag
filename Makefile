PYTHON ?= .venv/Scripts/python.exe

.PHONY: up down install init ingest index eval chunking index-bench ablation judge zh-charts all test lint typecheck reset

up:
	docker compose up -d db

down:
	docker compose down

install:
	$(PYTHON) -m pip install -e ".[api,cross-encoder,dev,openai]"

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

zh-charts:
	$(PYTHON) scripts/generate_zh_charts.py

all:
	$(PYTHON) -m src.cli all

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

typecheck:
	$(PYTHON) -m mypy src

reset:
	$(PYTHON) -m src.cli reset-db --yes

