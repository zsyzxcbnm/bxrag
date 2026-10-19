.PHONY: up down install init ingest index eval all test lint typecheck reset

up:
	docker compose up -d db

down:
	docker compose down

install:
	python -m pip install -e ".[dev]"

init:
	python -m src.cli init-db

ingest:
	python -m src.cli ingest

index:
	python -m src.cli index

eval:
	python -m src.cli eval

all:
	python -m src.cli all

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src

reset:
	python -m src.cli reset-db --yes

