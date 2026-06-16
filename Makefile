# Makefile-compatible convenience targets for RecBench
# Usage: make train, make benchmark-classical, make test, make lint

.PHONY: help install dev-install train benchmark benchmark-classical benchmark-all test lint format clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

dev-install: ## Install dev + production dependencies
	pip install -e ".[dev]"

train: ## Run single model training (override via MODEL=xxx DATA=xxx)
	python scripts/run_single.py --config configs/config.yaml

benchmark: ## Run benchmark (override via CONFIG=xxx)
	python scripts/run_benchmark.py --config configs/experiment/benchmark_deep_ctr.yaml

benchmark-classical: ## Run classical benchmark
	python scripts/run_benchmark.py --config configs/experiment/benchmark_classical.yaml

benchmark-all: ## Run full benchmark (all models)
	python scripts/run_benchmark.py --config configs/experiment/benchmark_all.yaml

test: ## Run tests
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

lint: ## Run linters
	ruff check src/ tests/
	black --check src/ tests/
	isort --check src/ tests/

format: ## Auto-format code
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/
