.PHONY: help setup install lint format test cov train clean

help:
	@echo "setup    - create conda env and install package (scripts/setup_env.sh)"
	@echo "install  - pip editable install with dev extras"
	@echo "lint     - ruff + black --check + mypy"
	@echo "format   - apply black + ruff --fix"
	@echo "test     - run pytest"
	@echo "cov      - run pytest with coverage"
	@echo "train    - run the default training entrypoint"
	@echo "clean    - remove caches and build artefacts"

setup:
	bash scripts/setup_env.sh

install:
	pip install -e ".[dev]"

lint:
	ruff check src envs experiments tests
	black --check src envs experiments tests
	mypy src

format:
	ruff check --fix src envs experiments tests
	black src envs experiments tests

test:
	pytest

cov:
	pytest --cov=govgridmarl --cov-report=term-missing

train:
	python experiments/train.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ build dist *.egg-info
