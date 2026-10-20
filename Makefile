.PHONY: help setup install lint format test cov train smoke dashboard clean validate-env baselines evaluate show phase2-m4 pareto phase2-m5 evaluate-phase2 thesis

help:
	@echo "setup    - create conda env and install package (scripts/setup_env.sh)"
	@echo "install  - pip editable install with dev extras"
	@echo "lint     - ruff + black --check + mypy"
	@echo "format   - apply black + ruff --fix"
	@echo "test     - run pytest"
	@echo "cov      - run pytest with coverage"
	@echo "train    - run the default training entrypoint"
	@echo "smoke    - verify CUDA + run GPU smoke training job (task m1-t3)"
	@echo "dashboard- launch local MLflow dashboard at http://127.0.0.1:5000"
	@echo "validate-env - random-rollout environment validation (m2-t14)"
	@echo "baselines- cloud/CPU baseline runs, 3 learners x 3 seeds (m3-t5)"
	@echo "evaluate - benchmark tables + figures + Phase-1 gate check (m3-t8/t10)"
	@echo "show     - replay a day on IEEE-14 and open the interactive status page"
	@echo "phase2-m4- Month-4 runs: fairness-weight sweep, form ablation, MOPPO (m4-t5/t6/t8)"
	@echo "pareto   - Pareto front, trade-off tables/figures, operating point (m4-t7/t9/t11-t13)"
	@echo "phase2-m5- Month-5 runs: unconstrained / Lagrangian / PID / joint / lr sweep (m5-t3..t10)"
	@echo "evaluate-phase2 - ablation, joint targets, stress + N-1 tests, safety figures (m5-t9/t11-t13)"
	@echo "thesis   - build docs/thesis/build/thesis.{md,html[,docx,pdf]} (m1-t12)"
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

smoke:
	python scripts/check_compute.py
	python scripts/smoke_train.py --track

validate-env:
	python scripts/validate_env.py

baselines:
	bash scripts/run_baselines_cloud.sh

evaluate:
	python experiments/evaluate.py

dashboard:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ build dist *.egg-info

show:
	python scripts/showcase.py

phase2-m4:
	bash scripts/run_phase2_month4.sh

pareto:
	python experiments/pareto.py

phase2-m5:
	bash scripts/run_phase2_month5.sh

evaluate-phase2:
	python experiments/evaluate_phase2.py

thesis:
	python scripts/build_thesis.py
