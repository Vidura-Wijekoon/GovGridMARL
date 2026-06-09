# Contributing

This is the research codebase for the GovGridMARL MSc thesis. Contributions
during the project follow a lightweight workflow.

## Setup

```bash
bash scripts/setup_env.sh        # conda env + editable install
pre-commit install               # enable lint/format hooks
```

## Workflow

1. Branch from `main`: `git checkout -b feat/<short-name>`.
2. Keep changes scoped to a roadmap task; reference the task id (e.g. `m3-t3`)
   in the commit body.
3. Run checks locally before pushing:
   ```bash
   make lint
   make test
   ```
4. Open a PR; CI (lint + tests) must pass before merge.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `exp:` (experiments).

## Code style

- `ruff` + `black`, line length 100.
- Type hints required for new public functions; `mypy` should stay clean.
- New functionality needs a test.
