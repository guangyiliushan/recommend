# Contributing to RecBench

Thanks for considering a contribution to RecBench.

This repository is in an active architecture-building phase. That means high-value contributions are the ones that strengthen contracts, reproducibility, and maintainability before adding broad algorithm coverage.

## Before You Start

Please read:

- `README.md`
- `docs/architecture.md`
- `docs/development.md`
- `docs/models.md`

## Working Agreement

- Use `uv` for Python environment management to stay consistent with CI
- Keep changes aligned with the package root `src/recsys`
- Prefer small, reviewable pull requests over broad rewrites
- Do not describe unfinished modules as production-ready
- Add or update tests when a change affects contracts or behavior

## What To Contribute First

Good first categories:

- config normalization
- registry improvements
- dataset adapter hardening
- evaluator and metric contracts
- experiment and benchmark orchestration
- documentation and tests

Lower priority during the current phase:

- large batches of model implementations without shared runtime support
- broad refactors that rename everything at once
- feature additions that bypass config or registry contracts

## Local Setup

```bash
uv sync --extra dev
```

Common commands:

```bash
uv run pytest -v
uv run ruff check .
uv run zensical build --strict --clean
```

## Branch and PR Guidance

For pull requests:

1. explain the problem being solved
2. describe the architectural impact
3. list affected modules
4. mention any tests added or reasons tests were not added
5. note any follow-up work still needed

Recommended PR scopes:

- one contract change
- one runtime improvement
- one dataset adapter improvement
- one model sample integration
- one documentation package

## Model Contributions

When adding a model:

1. define its registry metadata clearly
2. confirm the model fits an existing task contract
3. document required input features
4. document expected output fields
5. add at least one focused test around shape, registration, or runtime compatibility

Do not add a model that depends on ad hoc batch formats or special-case orchestration unless the shared contract is extended first.

## Documentation Contributions

Documentation is a first-class contribution area in this repository.

The docs site is sourced from `docs/` and built with Zensical. If you update developer-facing behavior, update the relevant documentation in the same pull request whenever possible.

## Communication

If you want to make a substantial architectural change, open an issue or draft PR first so the shared direction can be agreed before implementation expands.
