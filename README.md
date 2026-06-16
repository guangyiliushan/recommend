# RecBench

RecBench is a recommendation-system benchmark project that aims to unify classical collaborative filtering, deep CTR/CVR models, sequence recommendation, feature crossing, multi-task PCVR modeling, and emerging generative recommendation methods under one reproducible engineering workflow.

The repository currently contains:

- a clear package layout under `src/recsys`
- dataset adapters for TAAC 2025 and TAAC 2026 samples
- model family placeholders for 50+ algorithms
- benchmark-oriented configuration structure under `configs/`
- CI, release, and documentation deployment workflows

The repository does not yet provide a fully completed end-to-end benchmark runtime. Several core modules are still scaffolded and are being standardized before wider model implementation.

## Why This Project

- Unify multiple recommendation paradigms under one experiment contract
- Compare models with consistent configuration, training, and evaluation logic
- Provide a maintainable codebase instead of a collection of disconnected scripts
- Make benchmark runs reproducible in local development and CI

## Current Status

- Package root: `src/recsys`
- Dependency management direction: `uv` in CI, with documentation now aligned to that workflow
- Implemented data adapters: `taac2025_*`, `taac2026_*`
- Major model families present as skeletons
- Core orchestration modules such as trainer, evaluator, experiment, and benchmark runner are still under construction

If you plan to contribute code, read [CONTRIBUTING.md](file:///d:/Project/Project/recommend/CONTRIBUTING.md) first and then the documentation in `docs/`.

## Project Layout

```text
.
|-- configs/                  # Hydra-style configuration entrypoints
|-- docs/                     # Zensical documentation source
|-- scripts/                  # User-facing CLI scripts
|-- src/recsys/
|   |-- core/                 # Registries and base contracts
|   |-- data/                 # Dataset adapters and preprocessing
|   |-- evaluation/           # Metrics, evaluator, visualization
|   |-- models/               # Model families and individual algorithms
|   |-- pipeline/             # Experiment and benchmark orchestration
|   |-- training/             # Trainer abstractions, callbacks, losses
|   `-- utils/                # Config, logging, device, reproducibility
|-- tests/                    # Contract and regression tests
`-- .github/workflows/        # CI, docs deploy, release automation
```

## Quick Start

### 1. Create or sync the environment

This repository should be operated with `uv` for consistency with GitHub Actions.

```bash
uv sync --extra dev
```

### 2. Run tests

```bash
uv run pytest -v
```

### 3. Run lint checks

```bash
uv run ruff check .
```

### 4. Build the documentation site locally

```bash
uv run zensical build --strict --clean
```

## Documentation

- [Documentation Home](file:///d:/Project/Project/recommend/docs/index.md)
- [Getting Started](file:///d:/Project/Project/recommend/docs/getting-started.md)
- [Concepts: Architecture](file:///d:/Project/Project/recommend/docs/concepts/architecture.md)
- [Concepts: Configuration](file:///d:/Project/Project/recommend/docs/concepts/configuration.md)
- [Project: Structure](file:///d:/Project/Project/recommend/docs/project/structure.md)
- [Project: API Contracts](file:///d:/Project/Project/recommend/docs/project/api-contracts.md)
- [Project: Artifacts](file:///d:/Project/Project/recommend/docs/project/artifacts.md)
- [Project: Dataset Guide](file:///d:/Project/Project/recommend/docs/project/datasets.md)
- [Project: Evaluation Guide](file:///d:/Project/Project/recommend/docs/project/evaluation.md)
- [Project: Pipeline Guide](file:///d:/Project/Project/recommend/docs/project/pipeline.md)
- [Project: Benchmarking Guide](file:///d:/Project/Project/recommend/docs/project/benchmarking.md)
- [Project: Development Guide](file:///d:/Project/Project/recommend/docs/project/development.md)
- [Project: Model Integration](file:///d:/Project/Project/recommend/docs/project/models.md)
- [Experiments](file:///d:/Project/Project/recommend/docs/experiments/index.md)
- [Guides](file:///d:/Project/Project/recommend/docs/guides/index.md)
- [Papers](file:///d:/Project/Project/recommend/docs/papers/index.md)
- [Operations: Overview](file:///d:/Project/Project/recommend/docs/operations/overview.md)
- [Operations: Maintenance](file:///d:/Project/Project/recommend/docs/operations/maintenance.md)

The documentation site is built from `docs/` and deployed through [deploy-docs.yml](file:///d:/Project/Project/recommend/.github/workflows/deploy-docs.yml).

## Recommended Roadmap

The next stable milestone is not "implement all 50+ models", but:

1. standardize config, registry, dataset, and model contracts
2. complete the minimal single-experiment runtime
3. support one non-neural baseline and one trainable neural baseline end-to-end
4. stabilize evaluation outputs and artifact layout
5. expand model families with contract tests

## Development Principles

- Prefer `uv` over `pip` or `npm`-style workflows for Python environment operations
- Keep configuration and code contracts aligned
- Treat the current repository state honestly in docs and READMEs
- Add new models only after the shared runtime contract is stable
- Favor focused tests around contracts and regression risk

## Contributing

Issues and pull requests are welcome once they align with the current architecture direction. Before making large changes:

1. read [CONTRIBUTING.md](file:///d:/Project/Project/recommend/CONTRIBUTING.md)
2. check the docs in `docs/`
3. avoid adding new model implementations on top of unstable base contracts

## License

This project is published under the MIT License as declared in `pyproject.toml`.
