"""Configuration management — Hydra + YAML + dataclass hybrid.

RecBenchConfig dataclass:
    experiment: ExperimentConfig (name, seed, device, output_dir, ...)
    data: DataConfig (name, batch_size, split_ratios, ...)
    model: ModelConfig (name, family, params)
    training: TrainingConfig (epochs, lr, optimizer, ...)
    evaluation: EvaluationConfig (metrics, ranking_k, threshold, ...)

Support:
    - YAML file loading
    - Hydra CLI override
    - Config validation
"""
# TODO: Implement RecBenchConfig and loaders
