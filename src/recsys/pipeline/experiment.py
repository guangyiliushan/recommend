"""Single experiment orchestration.

run_experiment(config: RecBenchConfig) -> Dict[str, float]:
    1. Set random seed
    2. Load dataset (train/val/test)
    3. Instantiate model from registry
    4. Train with Lightning Trainer
    5. Evaluate on test set
    6. Generate curves
    7. Save results & checkpoint
"""
# TODO: Implement run_experiment()
