"""Evaluation pipeline — runs all metrics on test set.

- collect_predictions(model, dataloader) -> y_true, y_pred, y_score
- evaluate_model(model, dataloader, config) -> metric dict
- generate_curves(y_true, y_score) -> ROC/PR curve data
"""
# TODO: Implement evaluator pipeline
