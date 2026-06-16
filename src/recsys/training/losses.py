"""Loss function library.

Supported losses:
    - BCELoss / BCEWithLogitsLoss — binary classification (CTR/CVR)
    - CrossEntropyLoss — multi-class
    - BPRLoss — Bayesian Personalized Ranking
    - InfoNCE — contrastive learning
    - TOP1 / BPR-Max — session recommendation
    - MultiTaskLoss — weighted sum / uncertainty weighting
    - FocalLoss — class imbalance
    - AdaptiveHuberLoss — robust regression
"""
# TODO: Implement loss functions with LOSS_REGISTRY
