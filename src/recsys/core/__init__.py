"""Core abstractions: base model, base dataset, registry."""

from recsys.core.base_dataset import BaseDataset  # noqa: F401
from recsys.core.registry import (  # noqa: F401
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    Registry,
)
