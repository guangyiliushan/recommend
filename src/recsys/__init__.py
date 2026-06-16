"""RecBench — Comprehensive Recommendation System Benchmark.

RecBench provides a unified framework for benchmarking recommendation algorithms
from 2001 to 2026, covering:

- **Classical CF/MF/FM** (2001-2015): ItemCF, UserCF, MF, FM
- **Deep CTR** (2016-2019): DeepFM, Wide&Deep, DIN, DIEN, DLRM
- **Sequence Recommendation**: SASRec, BERT4Rec, SIM, MIMN
- **Feature Crossing**: DCN, DCNv2, DHEN
- **PCVR Multi-Task Modeling**: ESMM, ESM2, RankUp
- **Unified Architectures**: HSTU, InterFormer, OneTrans
- **Generative Recommendation**: RecGPT, TIGER, IDGenRec

Architecture
------------
The project is organized around four stable boundaries:

1. **Config**: Hydra + dataclass for configuration management
2. **Dataset Adapter**: Unified data loading and preprocessing
3. **Model Contract**: BaseRecommender with capability-based interfaces
4. **Experiment Runtime**: Pipeline for training, evaluation, and benchmarking

Quick Start
-----------
>>> # Auto-discover all models
>>> from recsys.models import auto_discover_models
>>> auto_discover_models()
>>>
>>> # Access the model registry
>>> from recsys.core.registry import MODEL_REGISTRY
>>> print(MODEL_REGISTRY.list())
>>>
>>> # Get a model class
>>> from recsys.models import get_model
>>> ItemCF = get_model("itemcf")
>>> model = ItemCF(similarity="cosine", top_k_neighbors=50)

Modules
-------
- :mod:`recsys.core`: Base classes, registry, and prediction bundle
- :mod:`recsys.data`: Dataset adapters and preprocessing
- :mod:`recsys.models`: Model implementations by family
- :mod:`recsys.training`: Trainer, callbacks, losses, optimizers
- :mod:`recsys.evaluation`: Metrics and evaluators
- :mod:`recsys.pipeline`: Experiment and benchmark orchestration
- :mod:`recsys.utils`: Configuration, logging, and utilities

References
----------
- Architecture: docs/concepts/architecture.md
- Configuration: docs/concepts/configuration.md
- API Contracts: docs/project/api-contracts.md
- Model Integration: docs/project/models.md
"""

from __future__ import annotations

__version__ = "0.1.0"

# Core components
from recsys.core.base_model import (
    BaseRecommender,
    Batch,
    Capability,
    ModelContractError,
    ModelErrorCode,
    ModelOutput,
    NeuralRecommender,
)
from recsys.core.prediction_bundle import PredictionBundle
from recsys.core.registry import (
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
)

# Model registry utilities
from recsys.models import (
    auto_discover_models,
    get_model,
    get_model_metadata,
    list_models,
    list_models_by_family,
    list_models_by_task_type,
)

__all__ = [
    # Version
    "__version__",
    # Registries
    "MODEL_REGISTRY",
    "DATASET_REGISTRY",
    "METRIC_REGISTRY",
    "LOSS_REGISTRY",
    # Base classes
    "BaseRecommender",
    "NeuralRecommender",
    "Batch",
    "ModelOutput",
    "Capability",
    "PredictionBundle",
    # Errors
    "ModelContractError",
    "ModelErrorCode",
    # Model utilities
    "auto_discover_models",
    "get_model",
    "get_model_metadata",
    "list_models",
    "list_models_by_family",
    "list_models_by_task_type",
]
