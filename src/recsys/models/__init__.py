"""Model zoo organized by family.

This package provides the model layer of RecBench, organized by algorithm family:
- classical: Collaborative filtering, matrix factorization, FM-based methods
- deep_ctr: Deep learning CTR models (DeepFM, DIN, DIEN, DLRM, Wide&Deep)
- sequence: Sequential recommendation models (SASRec, BERT4Rec, SIM, MIMN)
- feature_cross: Feature interaction models (DCN, DCNv2, DHEN)
- pcvr: Multi-task CVR estimation models (ESMM, ESM2, RankUp)
- unified: Unified architectures (HSTU, InterFormer, OneTrans)
- generative: Generative recommendation models (RecGPT, TIGER, IDGenRec)

Usage:
    # Auto-discover all registered models
    from recsys.models import auto_discover_models
    auto_discover_models()

    # Access the model registry
    from recsys.core.registry import MODEL_REGISTRY
    print(MODEL_REGISTRY.list())

    # Get a specific model class
    ModelCls = MODEL_REGISTRY.get("itemcf")

    # List models by family
    classical_models = MODEL_REGISTRY.list_by("family", "classical")

    # List models by task type
    ranking_models = MODEL_REGISTRY.list_by("task_type", "ranking")
"""

from __future__ import annotations

from recsys.models.model_registry import (
    auto_discover_models,
    get_model,
    get_model_metadata,
    list_models,
    list_models_by_family,
    list_models_by_task_type,
)

__all__ = [
    # Auto-discovery
    "auto_discover_models",
    # Registry access
    "get_model",
    "get_model_metadata",
    "list_models",
    "list_models_by_family",
    "list_models_by_task_type",
]
