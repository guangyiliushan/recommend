"""Model registration and auto-discovery.

This module provides utilities for discovering and accessing registered models.
All models are registered via the MODEL_REGISTRY in recsys.core.registry.

The auto-discovery mechanism imports all model modules under recsys.models
to trigger their @MODEL_REGISTRY.register() decorators.

Usage:
    # Auto-discover all models (typically called at startup)
    from recsys.models.model_registry import auto_discover_models
    count = auto_discover_models()
    print(f"Discovered {count} model modules")

    # Get a model class by name
    from recsys.models.model_registry import get_model
    ItemCF = get_model("itemcf")

    # List all registered models
    from recsys.models.model_registry import list_models
    print(list_models())

    # Filter by family or task type
    classical = list_models_by_family("classical")
    ranking = list_models_by_task_type("ranking")
"""

from __future__ import annotations

from typing import Any, Dict, List

from recsys.core.registry import MODEL_REGISTRY

# Model families defined in the project
MODEL_FAMILIES = [
    "classical",      # CF, MF, FM-based methods (2001-2015)
    "deep_ctr",       # Deep CTR models (2016-2019)
    "sequence",       # Sequential recommendation
    "feature_cross",  # Feature interaction models
    "pcvr",           # Multi-task CVR estimation
    "unified",        # Unified architectures
    "generative",     # Generative recommendation
]

# Task types supported by the project
TASK_TYPES = [
    "pointwise",  # CTR/CVR classification
    "ranking",    # Learning-to-rank
    "multitask",  # Multi-task learning
]


def auto_discover_models() -> int:
    """Auto-discover all model modules by importing them.

    This triggers the @MODEL_REGISTRY.register() decorators in each model file.
    Should be called once at application startup.

    Returns
    -------
    int
        Number of model modules loaded.

    Example
    -------
    >>> from recsys.models.model_registry import auto_discover_models
    >>> count = auto_discover_models()
    >>> print(f"Loaded {count} model modules")
    """
    return MODEL_REGISTRY.auto_discover("recsys.models")


def get_model(name: str) -> type:
    """Get a registered model class by name.

    Parameters
    ----------
    name : str
        Model name (e.g., "itemcf", "deepfm", "sasrec").

    Returns
    -------
    type
        The model class.

    Raises
    ------
    KeyError
        If the model is not registered.

    Example
    -------
    >>> from recsys.models.model_registry import get_model
    >>> ItemCF = get_model("itemcf")
    >>> model = ItemCF(similarity="cosine", top_k_neighbors=50)
    """
    return MODEL_REGISTRY.get(name)


def get_model_metadata(name: str) -> Dict[str, Any]:
    """Get metadata for a registered model.

    Parameters
    ----------
    name : str
        Model name.

    Returns
    -------
    Dict[str, Any]
        Model metadata including:
        - family: Model family (e.g., "classical")
        - year: Publication year
        - task_type: Task type (e.g., "ranking")
        - supports_training: Whether the model supports training
        - required_features: Required input features
        - default_metrics: Default evaluation metrics

    Raises
    ------
    KeyError
        If the model is not registered.

    Example
    -------
    >>> from recsys.models.model_registry import get_model_metadata
    >>> meta = get_model_metadata("itemcf")
    >>> print(meta["family"])  # "classical"
    >>> print(meta["task_type"])  # "ranking"
    """
    return MODEL_REGISTRY.get_metadata(name)


def list_models() -> List[str]:
    """List all registered model names.

    Returns
    -------
    List[str]
        Sorted list of model names.

    Example
    -------
    >>> from recsys.models.model_registry import list_models
    >>> print(list_models())
    ['deepfm', 'din', 'itemcf', 'sasrec', ...]
    """
    return MODEL_REGISTRY.list()


def list_models_by_family(family: str) -> List[str]:
    """List models belonging to a specific family.

    Parameters
    ----------
    family : str
        Model family name. Valid values:
        - "classical": CF, MF, FM-based methods
        - "deep_ctr": Deep CTR models
        - "sequence": Sequential recommendation
        - "feature_cross": Feature interaction models
        - "pcvr": Multi-task CVR estimation
        - "unified": Unified architectures
        - "generative": Generative recommendation

    Returns
    -------
    List[str]
        Sorted list of model names in the family.

    Example
    -------
    >>> from recsys.models.model_registry import list_models_by_family
    >>> classical = list_models_by_family("classical")
    >>> print(classical)
    ['itemcf', 'mf', 'fm', ...]
    """
    return MODEL_REGISTRY.list_by("family", family)


def list_models_by_task_type(task_type: str) -> List[str]:
    """List models supporting a specific task type.

    Parameters
    ----------
    task_type : str
        Task type. Valid values:
        - "pointwise": CTR/CVR classification
        - "ranking": Learning-to-rank
        - "multitask": Multi-task learning

    Returns
    -------
    List[str]
        Sorted list of model names supporting the task type.

    Example
    -------
    >>> from recsys.models.model_registry import list_models_by_task_type
    >>> ranking = list_models_by_task_type("ranking")
    >>> print(ranking)
    ['itemcf', 'sasrec', 'bert4rec', ...]
    """
    return MODEL_REGISTRY.list_by("task_type", task_type)


def list_trainable_models() -> List[str]:
    """List models that support gradient-based training.

    Returns
    -------
    List[str]
        Sorted list of trainable model names.

    Example
    -------
    >>> from recsys.models.model_registry import list_trainable_models
    >>> trainable = list_trainable_models()
    >>> print(trainable)
    ['deepfm', 'din', 'sasrec', ...]
    """
    return MODEL_REGISTRY.list_by("supports_training", True)


def list_non_trainable_models() -> List[str]:
    """List models that do not require gradient-based training.

    These are typically classical methods like ItemCF, UserCF, etc.

    Returns
    -------
    List[str]
        Sorted list of non-trainable model names.

    Example
    -------
    >>> from recsys.models.model_registry import list_non_trainable_models
    >>> non_trainable = list_non_trainable_models()
    >>> print(non_trainable)
    ['itemcf', 'usercf', ...]
    """
    return MODEL_REGISTRY.list_by("supports_training", False)
