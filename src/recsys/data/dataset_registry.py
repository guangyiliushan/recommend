"""Dataset registration and auto-discovery.

Registers all dataset adapters with DATASET_REGISTRY.
"""

# Import registries so that side-effect registration happens.
from recsys.core.registry import DATASET_REGISTRY  # noqa: F401

# Trigger auto-import of all dataset adapters to register them.
import recsys.data.datasets.taac2025  # noqa: F401
import recsys.data.datasets.taac2026  # noqa: F401
