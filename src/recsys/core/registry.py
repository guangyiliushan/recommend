"""Registry pattern for models, datasets, metrics, and losses.

Provides auto-discovery and decorator-based registration.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Dict, List, Type, TypeVar

T = TypeVar("T")


class Registry:
    """A generic registry that maps names to classes with metadata."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: Dict[str, Type[Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(
        self,
        name: str,
        **metadata: Any,
    ) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class with the given name and metadata."""

        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._items:
                raise KeyError(f"{self._name} '{name}' is already registered")
            self._items[name] = cls
            self._metadata[name] = metadata
            return cls

        return decorator

    def get(self, name: str) -> Type[Any]:
        """Retrieve a registered class by name."""
        if name not in self._items:
            available = ", ".join(sorted(self._items.keys()))
            raise KeyError(
                f"{self._name} '{name}' not found. Available: {available}"
            )
        return self._items[name]

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Retrieve metadata for a registered item."""
        if name not in self._metadata:
            raise KeyError(f"No metadata for '{name}' in {self._name}")
        return self._metadata[name]

    def list(self) -> List[str]:
        """Return all registered names."""
        return sorted(self._items.keys())

    def list_by(self, key: str, value: Any) -> List[str]:
        """Return names that match a specific metadata key-value pair."""
        return sorted(
            name
            for name, meta in self._metadata.items()
            if meta.get(key) == value
        )

    def auto_discover(self, package_path: str) -> int:
        """Recursively import all modules under *package_path* for side-effect registration.

        Returns the number of modules loaded.
        """
        package = importlib.import_module(package_path)
        count = 0
        for _, mod_name, _is_pkg in pkgutil.walk_packages(
            package.__path__,
            prefix=package.__name__ + ".",
        ):
            importlib.import_module(mod_name)
            count += 1
        return count

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry('{self._name}', items={len(self._items)})"


# Global registries used throughout the project.
MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
METRIC_REGISTRY = Registry("metric")
LOSS_REGISTRY = Registry("loss")
