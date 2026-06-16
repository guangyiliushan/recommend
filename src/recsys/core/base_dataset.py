"""BaseDataset — abstract base class for all datasets.

Standardized data loading pipeline:
    load() → train/val/test splits or single Dataset
    split(ratios) → (train, val, test)
    __len__() / __getitem__(idx)
    get_dataloader(batch_size, ...) → DataLoader

Subclasses implement:
    _load_raw() → raw data dict
    _prepare_splits(raw) → (train, val, test) splits
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import torch
from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset[Dict[str, torch.Tensor]], ABC):
    """Abstract base for all RecSys datasets.

    Each subclass defines the dataset-specific loading logic while
    inheriting common utilities (splitting, DataLoader creation, etc.)

    Metadata properties (required):
        dataset_name: str       – unique name registered in DATASET_REGISTRY
        dataset_url: str        – source URL (e.g. HuggingFace repo)
        feature_cols: List[str] – column names used as model inputs
        label_col: str          – target column name
        num_users: int          – number of unique users
        num_items: int          – number of unique items
    """

    def __init__(
        self,
        root_dir: str = "./data",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        max_seq_len: int = 50,
        min_seq_len: int = 2,
        neg_sample_count: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.split_ratios = split_ratios
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.neg_sample_count = neg_sample_count

        self._train: Optional[Dataset[Any]] = None
        self._val: Optional[Dataset[Any]] = None
        self._test: Optional[Dataset[Any]] = None
        self._full: Optional[Dataset[Any]] = None

    # ------------------------------------------------------------------
    # Properties (subclasses must override / set these)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        ...

    @property
    @abstractmethod
    def dataset_url(self) -> str:
        ...

    @property
    @abstractmethod
    def feature_cols(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def label_col(self) -> str:
        ...

    @property
    @abstractmethod
    def num_users(self) -> int:
        ...

    @property
    @abstractmethod
    def num_items(self) -> int:
        ...

    # ------------------------------------------------------------------
    # Core loading pipeline
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_raw(self) -> Any:
        """Download / load raw data. Return any format convenient for _prepare_splits."""

    @abstractmethod
    def _prepare_splits(self, raw: Any) -> Tuple[Any, Any, Any]:
        """Convert raw data into (train, val, test) splits.

        Each split should be a torch Dataset or dict that __getitem__ can consume.
        """

    def load(self) -> "BaseDataset":
        """Full pipeline: download → preprocess → split."""
        raw = self._load_raw()
        self._train, self._val, self._test = self._prepare_splits(raw)
        return self

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------

    def get_dataloader(
        self,
        split: str = "train",
        batch_size: int = 256,
        num_workers: int = 4,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> DataLoader:
        """Return a DataLoader for *split* ('train' / 'val' / 'test' / 'full')."""
        ds = self.get_split(split)
        dl_kwargs: Dict[str, Any] = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split == "train" and shuffle),
            pin_memory=True,
        )
        dl_kwargs.update(kwargs)
        return DataLoader(ds, **dl_kwargs)

    def get_split(self, split: str) -> Dataset[Any]:
        """Return the underlying Dataset for a split."""
        mapping: Dict[str, Optional[Dataset[Any]]] = {
            "train": self._train,
            "val": self._val,
            "test": self._test,
            "full": self._full,
        }
        if split not in mapping:
            raise ValueError(f"Unknown split '{split}'. Use train/val/test/full.")
        ds = mapping[split]
        if ds is None:
            raise RuntimeError(
                f"'{split}' split is not available. Call .load() first."
            )
        return ds

    # ------------------------------------------------------------------
    # Abstract – subclasses implement dataset indexing
    # ------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples (full dataset)."""

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single sample as a dict of tensors."""


# ------------------------------------------------------------------
# Type helper for typed registries
# ------------------------------------------------------------------

T_co = TypeVar("T_co", covariant=True)


class SplitDataset(Dataset[T_co]):
    """A thin wrapper that delegates __len__/__getitem__ to an inner iterable.

    Useful when each split is represented by a simple list/dict of tensors.
    """

    def __init__(self, data: Any) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> T_co:
        return self._data[idx]
