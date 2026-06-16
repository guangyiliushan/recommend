"""TAAC 2025 dataset adapter — Tencent Generative Recommendation.

Full-modal advertising generative recommendation dataset from TAAC 2025 competition.
Reference: arxiv:2604.04976

Supports two scale variants:
    - TAAC2025/TencentGR-1M  (~1M  seq rows, ~660k  candidates)
    - TAAC2025/TencentGR-10M (~10M seq rows, ~3.64M candidates)

Dataset subsets:
    seq         – user behavior sequences (main interaction data)
    user_feat   – user-side features (demographic / profile)
    item_feat   – item-side features (categorical + text)
    candidate   – candidate item pool for retrieval evaluation
    mm_emb_*    – multimodal embeddings (text/image/video, various dims)

Usage:
    ds = TAAC2025Dataset(
        version="10M",
        root_dir="./data",
        split_ratios=(0.8, 0.1, 0.1),
        max_seq_len=50,
    ).load()
    train_loader = ds.get_dataloader("train", batch_size=64)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from recsys.core.base_dataset import BaseDataset, T_co

logger = logging.getLogger(__name__)

# HuggingFace repo
_TAAC2025_REPO = "TAAC2025"

# Available scale variants
_AVAILABLE_VERSIONS = ("1M", "10M")

# ---- Feature column names used across seq / candidate subsets ----
_SEQ_FEATURE_COLS = [
    "100", "101", "112", "114", "115", "116", "117", "118", "119",
    "120", "121", "122",
]


class _SequenceSplit(Dataset[Dict[str, torch.Tensor]]):
    """Internal dataset split that holds pre-built sequences."""

    def __init__(
        self,
        sequences: List[Dict[str, Any]],
        item_pool: Optional[torch.Tensor] = None,
        max_seq_len: int = 50,
        neg_sample_count: int = 4,
    ) -> None:
        self._sequences = sequences
        self._item_pool = item_pool
        self.max_seq_len = max_seq_len
        self.neg_sample_count = neg_sample_count

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self._sequences[idx]
        # Truncate sequence to max_seq_len
        item_ids = seq["item_ids"][: self.max_seq_len]
        labels = seq["labels"][: self.max_seq_len]

        # Pad if needed
        pad_len = self.max_seq_len - len(item_ids)
        if pad_len > 0:
            item_ids = np.pad(item_ids, (0, pad_len), constant_values=0)
            labels = np.pad(labels, (0, pad_len), constant_values=-100)

        return {
            "item_ids": torch.as_tensor(item_ids, dtype=torch.long),
            "labels": torch.as_tensor(labels, dtype=torch.long),
        }


class TAAC2025Dataset(BaseDataset):
    """TAAC 2025 TencentGR dataset loaded from HuggingFace.

    Parameters
    ----------
    version : str
        Either ``"1M"`` or ``"10M"``.
    root_dir : str
        Local cache directory for HuggingFace datasets.
    """

    def __init__(
        self,
        version: str = "10M",
        root_dir: str = "./data",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        max_seq_len: int = 50,
        min_seq_len: int = 2,
        neg_sample_count: int = 4,
        **kwargs: Any,
    ) -> None:
        if version not in _AVAILABLE_VERSIONS:
            raise ValueError(
                f"version must be one of {_AVAILABLE_VERSIONS}, got '{version}'"
            )
        self._version = version
        self._repo_id = f"{_TAAC2025_REPO}/TencentGR-{version}"
        super().__init__(
            root_dir=root_dir,
            split_ratios=split_ratios,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            neg_sample_count=neg_sample_count,
            **kwargs,
        )

    # ----- metadata ------------------------------------------------------

    @property
    def dataset_name(self) -> str:
        return f"taac2025_{self._version}"

    @property
    def dataset_url(self) -> str:
        return f"https://huggingface.co/datasets/{self._repo_id}"

    @property
    def feature_cols(self) -> List[str]:
        return _SEQ_FEATURE_COLS + ["user_feat", "item_feat", "mm_emb"]

    @property
    def label_col(self) -> str:
        return "label"

    @property
    def num_users(self) -> int:
        return self._num_users

    @property
    def num_items(self) -> int:
        return self._num_items

    # ----- loading -------------------------------------------------------

    def _load_raw(self) -> Dict[str, Any]:
        """Load the HF dataset subsets and return raw dict."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "huggingface datasets is required. "
                "Install with: pip install datasets"
            ) from None

        logger.info("Loading %s from HuggingFace …", self._repo_id)

        # Load core subsets
        ds_seq = load_dataset(
            self._repo_id, "seq", split="train", cache_dir=self.root_dir
        )
        ds_candidate = load_dataset(
            self._repo_id, "candidate", split="train", cache_dir=self.root_dir
        )
        ds_user_feat = load_dataset(
            self._repo_id, "user_feat", split="train", cache_dir=self.root_dir
        )
        ds_item_feat = load_dataset(
            self._repo_id, "item_feat", split="train", cache_dir=self.root_dir
        )

        return {
            "seq": ds_seq,
            "candidate": ds_candidate,
            "user_feat": ds_user_feat,
            "item_feat": ds_item_feat,
        }

    def _prepare_splits(
        self, raw: Dict[str, Any]
    ) -> Tuple[Dataset[Any], Dataset[Any], Dataset[Any]]:
        """Build user sequences from seq subset, then split by user."""
        ds_seq = raw["seq"]

        # Build user → item sequence mapping
        user_seqs: Dict[int, List[Dict[str, Any]]] = {}
        all_items: set = set()

        for row in ds_seq:
            # Determine user_id from user_feat or use retrieval_id as proxy
            # The "seq" subset rows are per-user interactions;
            # we group by implicit row index as user proxy for now.
            # In production, this should use a proper user_id from user_feat.
            uid = row.get("user_id") if "user_id" in row else hash(tuple(row.items())) % (10**7)
            if isinstance(uid, dict):
                uid = uid.get("feature_value", 0)
            uid = int(uid) if uid is not None else 0

            item_id = row.get("item_id", 0)
            if isinstance(item_id, dict):
                item_id = int(item_id.get("feature_value", 0))
            else:
                item_id = int(item_id) if item_id is not None else 0

            all_items.add(item_id)
            user_seqs.setdefault(uid, []).append(item_id)

        # Filter users with enough interactions
        user_ids: List[int] = []
        item_sequences: List[List[int]] = []
        for uid, items in user_seqs.items():
            if len(items) >= self.min_seq_len:
                user_ids.append(uid)
                item_sequences.append(items)

        self._num_users = len(user_ids)
        self._num_items = len(all_items)
        logger.info(
            "Built %d user sequences across %d unique items.",
            self._num_users,
            self._num_items,
        )

        # Build labeled sequences (each position predicts the next item)
        sequences: List[Dict[str, Any]] = []
        for items in item_sequences:
            for pos in range(1, len(items)):
                sequences.append({
                    "item_ids": np.array(items[:pos], dtype=np.int64),
                    "labels": np.array(items[1: pos + 1], dtype=np.int64),
                })

        # Shuffle and split
        rng = np.random.default_rng(42)
        rng.shuffle(sequences)  # type: ignore[arg-type]

        n = len(sequences)
        n_train = int(n * self.split_ratios[0])
        n_val = int(n * self.split_ratios[1])

        train_seqs = sequences[:n_train]
        val_seqs = sequences[n_train: n_train + n_val]
        test_seqs = sequences[n_train + n_val:]

        logger.info(
            "Splits: train=%d, val=%d, test=%d", len(train_seqs), len(val_seqs), len(test_seqs)
        )

        item_pool = torch.as_tensor(sorted(all_items), dtype=torch.long)

        train_ds = _SequenceSplit(
            train_seqs, item_pool=item_pool,
            max_seq_len=self.max_seq_len, neg_sample_count=self.neg_sample_count,
        )
        val_ds = _SequenceSplit(
            val_seqs, item_pool=item_pool,
            max_seq_len=self.max_seq_len, neg_sample_count=self.neg_sample_count,
        )
        test_ds = _SequenceSplit(
            test_seqs, item_pool=item_pool,
            max_seq_len=self.max_seq_len, neg_sample_count=self.neg_sample_count,
        )

        return train_ds, val_ds, test_ds

    # ----- iteration -----------------------------------------------------

    def __len__(self) -> int:
        if self._train is None:
            raise RuntimeError("Dataset not loaded. Call .load() first.")
        return (
            len(self._train)
            + len(self._val or [])
            + len(self._test or [])
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "TAAC2025Dataset uses per-split SequenceSplit datasets."
            " Use .get_dataloader(split='train') or .get_split(split)."
        )


# Auto-register both variants
from recsys.core.registry import DATASET_REGISTRY  # noqa: E402


@DATASET_REGISTRY.register(
    "taac2025_1M",
    family="generative",
    modality=("tabular", "text", "image", "video"),
    tasks=("ctr", "cvr", "retrieval", "ranking"),
)
class TAAC2025Dataset1M(TAAC2025Dataset):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("version", None)  # avoid duplicate kwarg
        super().__init__(version="1M", **kwargs)


@DATASET_REGISTRY.register(
    "taac2025_10M",
    family="generative",
    modality=("tabular", "text", "image", "video"),
    tasks=("ctr", "cvr", "retrieval", "ranking"),
)
class TAAC2025Dataset10M(TAAC2025Dataset):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("version", None)  # avoid duplicate kwarg
        super().__init__(version="10M", **kwargs)
