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

Optimization:
    Preprocessed sequences are cached to local Parquet files for fast reload.
    First run: processes raw HF data and saves cache.
    Subsequent runs: loads from cache in seconds.

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

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from recsys.core.base_dataset import BaseDataset

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
        user_id = seq["user_id"]
        candidate_items = seq.get("candidate_items", np.array([], dtype=np.int64))

        # Pad if needed
        pad_len = self.max_seq_len - len(item_ids)
        if pad_len > 0:
            item_ids = np.pad(item_ids, (0, pad_len), constant_values=0)
            labels = np.pad(labels, (0, pad_len), constant_values=-100)

        return {
            "user_id": torch.as_tensor(user_id, dtype=torch.long),
            "item_id": torch.as_tensor(
                item_ids[0] if len(item_ids) > 0 else 0, dtype=torch.long
            ),
            "item_ids": torch.as_tensor(item_ids, dtype=torch.long),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "candidate_items": torch.as_tensor(candidate_items, dtype=torch.long),
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

    def _get_cache_path(self) -> Path:
        """Get the cache file path for preprocessed sequences."""
        # Hash based on config to invalidate cache when settings change
        config_str = f"{self._version}_{self.split_ratios}_{self.min_seq_len}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        cache_dir = Path(self.root_dir) / "taac2025_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"sequences_{self._version}_{config_hash}.npz"

    def _load_from_cache(self, cache_path: Path) -> Optional[Tuple[Dict, Dict, Dict]]:
        """Load preprocessed sequences from cache if exists."""
        if not cache_path.exists():
            return None
        try:
            data = np.load(cache_path, allow_pickle=True)
            meta = json.loads(str(data["meta"]))
            train_data = data["train"].tolist()
            val_data = data["val"].tolist()
            test_data = data["test"].tolist()
            logger.info("Loaded preprocessed sequences from cache: %s", cache_path)
            return meta, train_data, val_data, test_data
        except Exception as e:
            logger.warning("Failed to load cache: %s", e)
            return None

    def _save_to_cache(
        self,
        cache_path: Path,
        meta: Dict,
        train_data: List,
        val_data: List,
        test_data: List,
    ) -> None:
        """Save preprocessed sequences to cache."""
        try:
            np.savez(
                cache_path,
                meta=np.array(json.dumps(meta)),
                train=np.array(train_data, dtype=object),
                val=np.array(val_data, dtype=object),
                test=np.array(test_data, dtype=object),
            )
            logger.info("Saved preprocessed sequences to cache: %s", cache_path)
        except Exception as e:
            logger.warning("Failed to save cache: %s", e)

    def _load_raw(self) -> Dict[str, Any]:
        """Load only the seq subset (lazy-load others on demand)."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "huggingface datasets is required. "
                "Install with: pip install datasets"
            ) from None

        logger.info("Loading %s seq split from HuggingFace …", self._repo_id)
        ds_seq = load_dataset(
            self._repo_id, "seq", split="train", cache_dir=self.root_dir
        )
        return {"seq": ds_seq}

    def _prepare_splits(
        self, raw: Dict[str, Any]
    ) -> Tuple[Dataset[Any], Dataset[Any], Dataset[Any]]:
        """Build user sequences from seq subset.

        TAAC2025 seq schema:
            user_id: int64
            seq: List<{item_id: int64, action_type: int32, timestamp: int64}>

        Uses caching to avoid reprocessing on subsequent runs.
        """
        # Check cache first
        cache_path = self._get_cache_path()
        cached = self._load_from_cache(cache_path)
        if cached:
            meta, train_data, val_data, test_data = cached
            self._num_users = meta.get("num_users", 0)
            self._num_items = meta.get("num_items", 0)
            all_items = set(meta.get("all_items", []))
        else:
            # Process raw data
            ds_seq = raw["seq"]
            total = len(ds_seq)
            logger.info("Processing %d user sequences (first run, caching results)...", total)

            # Build user → item sequence mapping
            user_seqs: Dict[int, List[int]] = {}
            all_items: Set[int] = set()

            # Use iter with batch_size for memory efficiency
            for batch in ds_seq.iter(batch_size=50000):
                for uid, seq_items in zip(batch["user_id"], batch["seq"], strict=False):
                    if uid is None:
                        continue
                    uid = int(uid)
                    for item in seq_items:
                        iid = int(item["item_id"])
                        user_seqs.setdefault(uid, []).append(iid)
                        all_items.add(iid)

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
            for uid, items in zip(user_ids, item_sequences, strict=False):
                for pos in range(1, len(items)):
                    sequences.append({
                        "user_id": uid,
                        "item_ids": items[:pos],
                        "labels": items[1: pos + 1],
                    })

            if len(sequences) == 0:
                raise RuntimeError(
                    f"No valid sequences generated. "
                    f"Check min_seq_len ({self.min_seq_len}) or dataset content."
                )

            # Shuffle and split
            rng = np.random.default_rng(42)
            indices = np.arange(len(sequences))
            rng.shuffle(indices)
            sequences = [sequences[i] for i in indices]

            n = len(sequences)
            n_train = int(n * self.split_ratios[0])
            n_val = int(n * self.split_ratios[1])

            train_data = sequences[:n_train]
            val_data = sequences[n_train: n_train + n_val]
            test_data = sequences[n_train + n_val:]

            # Save to cache
            meta = {
                "num_users": self._num_users,
                "num_items": self._num_items,
                "all_items": list(all_items),
            }
            self._save_to_cache(cache_path, meta, train_data, val_data, test_data)

        logger.info(
            "Splits: train=%d, val=%d, test=%d",
            len(train_data), len(val_data), len(test_data)
        )

        # Convert to numpy arrays for Dataset
        def to_numpy(seqs):
            return [
                {
                    "user_id": s["user_id"],
                    "item_ids": np.array(s["item_ids"], dtype=np.int64),
                    "labels": np.array(s["labels"], dtype=np.int64),
                }
                for s in seqs
            ]

        item_pool = torch.as_tensor(sorted(all_items), dtype=torch.long)

        train_ds = _SequenceSplit(
            to_numpy(train_data), item_pool=item_pool,
            max_seq_len=self.max_seq_len, neg_sample_count=self.neg_sample_count,
        )
        val_ds = _SequenceSplit(
            to_numpy(val_data), item_pool=item_pool,
            max_seq_len=self.max_seq_len, neg_sample_count=self.neg_sample_count,
        )
        test_ds = _SequenceSplit(
            to_numpy(test_data), item_pool=item_pool,
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
