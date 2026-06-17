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
    """Lazy sequence split — computes labeled samples on demand.

    Instead of pre-expanding all positions into a flat list (which causes OOM
    for 1M+ users), we store only the compact user→items mapping and compute
    each (user_id, item_ids[:pos], labels[1:pos+1]) sample in __getitem__.

    Memory: O(num_users) instead of O(total_interactions * avg_seq_len).
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_sequences: List[np.ndarray],
        item_pool: Optional[torch.Tensor] = None,
        max_seq_len: int = 50,
        neg_sample_count: int = 4,
    ) -> None:
        self._user_ids = user_ids
        self._item_sequences = item_sequences
        self._item_pool = item_pool
        self.max_seq_len = max_seq_len
        self.neg_sample_count = neg_sample_count

        # Pre-compute cumulative lengths for O(log n) index lookup.
        # Each user with seq length L contributes (L-1) labeled positions
        # (positions 1..L-1, where pos=i predicts item[i] from items[:i]).
        self._lengths = np.array(
            [max(0, len(seq) - 1) for seq in item_sequences], dtype=np.int64
        )
        self._cum_lengths = np.cumsum(self._lengths)

    def __len__(self) -> int:
        return int(self._cum_lengths[-1]) if len(self._cum_lengths) > 0 else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0:
            idx += len(self)

        # Binary search: which user does this flat index belong to?
        user_idx = int(np.searchsorted(self._cum_lengths, idx, side="right"))
        prev_cum = int(self._cum_lengths[user_idx - 1]) if user_idx > 0 else 0
        pos = idx - prev_cum + 1  # position in sequence (1-based)

        uid = int(self._user_ids[user_idx])
        items = self._item_sequences[user_idx]

        # Labeled sequence: items[:pos] → predict items[1:pos+1]
        item_ids = items[:pos][: self.max_seq_len].copy()
        labels = items[1 : pos + 1][: self.max_seq_len].copy()

        # Pad if needed
        pad_len = self.max_seq_len - len(item_ids)
        if pad_len > 0:
            item_ids = np.pad(item_ids, (0, pad_len), constant_values=0)
            labels = np.pad(labels, (0, pad_len), constant_values=-100)

        return {
            "user_id": torch.as_tensor(uid, dtype=torch.long),
            "item_id": torch.as_tensor(
                int(item_ids[0]) if len(item_ids) > 0 else 0, dtype=torch.long
            ),
            "item_ids": torch.as_tensor(item_ids, dtype=torch.long),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "candidate_items": torch.as_tensor(
                np.array([], dtype=np.int64), dtype=torch.long
            ),
        }

    # ---- Fast extraction (O(num_users) vs O(total_positions)) ----

    def iter_user_item_pairs_fast(self):
        """Yield (user_id, item_id) pairs from compact mapping — O(users).

        Yields per-user, per-item tuples. Uses numpy.tolist() internally
        to convert int64→Python int in bulk (C-level) instead of per-element.
        """
        for uid, items in zip(self._user_ids, self._item_sequences, strict=False):
            uid_int = int(uid)
            for iid in items.tolist():  # batch C→Python conversion
                yield uid_int, iid

    def extract_user_item_mapping_fast(self) -> dict:
        """Extract user_id → item_ids from compact mapping — O(users).

        Returns dict[int, np.ndarray] (not set) for zero-copy access.
        ItemCF._build_cooccurrence_matrix accepts both set and ndarray.
        """
        return {
            int(uid): items  # keep as np.ndarray, no conversion
            for uid, items in zip(self._user_ids, self._item_sequences, strict=False)
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

    def _load_from_cache(self, cache_path: Path) -> Optional[Tuple[Dict, np.ndarray, List[np.ndarray]]]:
        """Load preprocessed compact sequences from cache if exists.

        Returns (meta, user_ids, item_sequences) or None.
        """
        if not cache_path.exists():
            return None
        try:
            data = np.load(cache_path, allow_pickle=True)
            meta = json.loads(str(data["meta"]))
            user_ids = data["user_ids"]
            item_sequences = data["item_sequences"].tolist()
            logger.info("Loaded preprocessed sequences from cache: %s", cache_path)
            return meta, user_ids, item_sequences
        except Exception as e:
            logger.warning("Failed to load cache: %s", e)
            return None

    def _save_to_cache(
        self,
        cache_path: Path,
        meta: Dict,
        user_ids: np.ndarray,
        item_sequences: List[np.ndarray],
    ) -> None:
        """Save preprocessed compact sequences to cache."""
        try:
            np.savez(
                cache_path,
                meta=np.array(json.dumps(meta)),
                user_ids=user_ids,
                item_sequences=np.array(item_sequences, dtype=object),
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

        Memory optimization:
            - Uses Arrow columnar extraction (no to_pydict deserialization)
            - Stores compact user→items mapping instead of expanded sequences
            - _SequenceSplit computes labeled samples on demand via __getitem__
            - Cache stores compact mapping (1M entries vs 20M expanded dicts)
        """
        # Check cache first
        import pyarrow as pa

        cache_path = self._get_cache_path()
        cached = self._load_from_cache(cache_path)
        if cached:
            meta, user_ids_np, item_sequences = cached
            self._num_users = meta.get("num_users", 0)
            self._num_items = meta.get("num_items", 0)
            all_items = set(meta.get("all_items", []))
            # Ensure item_sequences are numpy arrays
            item_sequences = [np.array(s, dtype=np.int64) for s in item_sequences]
        else:
            # Process raw data
            ds_seq = raw["seq"]
            total = len(ds_seq)
            logger.info("Processing %d user sequences (first run, caching results)...", total)

            # Build user → item sequence mapping
            # Strategy: use pure Arrow columnar operations to extract item_id
            # from nested List<Struct<item_id, ...>> without deserializing to
            # Python dicts — this avoids the memory explosion from to_pydict().
            user_seqs: Dict[int, List[int]] = {}
            all_items: Set[int] = set()

            arrow_table = ds_seq.data
            n_rows = len(arrow_table)

            # Get the two columns we need
            user_id_col = arrow_table.column("user_id")
            seq_col = arrow_table.column("seq")

            # Flatten the List<Struct> to access the struct fields directly.
            # seq_col is ChunkedArray<List<Struct<item_id: int64, ...>>>
            # Concatenate chunks into a single array, then get offsets + values.
            seq_chunks = seq_col.chunks
            seq_combined = seq_chunks[0] if len(seq_chunks) == 1 else pa.concat_arrays(seq_chunks)
            offsets = seq_combined.offsets.to_numpy(zero_copy_only=False)
            # The values (flat struct array) contains all struct elements
            flat_values = seq_combined.values

            # Extract just the item_id field from the flat struct array
            # This is a pure Arrow operation — no Python dict creation
            flat_item_ids = flat_values.field("item_id").to_numpy(zero_copy_only=False)

            # Also get user_ids as numpy
            user_ids_all = user_id_col.to_numpy(zero_copy_only=False)

            # Now iterate per-user using offsets (no Python dict creation)
            for i in range(n_rows):
                uid = int(user_ids_all[i])
                start = int(offsets[i])
                end_offset = int(offsets[i + 1])
                if start == end_offset:
                    continue
                # Slice the item_id numpy array for this user's sequence
                user_item_ids = flat_item_ids[start:end_offset].tolist()
                user_seqs[uid] = user_item_ids
                all_items.update(user_item_ids)

                if i % 200_000 == 0 and i > 0:
                    logger.debug(
                        "Processed %d / %d users (%d items)",
                        i, n_rows, len(all_items),
                    )

            # Free Arrow memory eagerly
            del arrow_table, seq_combined, flat_values, flat_item_ids, offsets, user_ids_all

            # Filter users with enough interactions
            user_ids_list: List[int] = []
            item_sequences_list: List[np.ndarray] = []
            for uid, items in user_seqs.items():
                if len(items) >= self.min_seq_len:
                    user_ids_list.append(uid)
                    item_sequences_list.append(np.array(items, dtype=np.int64))
            del user_seqs  # free dict

            self._num_users = len(user_ids_list)
            self._num_items = len(all_items)
            logger.info(
                "Built %d user sequences across %d unique items.",
                self._num_users,
                self._num_items,
            )

            # Shuffle users (not expanded sequences — much smaller)
            rng = np.random.default_rng(42)
            indices = np.arange(len(user_ids_list))
            rng.shuffle(indices)
            user_ids_np = np.array(user_ids_list, dtype=np.int64)[indices]
            item_sequences = [item_sequences_list[i] for i in indices]
            del user_ids_list, item_sequences_list

            # Save compact mapping to cache (1M entries vs 20M expanded dicts)
            meta = {
                "num_users": self._num_users,
                "num_items": self._num_items,
                "all_items": list(all_items),
            }
            self._save_to_cache(cache_path, meta, user_ids_np, item_sequences)

        total_positions = sum(max(0, len(s) - 1) for s in item_sequences)
        n_train = int(total_positions * self.split_ratios[0])
        n_val = int(total_positions * self.split_ratios[1])

        logger.info(
            "Splits: train=%d, val=%d, test=%d (total positions: %d)",
            n_train, n_val, total_positions - n_train - n_val, total_positions,
        )

        # Split by user for memory efficiency — each user goes to one split only.
        # This avoids creating overlapping sequence views.
        item_pool = torch.as_tensor(sorted(all_items), dtype=torch.long)

        # Assign users to splits based on cumulative position counts
        cum = 0
        train_user_ids: List[int] = []
        train_seqs: List[np.ndarray] = []
        val_user_ids: List[int] = []
        val_seqs: List[np.ndarray] = []
        test_user_ids: List[int] = []
        test_seqs: List[np.ndarray] = []

        for uid, seq in zip(user_ids_np, item_sequences, strict=False):
            n_pos = max(0, len(seq) - 1)
            if n_pos == 0:
                continue
            cum += n_pos
            if cum <= n_train:
                train_user_ids.append(int(uid))
                train_seqs.append(seq)
            elif cum <= n_train + n_val:
                val_user_ids.append(int(uid))
                val_seqs.append(seq)
            else:
                test_user_ids.append(int(uid))
                test_seqs.append(seq)

        train_ds = _SequenceSplit(
            user_ids=np.array(train_user_ids, dtype=np.int64),
            item_sequences=train_seqs,
            item_pool=item_pool,
            max_seq_len=self.max_seq_len,
            neg_sample_count=self.neg_sample_count,
        )
        val_ds = _SequenceSplit(
            user_ids=np.array(val_user_ids, dtype=np.int64),
            item_sequences=val_seqs,
            item_pool=item_pool,
            max_seq_len=self.max_seq_len,
            neg_sample_count=self.neg_sample_count,
        )
        test_ds = _SequenceSplit(
            user_ids=np.array(test_user_ids, dtype=np.int64),
            item_sequences=test_seqs,
            item_pool=item_pool,
            max_seq_len=self.max_seq_len,
            neg_sample_count=self.neg_sample_count,
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
