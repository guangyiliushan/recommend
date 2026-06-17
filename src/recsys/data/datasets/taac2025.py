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
from dataclasses import dataclass, field
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

# ---- Known subset configs on HuggingFace ----
_KNOWN_SUBSETS: Dict[str, Dict[str, Any]] = {
    "seq": {
        "hf_config": "seq",
        "primary_key": "user_id",
        "join_key": None,
        "recommended_profile": "behavior",
        "is_vector": False,
        "description": "User behavior sequences (main interaction data)",
    },
    "user_feat": {
        "hf_config": "user_feat",
        "primary_key": "user_id",
        "join_key": "user_id",
        "recommended_profile": "feature",
        "is_vector": False,
        "description": "User-side features (demographic / profile)",
    },
    "item_feat": {
        "hf_config": "item_feat",
        "primary_key": "item_id",
        "join_key": "item_id",
        "recommended_profile": "feature",
        "is_vector": False,
        "description": "Item-side features (categorical + text)",
    },
    "candidate": {
        "hf_config": "candidate",
        "primary_key": "item_id",
        "join_key": "item_id",
        "recommended_profile": "candidate",
        "is_vector": False,
        "description": "Candidate item pool for retrieval evaluation",
    },
    "mm_emb_text": {
        "hf_config": "mm_emb_text",
        "primary_key": "item_id",
        "join_key": "item_id",
        "recommended_profile": "vector",
        "is_vector": True,
        "vector_dim": None,  # discovered at load time
        "description": "Text modality embeddings",
    },
    "mm_emb_image": {
        "hf_config": "mm_emb_image",
        "primary_key": "item_id",
        "join_key": "item_id",
        "recommended_profile": "vector",
        "is_vector": True,
        "vector_dim": None,
        "description": "Image modality embeddings",
    },
    "mm_emb_video": {
        "hf_config": "mm_emb_video",
        "primary_key": "item_id",
        "join_key": "item_id",
        "recommended_profile": "vector",
        "is_vector": True,
        "vector_dim": None,
        "description": "Video modality embeddings",
    },
}


# ============================================================================
# Schema-aware data structures
# ============================================================================


@dataclass
class SubsetDescriptor:
    """Metadata descriptor for a single dataset subset.

    Does NOT hold any data references — pure metadata suitable for caching.
    """

    name: str  # "seq" | "user_feat" | "item_feat" | "candidate" | "mm_emb_text" ...
    hf_config: str  # HuggingFace config name
    primary_key: str  # e.g. "user_id" / "item_id"
    join_key: Optional[str]  # key for joining with seq table
    estimated_rows: int  # row count estimate (from HF metadata or first load)
    recommended_profile: str  # "behavior" | "feature" | "candidate" | "vector"
    columns: List[str] = field(default_factory=list)
    is_vector: bool = False
    vector_dim: Optional[int] = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hf_config": self.hf_config,
            "primary_key": self.primary_key,
            "join_key": self.join_key,
            "estimated_rows": self.estimated_rows,
            "recommended_profile": self.recommended_profile,
            "columns": self.columns,
            "is_vector": self.is_vector,
            "vector_dim": self.vector_dim,
            "description": self.description,
        }

    @staticmethod
    def from_dict(d: dict) -> "SubsetDescriptor":
        return SubsetDescriptor(
            name=d["name"],
            hf_config=d["hf_config"],
            primary_key=d["primary_key"],
            join_key=d.get("join_key"),
            estimated_rows=d.get("estimated_rows", 0),
            recommended_profile=d.get("recommended_profile", "behavior"),
            columns=d.get("columns", []),
            is_vector=d.get("is_vector", False),
            vector_dim=d.get("vector_dim"),
            description=d.get("description", ""),
        )


@dataclass
class DatasetSchemaManifest:
    """Dataset schema manifest — pure metadata, no data references.

    Cached as JSON to avoid repeated HuggingFace metadata queries.
    """

    dataset_id: str  # "taac2025_1M" | "taac2025_10M"
    version: str  # "1M" | "10M"
    subsets: Dict[str, SubsetDescriptor] = field(default_factory=dict)
    default_eda_subset: str = "seq"
    supports_candidates: bool = True
    supports_vector_embeddings: bool = True
    repo_commit_hash: str = ""  # for cache invalidation

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "subsets": {k: v.to_dict() for k, v in self.subsets.items()},
            "default_eda_subset": self.default_eda_subset,
            "supports_candidates": self.supports_candidates,
            "supports_vector_embeddings": self.supports_vector_embeddings,
            "repo_commit_hash": self.repo_commit_hash,
        }

    @staticmethod
    def from_dict(d: dict) -> "DatasetSchemaManifest":
        subsets = {
            k: SubsetDescriptor.from_dict(v)
            for k, v in d.get("subsets", {}).items()
        }
        return DatasetSchemaManifest(
            dataset_id=d["dataset_id"],
            version=d["version"],
            subsets=subsets,
            default_eda_subset=d.get("default_eda_subset", "seq"),
            supports_candidates=d.get("supports_candidates", True),
            supports_vector_embeddings=d.get("supports_vector_embeddings", True),
            repo_commit_hash=d.get("repo_commit_hash", ""),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> Optional["DatasetSchemaManifest"]:
        if not path.exists():
            return None
        try:
            return DatasetSchemaManifest.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except Exception:
            return None

    def list_subsets(self) -> List[str]:
        return sorted(self.subsets.keys())

    def get_subset(self, name: str) -> Optional[SubsetDescriptor]:
        return self.subsets.get(name)

    def auto_profile(self, subset: str) -> str:
        """Infer analysis profile from subset name."""
        desc = self.subsets.get(subset)
        if desc:
            return desc.recommended_profile
        return "behavior"


@dataclass
class VectorStore:
    """Lightweight vector store — does not hold full data in memory.

    Stores pre-sampled vectors + item_id index for EDA analysis.
    """

    item_ids: np.ndarray  # (N,) item_id array
    vectors: np.ndarray  # (N, dim) vector matrix (pre-sampled)
    dim: int  # vector dimension
    modality: str  # "text" | "image" | "video"
    original_count: int  # total vectors before sampling
    sampled_at_load: bool = False

    def __post_init__(self):
        self.item_ids = np.asarray(self.item_ids, dtype=np.int64)
        self.vectors = np.asarray(self.vectors, dtype=np.float32)

    def norms(self) -> np.ndarray:
        """L2 norms of all vectors."""
        return np.linalg.norm(self.vectors, axis=1)

    def duplicate_ratio(self) -> float:
        """Fraction of vectors that are exact duplicates."""
        if len(self.vectors) <= 1:
            return 0.0
        # Compare via hash of rounded bytes (fast approximate)
        hashes = set()
        duplicates = 0
        for row in self.vectors:
            key = row.round(decimals=4).tobytes()
            if key in hashes:
                duplicates += 1
            else:
                hashes.add(key)
        return duplicates / len(self.vectors)

    def dim_stats(self) -> dict:
        """Per-dimension statistics (mean, std, min, max)."""
        dim_means = self.vectors.mean(axis=0)
        dim_stds = self.vectors.std(axis=0)
        return {
            "mean": round(float(dim_means.mean()), 6),
            "std": round(float(dim_stds.mean()), 6),
            "min_mean": round(float(dim_means.min()), 6),
            "max_mean": round(float(dim_means.max()), 6),
            "dim": self.dim,
        }

    def to_dataframe(self):
        """Convert to a pandas DataFrame for tabular EDA."""
        import pandas as pd

        return pd.DataFrame({"item_id": self.item_ids, "norm": self.norms()})


# ============================================================================
# Helpers
# ============================================================================


def _estimate_subset_rows(repo_id: str, config: str, cache_dir: str) -> int:
    """Estimate row count for a HuggingFace dataset config without full load.

    Returns 0 if the config is unavailable or cannot be queried.
    """
    try:
        from datasets import get_dataset_split_names

        splits = get_dataset_split_names(repo_id, config)
        if not splits:
            return 0
        # Use the first available split for estimation
        from datasets import load_dataset

        ds = load_dataset(repo_id, config, split=splits[0], cache_dir=cache_dir)
        return len(ds)
    except Exception:
        return 0


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
        candidate_pool: Optional[torch.Tensor] = None,
    ) -> None:
        self._user_ids = user_ids
        self._item_sequences = item_sequences
        self._item_pool = item_pool
        self.max_seq_len = max_seq_len
        self.neg_sample_count = neg_sample_count
        self._candidate_pool = candidate_pool

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

        # Candidate items: use loaded candidate pool if available,
        # capped at 100 for memory safety in evaluation batches.
        if self._candidate_pool is not None:
            candidate_items = self._candidate_pool[:100]
        else:
            candidate_items = torch.as_tensor(np.array([], dtype=np.int64), dtype=torch.long)

        return {
            "user_id": torch.as_tensor(uid, dtype=torch.long),
            "item_id": torch.as_tensor(
                int(item_ids[0]) if len(item_ids) > 0 else 0, dtype=torch.long
            ),
            "item_ids": torch.as_tensor(item_ids, dtype=torch.long),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "candidate_items": candidate_items,
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
        """Available feature columns (grouped by subset for clarity)."""
        manifest = self.get_schema_manifest()
        return manifest.list_subsets()

    @property
    def label_col(self) -> str:
        return "label"

    @property
    def num_users(self) -> int:
        return self._num_users

    @property
    def num_items(self) -> int:
        return self._num_items

    # ----- schema manifest ----------------------------------------------

    def get_schema_manifest(self) -> DatasetSchemaManifest:
        """Return (or load from cache) the dataset schema manifest.

        First call queries HuggingFace metadata for row counts;
        subsequent calls load from local JSON cache.
        Cache path: {root_dir}/taac2025_cache/manifest_{version}.json
        """
        cache_dir = Path(self.root_dir) / "taac2025_cache"
        cache_path = cache_dir / f"manifest_{self._version}.json"

        # Try cache first
        cached = DatasetSchemaManifest.load(cache_path)
        if cached is not None:
            return cached

        # Build fresh
        manifest = self._build_manifest()
        manifest.save(cache_path)
        logger.info("Schema manifest cached to %s", cache_path)
        return manifest

    def _build_manifest(self) -> DatasetSchemaManifest:
        """Build manifest from known subset list + HF metadata queries."""
        manifest = DatasetSchemaManifest(
            dataset_id=f"taac2025_{self._version}",
            version=self._version,
            default_eda_subset="seq",
            supports_candidates=True,
            supports_vector_embeddings=True,
        )

        # Try to get commit hash for cache invalidation
        try:
            from datasets import get_dataset_config_names

            configs = get_dataset_config_names(self._repo_id)
            logger.debug("Available HF configs for %s: %s", self._repo_id, configs)
        except Exception:
            configs = list(_KNOWN_SUBSETS.keys())

        for name, info in _KNOWN_SUBSETS.items():
            hf_config = info["hf_config"]
            # Only include subsets known to exist (or try anyway for vector)
            if info.get("is_vector") or hf_config in ("seq", "user_feat", "item_feat", "candidate"):
                try:
                    est_rows = _estimate_subset_rows(self._repo_id, hf_config, self.root_dir)
                except Exception:
                    est_rows = 0

                manifest.subsets[name] = SubsetDescriptor(
                    name=name,
                    hf_config=hf_config,
                    primary_key=info["primary_key"],
                    join_key=info.get("join_key"),
                    estimated_rows=est_rows,
                    recommended_profile=info["recommended_profile"],
                    is_vector=info.get("is_vector", False),
                    vector_dim=info.get("vector_dim"),
                    description=info["description"],
                )

        return manifest

    # ----- lazy subset loading ------------------------------------------

    def load_subset(
        self,
        name: str,
        max_rows: int = 500_000,
        seed: int = 42,
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Load a single subset lazily, returning (data, load_meta).

        Returns:
            - For tabular subsets: (pd.DataFrame, load_meta dict or None)
            - For vector subsets: (VectorStore, load_meta dict or None)

        Never loads other subsets simultaneously — memory-safe.
        """
        manifest = self.get_schema_manifest()
        desc = manifest.get_subset(name)

        if desc is None:
            raise ValueError(
                f"Unknown subset '{name}'. Available: {manifest.list_subsets()}"
            )

        if desc.is_vector:
            return self._load_subset_mm_emb(name, max_rows, seed)

        return self._load_subset_tabular(desc.hf_config, max_rows, seed)

    def _load_subset_tabular(
        self,
        hf_config: str,
        max_rows: int,
        seed: int,
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Load a tabular subset (seq/user_feat/item_feat/candidate) with pre-sampling."""
        import numpy as np
        from datasets import load_dataset

        rng = np.random.default_rng(seed)

        ds = load_dataset(
            self._repo_id, hf_config, split="train", cache_dir=self.root_dir
        )
        total = len(ds)
        load_meta: Optional[Dict[str, Any]] = None

        if total > max_rows and hasattr(ds, "select"):
            indices = sorted(rng.choice(total, max_rows, replace=False).tolist())
            ds = ds.select(indices)
            load_meta = {"original_rows": total, "sampled_at_load": True}
            logger.info(
                "Pre-sampled '%s' subset: %d → %d rows.", hf_config, total, max_rows
            )

        return ds.to_pandas(), load_meta

    def _load_subset_mm_emb(
        self,
        name: str,
        max_rows: int,
        seed: int,
    ) -> Tuple[VectorStore, Optional[Dict[str, Any]]]:
        """Load a multimodal embedding subset (mm_emb_text/image/video) as VectorStore."""
        import numpy as np
        from datasets import load_dataset

        rng = np.random.default_rng(seed)

        desc = self.get_schema_manifest().get_subset(name)
        if desc is None:
            raise ValueError(f"Unknown subset: {name}")

        modality = name.replace("mm_emb_", "")

        try:
            ds = load_dataset(
                self._repo_id, desc.hf_config, split="train", cache_dir=self.root_dir
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load mm_emb subset '{name}' "
                f"(config='{desc.hf_config}'): {e}"
            ) from e

        total = len(ds)
        load_meta: Optional[Dict[str, Any]] = None

        if total > max_rows and hasattr(ds, "select"):
            indices = sorted(rng.choice(total, max_rows, replace=False).tolist())
            ds = ds.select(indices)
            load_meta = {"original_rows": total, "sampled_at_load": True}
            logger.info(
                "Pre-sampled '%s' subset: %d → %d rows.", name, total, max_rows
            )

        df = ds.to_pandas()
        item_ids = df["item_id"].values.astype(np.int64)

        # Detect embedding column (usually named "emb" or the first float array)
        emb_col = None
        for col in df.columns:
            if col == "item_id":
                continue
            sample = df[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)) and len(sample) > 0:
                emb_col = col
                break

        if emb_col is None:
            raise RuntimeError(
                f"No embedding column found in '{name}'. "
                f"Columns: {list(df.columns)}"
            )

        vectors = np.array(df[emb_col].tolist(), dtype=np.float32)
        dim = vectors.shape[1]

        return VectorStore(
            item_ids=item_ids,
            vectors=vectors,
            dim=dim,
            modality=modality,
            original_count=total,
            sampled_at_load=(load_meta is not None and load_meta.get("sampled_at_load", False)),
        ), load_meta

    # ----- candidate pool ------------------------------------------------

    def get_candidate_pool(self) -> torch.Tensor:
        """Load the candidate item pool, cached to disk for reuse.

        Used by evaluation pipeline and negative sampling.
        Cache: {root_dir}/taac2025_cache/candidates_{version}.pt
        """
        cache_dir = Path(self.root_dir) / "taac2025_cache"
        cache_path = cache_dir / f"candidates_{self._version}.pt"

        if cache_path.exists():
            logger.info("Loading candidate pool from cache: %s", cache_path)
            return torch.load(cache_path)

        from datasets import load_dataset

        ds = load_dataset(
            self._repo_id, "candidate", split="train", cache_dir=self.root_dir
        )
        df = ds.to_pandas()
        item_ids = torch.as_tensor(df["item_id"].values, dtype=torch.long)

        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(item_ids, cache_path)
        logger.info("Cached candidate pool (%d items) to %s", len(item_ids), cache_path)
        return item_ids

    # ----- sequence cache ------------------------------------------------

    def _get_cache_path(self) -> Path:
        """Get the cache file path for preprocessed sequences."""
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
        """Return manifest + seq lazy handle (backward-compatible with _prepare_splits).

        Previously returned only {"seq": ds_seq}; now also includes manifest
        so that EDA/subset-aware code paths can discover the full schema.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "huggingface datasets is required. "
                "Install with: pip install datasets"
            ) from None

        manifest = self.get_schema_manifest()
        logger.info("Loading %s seq split from HuggingFace …", self._repo_id)
        ds_seq = load_dataset(
            self._repo_id, "seq", split="train", cache_dir=self.root_dir
        )
        return {"manifest": manifest, "seq": ds_seq}

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

        # Load candidate pool lazily (used for evaluation)
        candidate_pool = self.get_candidate_pool()

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
            candidate_pool=candidate_pool,
        )
        val_ds = _SequenceSplit(
            user_ids=np.array(val_user_ids, dtype=np.int64),
            item_sequences=val_seqs,
            item_pool=item_pool,
            max_seq_len=self.max_seq_len,
            neg_sample_count=self.neg_sample_count,
            candidate_pool=candidate_pool,
        )
        test_ds = _SequenceSplit(
            user_ids=np.array(test_user_ids, dtype=np.int64),
            item_sequences=test_seqs,
            item_pool=item_pool,
            max_seq_len=self.max_seq_len,
            neg_sample_count=self.neg_sample_count,
            candidate_pool=candidate_pool,
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
    supports_multi_subset=True,
    supports_candidates=True,
    supports_vector_embeddings=True,
    default_eda_subset="seq",
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
    supports_multi_subset=True,
    supports_candidates=True,
    supports_vector_embeddings=True,
    default_eda_subset="seq",
)
class TAAC2025Dataset10M(TAAC2025Dataset):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("version", None)  # avoid duplicate kwarg
        super().__init__(version="10M", **kwargs)
