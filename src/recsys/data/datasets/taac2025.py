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

import gc
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset

from recsys.core.base_dataset import BaseDataset
from recsys.data.split_utils import SequenceSplit as _SequenceSplit  # noqa: F401

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


def _list_dataset_configs(repo_id: str) -> List[str]:
    """List available HuggingFace dataset configs, returning [] on failure."""
    try:
        from datasets import get_dataset_config_names

        return list(get_dataset_config_names(repo_id))
    except Exception:
        return []


def _is_hf_offline_mode() -> bool:
    """Return True when HuggingFace offline mode is explicitly enabled."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _list_cached_dataset_configs(root_dir: str, version: str) -> Set[str]:
    """List locally cached TAAC2025 subset configs with complete dataset metadata."""
    cache_root = Path(root_dir)
    version_suffix = version.lower().replace("m", "_m")
    dataset_roots = [
        path
        for path in cache_root.glob("TAAC2025___tencent_gr-*")
        if path.is_dir() and path.name.endswith(version_suffix)
    ]

    cached_configs: Set[str] = set()
    for dataset_root in dataset_roots:
        for config_dir in dataset_root.iterdir():
            if not config_dir.is_dir():
                continue
            has_dataset_info = any(
                config_dir.glob("0.0.0/*/dataset_info.json")
            )
            if has_dataset_info:
                cached_configs.add(config_dir.name)
    return cached_configs


def _find_cached_dataset_info(
    root_dir: str, version: str, config: str
) -> Optional[Path]:
    """Find the cached dataset_info.json path for a TAAC2025 config."""
    cache_root = Path(root_dir)
    version_suffix = version.lower().replace("m", "_m")
    for dataset_root in cache_root.glob("TAAC2025___tencent_gr-*"):
        if not dataset_root.is_dir() or not dataset_root.name.endswith(version_suffix):
            continue
        matches = list(dataset_root.glob(f"{config}/0.0.0/*/dataset_info.json"))
        if matches:
            return matches[0]
    return None


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
        self.item_ids = np.asarray(self.item_ids)
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
    version = repo_id.rsplit("-", 1)[-1]
    cached_info = _find_cached_dataset_info(cache_dir, version, config)
    if cached_info is not None:
        try:
            data = json.loads(cached_info.read_text(encoding="utf-8"))
            splits = data.get("splits", {})
            for split in splits.values():
                num_examples = split.get("num_examples")
                if isinstance(num_examples, int) and num_examples > 0:
                    return num_examples
        except Exception:
            pass

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


# ============================================================================
# HyFormer 结构化特征支持 — 零拷贝特征加载 + 结构化 Split
# ============================================================================

# TAAC2025 user_feat 列定义
_USER_FEAT_SCALAR_COLS = ["103", "104", "105", "109"]  # int64 标量
_USER_FEAT_LIST_COLS = ["106", "107", "108", "110"]     # List[int64] 多值
_USER_FEAT_ALL_COLS = _USER_FEAT_SCALAR_COLS + _USER_FEAT_LIST_COLS

# TAAC2025 item_feat 列定义 — 12 个 int64 标量
_ITEM_FEAT_COLS = [
    "100", "101", "102", "112", "114", "115", "116",
    "117", "118", "119", "120", "121", "122",
]

# HyFormer 序列域配置：行为序列域，sideinfo = [item_id, action_type]
_SEQ_DOMAIN_NAME = "behavior"
_SEQ_SIDEINFO_COLS = 2  # item_id + action_type


def _log_memory(fmt: str, *args: Any) -> None:
    """Log memory usage for OOM diagnosis. Only logs if psutil is available."""
    try:
        import psutil
        proc = psutil.Process()
        mem_gb = proc.memory_info().rss / (1024 ** 3)
        logger.debug(fmt + "  [mem=%.1f GB]", *args, mem_gb)
    except Exception:
        logger.debug(fmt, *args)


def _extract_scalar_int_col(arrow_table, col_name: str) -> np.ndarray:
    """从 Arrow 表中提取标量 int64 列为 numpy 数组（零拷贝优先）。"""
    col = arrow_table.column(col_name)
    # 合并可能的多个 chunk 为单个数组
    arr = col.chunks[0] if len(col.chunks) == 1 else pa.concat_arrays(col.chunks)
    return arr.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _extract_list_int_col(
    arrow_table, col_name: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """从 Arrow 表中提取 List<int64> 列。

    Returns:
        (flat_values, offsets, max_len)
        - flat_values: (total_elements,) 展平后的所有 int 值
        - offsets: (n_rows + 1,) 每行的起止偏移
        - max_len: 所有行中最大列表长度
    """
    col = arrow_table.column(col_name)
    list_arr = col.chunks[0] if len(col.chunks) == 1 else pa.concat_arrays(col.chunks)
    offsets = list_arr.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    flat_vals = list_arr.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    # 计算最大列表长度
    max_len = int((offsets[1:] - offsets[:-1]).max()) if len(offsets) > 1 else 0
    return flat_vals, offsets, max_len


def _build_feature_specs_from_arrays(
    scalar_max_vals: Dict[str, int],
    list_max_vals: Dict[str, int],
    list_max_lens: Dict[str, int],
    scalar_cols: List[str],
    list_cols: List[str],
    max_vocab_cap: int = 500_000,
) -> Tuple[List[Tuple[int, int, int]], int]:
    """从预扫描的最大值构建 feature_specs。

    feature_specs = [(vocab_size, offset, length), ...]
    顺序：scalar_cols（length=1），然后 list_cols（length = max_list_len）。

    Returns:
        (feature_specs, total_int_dim)
    """
    specs: List[Tuple[int, int, int]] = []
    offset = 0
    for col in scalar_cols:
        raw_vocab = scalar_max_vals.get(col, 0) + 1
        vocab = min(raw_vocab, max_vocab_cap + 1)
        specs.append((vocab, offset, 1))
        offset += 1
    for col in list_cols:
        raw_vocab = list_max_vals.get(col, 0) + 1
        vocab = min(raw_vocab, max_vocab_cap + 1)
        length = list_max_lens.get(col, 1)
        specs.append((vocab, offset, length))
        offset += length
    return specs, offset


class _StructuredSequenceSplit(Dataset[Dict[str, torch.Tensor]]):
    """惰性序列 split — 产出 HyFormer 结构化特征键。

    与 SequenceSplit 相同的内存优化策略（O(num_users) 存储），
    额外产出 user_int_feats / item_int_feats / seq_data / seq_lens。

    Memory: O(num_users + num_items_in_pool) 用于特征查找表。
    """

    def __init__(  # noqa: PLR0913
        self,
        user_ids: np.ndarray,
        item_sequences: List[np.ndarray],
        action_sequences: List[np.ndarray],
        item_pool: Optional[torch.Tensor] = None,
        max_seq_len: int = 50,
        neg_sample_count: int = 4,
        candidate_pool: Optional[torch.Tensor] = None,
        # ---- 结构化特征 ----
        user_feat_dict: Optional[Dict[int, np.ndarray]] = None,
        item_feat_dict: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        self._user_ids = user_ids
        self._item_sequences = item_sequences
        self._action_sequences = action_sequences
        self._item_pool = item_pool
        self.max_seq_len = max_seq_len
        self.neg_sample_count = neg_sample_count
        self._candidate_pool = candidate_pool
        self._user_feat_dict = user_feat_dict or {}
        self._item_feat_dict = item_feat_dict or {}

        # 预计算累计长度用于 O(log n) 索引查找
        self._lengths = np.array(
            [max(0, len(seq) - 1) for seq in item_sequences], dtype=np.int64,
        )
        self._cum_lengths = np.cumsum(self._lengths)

    def __len__(self) -> int:
        return int(self._cum_lengths[-1]) if len(self._cum_lengths) > 0 else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0:
            idx += len(self)

        # 二分查找定位用户
        user_idx = int(np.searchsorted(self._cum_lengths, idx, side="right"))
        prev_cum = int(self._cum_lengths[user_idx - 1]) if user_idx > 0 else 0
        pos = idx - prev_cum + 1  # 序列中的位置（1-based）

        uid = int(self._user_ids[user_idx])
        items = self._item_sequences[user_idx]
        actions = self._action_sequences[user_idx]

        # 带标签序列：items[:pos] → 预测 items[pos]
        item_ids = items[:pos][: self.max_seq_len].copy()
        labels = items[1: pos + 1][: self.max_seq_len].copy()
        action_ids = actions[:pos][: self.max_seq_len].copy()

        # Padding
        actual_len = min(len(item_ids), self.max_seq_len)
        pad_len = self.max_seq_len - len(item_ids)
        if pad_len > 0:
            item_ids = np.pad(item_ids, (0, pad_len), constant_values=0)
            labels = np.pad(labels, (0, pad_len), constant_values=-100)
            action_ids = np.pad(action_ids, (0, pad_len), constant_values=0)

        # 候选物品池
        if self._candidate_pool is not None:
            candidate_items = self._candidate_pool[:100]
        else:
            candidate_items = torch.as_tensor(
                np.array([], dtype=np.int64), dtype=torch.long,
            )

        # ---- 结构化特征 ----
        # user_int_feats
        user_feat = self._user_feat_dict.get(uid)
        if user_feat is not None:
            user_int_feats = torch.as_tensor(user_feat, dtype=torch.long)
        else:
            user_int_feats = torch.zeros(1, dtype=torch.long)

        # item_int_feats（目标物品）
        target_iid = int(item_ids[actual_len - 1]) if actual_len > 0 else 0
        item_feat = self._item_feat_dict.get(target_iid)
        if item_feat is not None:
            item_int_feats = torch.as_tensor(item_feat, dtype=torch.long)
        else:
            item_int_feats = torch.zeros(1, dtype=torch.long)

        # seq_data / seq_lens
        # behavior 域：S=2 (item_id, action_type), L=max_seq_len
        seq_data_tensor = torch.stack([
            torch.as_tensor(item_ids, dtype=torch.long),
            torch.as_tensor(action_ids, dtype=torch.long),
        ], dim=0)  # (2, L)
        seq_lens_tensor = torch.as_tensor(actual_len, dtype=torch.long)

        result: Dict[str, Any] = {
            # 旧键（向后兼容）
            "user_id": torch.as_tensor(uid, dtype=torch.long),
            "item_id": torch.as_tensor(target_iid, dtype=torch.long),
            "item_ids": torch.as_tensor(item_ids, dtype=torch.long),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "candidate_items": candidate_items,
            # HyFormer 结构化键
            "user_int_feats": user_int_feats,
            "item_int_feats": item_int_feats,
            "seq_data": {_SEQ_DOMAIN_NAME: seq_data_tensor},
            "seq_lens": {_SEQ_DOMAIN_NAME: seq_lens_tensor},
        }
        return result

    # ---- 快速提取（向后兼容） ----

    def iter_user_item_pairs_fast(self):
        """Yield (user_id, item_id) pairs."""
        for uid, items in zip(self._user_ids, self._item_sequences, strict=False):
            uid_int = int(uid)
            for iid in items.tolist():
                yield uid_int, iid

    def extract_user_item_mapping_fast(self) -> dict:
        """Extract user_id → item_ids from compact mapping."""
        return {
            int(uid): items
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
    min_action_type : int
        Minimum action_type to include as a positive interaction.
        TencentGR-1M action_type: 0=曝光(exposure), 1=点击(click).
        默认 0（包含所有行为），设为 1 仅提取点击作为正交互。
        ItemCF 隐式推荐场景建议设为 1。
    """

    def __init__(
        self,
        version: str = "10M",
        root_dir: str = "./data",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        max_seq_len: int = 50,
        min_seq_len: int = 2,
        min_action_type: int = 0,
        neg_sample_count: int = 4,
        **kwargs: Any,
    ) -> None:
        if version not in _AVAILABLE_VERSIONS:
            raise ValueError(
                f"version must be one of {_AVAILABLE_VERSIONS}, got '{version}'"
            )
        self._version = version
        self._repo_id = f"{_TAAC2025_REPO}/TencentGR-{version}"
        self._min_action_type = min_action_type
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
        offline_cached_configs = _list_cached_dataset_configs(
            self.root_dir, self._version
        )

        # Try cache first
        cached = DatasetSchemaManifest.load(cache_path)
        if cached is not None:
            if _is_hf_offline_mode():
                cached_subset_configs = set(cached.subsets.keys())
                if cached_subset_configs == offline_cached_configs:
                    return cached

                logger.info(
                    "Schema manifest cache %s does not match offline cache "
                    "for %s; rebuilding.",
                    cache_path,
                    self._repo_id,
                )
            else:
                available_configs = set(_list_dataset_configs(self._repo_id))
                if not available_configs:
                    return cached

                cached_vector_configs = {
                    desc.hf_config
                    for desc in cached.subsets.values()
                    if desc.is_vector
                }
                available_vector_configs = {
                    cfg for cfg in available_configs if cfg.startswith("mm_emb_")
                }
                if cached_vector_configs == available_vector_configs:
                    return cached

                logger.info(
                    "Schema manifest cache %s is stale for %s; rebuilding.",
                    cache_path,
                    self._repo_id,
                )

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

        cached_configs = _list_cached_dataset_configs(self.root_dir, self._version)
        configs = _list_dataset_configs(self._repo_id)
        if _is_hf_offline_mode() and cached_configs:
            available_configs = cached_configs
            logger.info(
                "Using offline cached configs for %s: %s",
                self._repo_id,
                sorted(available_configs),
            )
        else:
            available_configs = set(configs) if configs else set(_KNOWN_SUBSETS.keys())
            if configs:
                logger.debug(
                    "Available HF configs for %s: %s", self._repo_id, configs
                )

        for name in ("seq", "user_feat", "item_feat", "candidate"):
            info = _KNOWN_SUBSETS[name]
            hf_config = info["hf_config"]
            if hf_config not in available_configs:
                continue
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
                is_vector=False,
                vector_dim=info.get("vector_dim"),
                description=info["description"],
            )

        vector_configs = sorted(
            cfg for cfg in available_configs if cfg.startswith("mm_emb_")
        )
        if vector_configs:
            for cfg in vector_configs:
                manifest.subsets[cfg] = SubsetDescriptor(
                    name=cfg,
                    hf_config=cfg,
                    primary_key="item_id",
                    join_key="item_id",
                    estimated_rows=0,
                    recommended_profile="vector",
                    is_vector=True,
                    vector_dim=None,
                    description=f"Multimodal embedding subset ({cfg})",
                )
        else:
            for name in ("mm_emb_text", "mm_emb_image", "mm_emb_video"):
                info = _KNOWN_SUBSETS[name]
                manifest.subsets[name] = SubsetDescriptor(
                    name=name,
                    hf_config=info["hf_config"],
                    primary_key=info["primary_key"],
                    join_key=info.get("join_key"),
                    estimated_rows=0,
                    recommended_profile=info["recommended_profile"],
                    is_vector=True,
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
        import pandas as pd
        from datasets import load_dataset

        total = _estimate_subset_rows(self._repo_id, hf_config, self.root_dir)
        load_meta: Optional[Dict[str, Any]] = None
        if total > max_rows:
            load_meta = {"original_rows": total, "sampled_at_load": True}
            logger.info(
                "Pre-sampled '%s' subset via streaming: %d → %d rows.",
                hf_config,
                total,
                max_rows,
            )
            ds = load_dataset(
                self._repo_id,
                hf_config,
                split="train",
                streaming=True,
                cache_dir=self.root_dir,
            )
            rows = list(islice(ds, max_rows))
            return pd.DataFrame(rows), load_meta

        ds = load_dataset(
            self._repo_id, hf_config, split="train", cache_dir=self.root_dir
        )
        if total <= 0:
            total = len(ds)

        return ds.to_pandas(), load_meta

    def _load_subset_mm_emb(
        self,
        name: str,
        max_rows: int,
        seed: int,
    ) -> Tuple[VectorStore, Optional[Dict[str, Any]]]:
        """Load a multimodal embedding subset (mm_emb_text/image/video) as VectorStore."""
        import pandas as pd
        from datasets import load_dataset

        desc = self.get_schema_manifest().get_subset(name)
        if desc is None:
            raise ValueError(f"Unknown subset: {name}")

        modality = name.replace("mm_emb_", "")
        total = _estimate_subset_rows(self._repo_id, desc.hf_config, self.root_dir)
        load_meta: Optional[Dict[str, Any]] = None
        try:
            if total > max_rows:
                load_meta = {"original_rows": total, "sampled_at_load": True}
                logger.info(
                    "Pre-sampled '%s' subset via streaming: %d → %d rows.",
                    name,
                    total,
                    max_rows,
                )
                ds = load_dataset(
                    self._repo_id,
                    desc.hf_config,
                    split="train",
                    streaming=True,
                    cache_dir=self.root_dir,
                )
                rows = list(islice(ds, max_rows))
                df = pd.DataFrame(rows)
            else:
                ds = load_dataset(
                    self._repo_id,
                    desc.hf_config,
                    split="train",
                    cache_dir=self.root_dir,
                )
                if total <= 0:
                    total = len(ds)
                df = ds.to_pandas()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load mm_emb subset '{name}' "
                f"(config='{desc.hf_config}'): {e}"
            ) from e

        # Detect embedding column (usually named "emb" or the first float array)
        emb_col = None
        for col in df.columns:
            if col in {"item_id", "anonymous_cid"}:
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

        id_col = None
        for candidate in ("item_id", "anonymous_cid", "cid"):
            if candidate in df.columns:
                id_col = candidate
                break
        if id_col is None:
            scalar_cols = [
                col
                for col in df.columns
                if col != emb_col and not isinstance(df[col].iloc[0], (list, np.ndarray))
            ]
            if scalar_cols:
                id_col = scalar_cols[0]

        if id_col is None:
            logger.warning(
                "No identifier column found in vector subset '%s'; falling back to row index.",
                name,
            )
            item_ids = np.arange(len(df), dtype=np.int64)
        else:
            item_ids = df[id_col].to_numpy()

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
        """Get the cache file path for preprocessed sequences.

        Cache is shared across different split_ratios — only parameters that
        affect the *content* of preprocessed data are included in the key.

        One cache file per (version, min_seq_len, min_action_type) combination.
        This means all experiments with different train/val/test splits share
        the same cached data, saving disk and enabling offline reuse.
        """
        config_str = (
            f"{self._version}_L{self.min_seq_len}_at{self._min_action_type}"
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        cache_dir = Path(self.root_dir) / "taac2025_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"sequences_{config_hash}.npz"

    def _load_from_cache(self, cache_path: Path) -> Optional[Tuple[Dict, np.ndarray, List[np.ndarray], List[np.ndarray]]]:
        """Load preprocessed compact sequences from cache if exists.

        Returns (meta, user_ids, item_sequences, action_sequences) or None.
        action_sequences may be empty list for legacy caches.
        """
        if not cache_path.exists():
            return None
        try:
            data = np.load(cache_path, allow_pickle=True)
            meta = json.loads(str(data["meta"]))
            user_ids = data["user_ids"]
            item_sequences = data["item_sequences"].tolist()
            # 向后兼容：旧缓存可能没有 action_sequences
            if "action_sequences" in data:
                action_sequences = data["action_sequences"].tolist()
            else:
                action_sequences = []
            logger.info("Loaded preprocessed sequences from cache: %s", cache_path)
            return meta, user_ids, item_sequences, action_sequences
        except Exception as e:
            logger.warning("Failed to load cache: %s", e)
            return None

    def _save_to_cache(
        self,
        cache_path: Path,
        meta: Dict,
        user_ids: np.ndarray,
        item_sequences: List[np.ndarray],
        action_sequences: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Save preprocessed compact sequences to cache."""
        try:
            save_kwargs: Dict[str, Any] = {
                "meta": np.array(json.dumps(meta)),
                "user_ids": user_ids,
                "item_sequences": np.array(item_sequences, dtype=object),
            }
            if action_sequences:
                save_kwargs["action_sequences"] = np.array(
                    action_sequences, dtype=object,
                )
            np.savez(cache_path, **save_kwargs)
            logger.info("Saved preprocessed sequences to cache: %s", cache_path)
        except Exception as e:
            logger.warning("Failed to save cache: %s", e)

    def _load_raw(self) -> Dict[str, Any]:
        """Return manifest + seq + feature table handles.

        Loads seq, user_feat, and item_feat datasets from HuggingFace.
        Feature tables are loaded lazily — only Arrow handles are returned,
        no data is deserialized until _prepare_splits processes them.
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
            self._repo_id, "seq", split="train", cache_dir=self.root_dir,
        )

        # 尝试加载特征表（失败不阻塞，优雅降级）
        ds_user = None
        ds_item = None
        try:
            ds_user = load_dataset(
                self._repo_id, "user_feat", split="train", cache_dir=self.root_dir,
            )
            logger.info("Loaded user_feat table (%d rows)", len(ds_user))
        except Exception:
            logger.warning("user_feat table unavailable, HyFormer 将仅使用序列特征")

        try:
            ds_item = load_dataset(
                self._repo_id, "item_feat", split="train", cache_dir=self.root_dir,
            )
            logger.info("Loaded item_feat table (%d rows)", len(ds_item))
        except Exception:
            logger.warning("item_feat table unavailable, HyFormer 将仅使用序列特征")

        return {
            "manifest": manifest,
            "seq": ds_seq,
            "user_feat": ds_user,
            "item_feat": ds_item,
        }

    def _build_feature_dicts(
        self, raw: Dict[str, Any], valid_user_ids: Set[int], valid_item_ids: Set[int],
    ) -> Tuple[Optional[Dict[int, np.ndarray]], Optional[Dict[int, np.ndarray]],
               Dict[str, Any], Dict[str, Any]]:
        """从 Arrow 表构建特征查找字典和元数据。

        仅在内存中保留出现在序列数据中的 user/item 特征，
        大幅减少特征表的内存占用。

        Returns:
            (user_feat_dict, item_feat_dict, user_spec_meta, item_spec_meta)
            如果特征表不可用，dict 为 None。
        """
        ds_user = raw.get("user_feat")
        ds_item = raw.get("item_feat")

        # --- 用户特征 ---
        user_feat_dict: Optional[Dict[int, np.ndarray]] = None
        user_spec_meta: Dict[str, Any] = {
            "specs": [], "dim": 0, "groups": [], "scalar_cols": [], "list_cols": [],
            "list_max_lens": {},
        }
        if ds_user is not None:
            try:
                user_table = ds_user.data
                user_id_arr = _extract_scalar_int_col(user_table, "user_id")

                # 扫描各列最大值（仅对有效用户）
                scalar_max: Dict[str, int] = {}
                list_max: Dict[str, int] = {}
                list_lens: Dict[str, int] = {}

                # 预先收集每列的原始 numpy 数组
                scalar_arrays: Dict[str, np.ndarray] = {}
                list_flat: Dict[str, np.ndarray] = {}
                list_offsets: Dict[str, np.ndarray] = {}
                for col in _USER_FEAT_SCALAR_COLS:
                    arr = _extract_scalar_int_col(user_table, col)
                    scalar_arrays[col] = arr
                    scalar_max[col] = int(arr.max())
                for col in _USER_FEAT_LIST_COLS:
                    flat, offsets, ml = _extract_list_int_col(user_table, col)
                    list_flat[col] = flat
                    list_offsets[col] = offsets
                    list_lens[col] = ml
                    if len(flat) > 0:
                        list_max[col] = int(flat.max())
                    else:
                        list_max[col] = 0

                # 构建 feature_specs
                specs, total_dim = _build_feature_specs_from_arrays(
                    scalar_max, list_max, list_lens,
                    _USER_FEAT_SCALAR_COLS, _USER_FEAT_LIST_COLS,
                )
                user_spec_meta = {
                    "specs": specs,
                    "dim": total_dim,
                    "groups": [[i] for i in range(len(specs))],
                    "scalar_cols": list(_USER_FEAT_SCALAR_COLS),
                    "list_cols": list(_USER_FEAT_LIST_COLS),
                    "list_max_lens": list_lens,
                }

                # 构建查找字典（仅有效用户）
                user_feat_dict = {}
                n_rows = len(user_id_arr)
                # 预分配零数组模板
                zero_template = np.zeros(total_dim, dtype=np.int64)
                for i in range(n_rows):
                    uid = int(user_id_arr[i])
                    if uid not in valid_user_ids:
                        continue
                    feat = zero_template.copy()
                    for spec_idx, col in enumerate(_USER_FEAT_SCALAR_COLS):
                        _, offset, _length = specs[spec_idx]
                        feat[offset] = scalar_arrays[col][i]
                    offset_scalar = len(_USER_FEAT_SCALAR_COLS)
                    for list_idx, col in enumerate(_USER_FEAT_LIST_COLS):
                        spec_idx = offset_scalar + list_idx
                        _, offset, length = specs[spec_idx]
                        start = int(list_offsets[col][i])
                        end = int(list_offsets[col][i + 1])
                        vals = list_flat[col][start:end]
                        n_vals = min(len(vals), length)
                        feat[offset:offset + n_vals] = vals[:n_vals]
                    user_feat_dict[uid] = feat

                logger.info(
                    "Built user feature dict: %d users × %d dims",
                    len(user_feat_dict), total_dim,
                )
            except Exception as e:
                logger.warning("Failed to build user feature dict: %s", e)
                user_feat_dict = None

        # --- 物品特征 ---
        item_feat_dict: Optional[Dict[int, np.ndarray]] = None
        item_spec_meta: Dict[str, Any] = {
            "specs": [], "dim": 0, "groups": [], "scalar_cols": [], "list_cols": [],
            "list_max_lens": {},
        }
        if ds_item is not None:
            try:
                item_table = ds_item.data
                item_id_arr = _extract_scalar_int_col(item_table, "item_id")

                # 所有物品特征列都是标量 int64
                scalar_max_i: Dict[str, int] = {}
                scalar_arrays_i: Dict[str, np.ndarray] = {}
                for col in _ITEM_FEAT_COLS:
                    arr = _extract_scalar_int_col(item_table, col)
                    scalar_arrays_i[col] = arr
                    scalar_max_i[col] = int(arr.max())

                specs_i, total_dim_i = _build_feature_specs_from_arrays(
                    scalar_max_i, {}, {},
                    _ITEM_FEAT_COLS, [],
                )
                item_spec_meta = {
                    "specs": specs_i,
                    "dim": total_dim_i,
                    "groups": [[i] for i in range(len(specs_i))],
                    "scalar_cols": list(_ITEM_FEAT_COLS),
                    "list_cols": [],
                    "list_max_lens": {},
                }

                # 构建查找字典（仅有效物品）
                item_feat_dict = {}
                n_rows_i = len(item_id_arr)
                zero_template_i = np.zeros(total_dim_i, dtype=np.int64)
                for i in range(n_rows_i):
                    iid = int(item_id_arr[i])
                    if iid not in valid_item_ids:
                        continue
                    feat = zero_template_i.copy()
                    for spec_idx, col in enumerate(_ITEM_FEAT_COLS):
                        _, offset, _length = specs_i[spec_idx]
                        feat[offset] = scalar_arrays_i[col][i]
                    item_feat_dict[iid] = feat

                logger.info(
                    "Built item feature dict: %d items × %d dims",
                    len(item_feat_dict), total_dim_i,
                )
            except Exception as e:
                logger.warning("Failed to build item feature dict: %s", e)
                item_feat_dict = None

        return user_feat_dict, item_feat_dict, user_spec_meta, item_spec_meta

    def _prepare_splits(
        self, raw: Dict[str, Any]
    ) -> Tuple[Dataset[Any], Dataset[Any], Dataset[Any]]:
        """Build user sequences + structured features.

        CRITICAL: All intermediate storage uses numpy int64 arrays (8 B/int),
        never Python lists (28 B/int).  This prevents OOM on 1M+ user datasets.

        Single-pass from Arrow → numpy arrays, no dict-of-lists intermediate.
        """
        candidate_pool = self.get_candidate_pool()

        cache_path = self._get_cache_path()
        cached = self._load_from_cache(cache_path)
        if cached:
            meta, user_ids_np, item_sequences, action_sequences = cached
            self._num_users = meta.get("num_users", 0)
            self._num_items = meta.get("num_items", 0)
            all_items = set(meta.get("all_items", []))
            item_sequences = [
                np.asarray(s, dtype=np.int64) for s in item_sequences
            ]
            if action_sequences:
                action_sequences = [
                    np.asarray(s, dtype=np.int64) for s in action_sequences
                ]
        else:
            # --- Arrow 零拷贝提取 ---
            ds_seq = raw["seq"]
            total = len(ds_seq)
            logger.info(
                "Processing %d user sequences (~%.1f GB raw, first run)...",
                total,
                total * 100 * 2 * 8 / (1024 ** 3),  # est: avg 100 items × 2 fields
            )

            arrow_table = ds_seq.data
            n_rows = len(arrow_table)
            user_id_col = arrow_table.column("user_id")
            seq_col = arrow_table.column("seq")
            seq_combined = (
                seq_col.chunks[0] if len(seq_col.chunks) == 1
                else pa.concat_arrays(seq_col.chunks)
            )
            offsets = seq_combined.offsets.to_numpy(zero_copy_only=False)
            flat_values = seq_combined.values
            flat_item_ids = flat_values.field("item_id").to_numpy(zero_copy_only=False)
            flat_action_types = flat_values.field("action_type").to_numpy(
                zero_copy_only=False,
            )
            user_ids_all = user_id_col.to_numpy(zero_copy_only=False)

            # --- 单次遍历：直接构建 numpy 数组列表 ---
            uid_list: List[int] = []
            seq_list: List[np.ndarray] = []
            act_list: List[np.ndarray] = []
            all_items_set: Set[int] = set()

            min_seq = self.min_seq_len
            min_action = self._min_action_type
            for i in range(n_rows):
                uid = int(user_ids_all[i])
                start = int(offsets[i])
                end = int(offsets[i + 1])
                if start == end:
                    continue
                iids = flat_item_ids[start:end]
                acts = flat_action_types[start:end]
                if min_action > 0:
                    keep = acts >= min_action
                    iids = iids[keep]
                    acts = acts[keep]
                    if len(iids) == 0:
                        continue
                if len(iids) < min_seq:
                    continue
                # 复制为自有 numpy 数组（解耦 Arrow 缓冲区）
                uid_list.append(uid)
                seq_list.append(iids.copy().astype(np.int64, copy=False))
                # action_type 为 int32，先安全转 float 处理可能的 nan，再转 int64
                act_list.append(
                    np.nan_to_num(acts.copy(), nan=0).astype(np.int64, copy=False),
                )
                all_items_set.update(iids.tolist())
                if i % 200_000 == 0 and i > 0:
                    _log_memory("Processed %d/%d users", i, n_rows)

            del arrow_table, seq_combined, flat_values
            del flat_item_ids, flat_action_types, offsets, user_ids_all
            gc.collect()

            self._num_users = len(uid_list)
            self._num_items = len(all_items_set)
            all_items = all_items_set  # ← 将局部变量提升到 if/else 共享作用域
            logger.info(
                "Built %d users × %d items.", self._num_users, self._num_items,
            )

            # Shuffle — 只打乱索引
            rng = np.random.default_rng(42)
            indices = np.arange(len(uid_list))
            rng.shuffle(indices)
            user_ids_np = np.array(uid_list, dtype=np.int64)[indices]
            item_sequences = [seq_list[i] for i in indices]
            action_sequences = [act_list[i] for i in indices]
            del uid_list, seq_list, act_list
            gc.collect()

            meta = {
                "num_users": self._num_users,
                "num_items": self._num_items,
                "all_items": list(all_items_set),
            }
            self._save_to_cache(
                cache_path, meta, user_ids_np, item_sequences, action_sequences,
            )
            _log_memory("Cache saved: %d users", self._num_users)

        # ---- 加载特征表 ----
        all_user_ids = set(int(uid) for uid in user_ids_np)
        user_feat_dict, item_feat_dict, user_spec_meta, item_spec_meta = (
            self._build_feature_dicts(raw, all_user_ids, all_items)
        )

        self._hyformer_schema_meta = {
            "user_int_feature_specs": user_spec_meta.get("specs", []),
            "item_int_feature_specs": item_spec_meta.get("specs", []),
            "user_dense_dim": 0,
            "item_dense_dim": 0,
            "seq_vocab_sizes": {
                _SEQ_DOMAIN_NAME: [self._num_items + 1, 3],
            },
            "user_ns_groups": user_spec_meta.get("groups", []),
            "item_ns_groups": item_spec_meta.get("groups", []),
            "seq_domains": [_SEQ_DOMAIN_NAME],
            "num_users": self._num_users,
            "num_items": self._num_items,
        }

        # ---- Split by user ----
        total_positions = sum(max(0, len(s) - 1) for s in item_sequences)
        n_train = int(total_positions * self.split_ratios[0])
        n_val = int(total_positions * self.split_ratios[1])
        logger.info(
            "Splits: train=%d, val=%d, test=%d (positions=%d)",
            n_train, n_val, total_positions - n_train - n_val, total_positions,
        )

        self._all_items = all_items
        item_pool = torch.as_tensor(sorted(all_items), dtype=torch.long)

        cum = 0
        train_uids: List[int] = []
        train_seqs: List[np.ndarray] = []
        train_acts: List[np.ndarray] = []
        val_uids: List[int] = []
        val_seqs: List[np.ndarray] = []
        val_acts: List[np.ndarray] = []
        test_uids: List[int] = []
        test_seqs: List[np.ndarray] = []
        test_acts: List[np.ndarray] = []

        for uid, seq, act in zip(
            user_ids_np, item_sequences, action_sequences, strict=True,
        ):
            n_pos = max(0, len(seq) - 1)
            if n_pos == 0:
                continue
            cum += n_pos
            target = (
                train_uids if cum <= n_train
                else val_uids if cum <= n_train + n_val
                else test_uids
            )
            target.append(int(uid))
            if target is train_uids:
                train_seqs.append(seq)
                train_acts.append(act)
            elif target is val_uids:
                val_seqs.append(seq)
                val_acts.append(act)
            else:
                test_seqs.append(seq)
                test_acts.append(act)

        train_ds = _StructuredSequenceSplit(
            user_ids=np.array(train_uids, dtype=np.int64),
            item_sequences=train_seqs,
            action_sequences=train_acts,
            item_pool=item_pool,
            max_seq_len=self.max_seq_len,
            neg_sample_count=self.neg_sample_count,
            candidate_pool=candidate_pool,
            user_feat_dict=user_feat_dict,
            item_feat_dict=item_feat_dict,
        )
        val_ds = _StructuredSequenceSplit(
            user_ids=np.array(val_uids, dtype=np.int64),
            item_sequences=val_seqs,
            action_sequences=val_acts,
            item_pool=item_pool,
            max_seq_len=self.max_seq_len,
            neg_sample_count=self.neg_sample_count,
            candidate_pool=candidate_pool,
            user_feat_dict=user_feat_dict,
            item_feat_dict=item_feat_dict,
        )
        test_ds = _StructuredSequenceSplit(
            user_ids=np.array(test_uids, dtype=np.int64),
            item_sequences=test_seqs,
            action_sequences=test_acts,
            item_pool=item_pool,
            max_seq_len=self.max_seq_len,
            neg_sample_count=self.neg_sample_count,
            candidate_pool=candidate_pool,
            user_feat_dict=user_feat_dict,
            item_feat_dict=item_feat_dict,
        )

        _log_memory(
            "Splits ready: train=%d val=%d test=%d",
            len(train_uids), len(val_uids), len(test_uids),
        )
        return train_ds, val_ds, test_ds

    # ----- schema metadata (HyFormer 集成点) -------------------------------

    def get_schema_metadata(self) -> Dict[str, Any]:
        """返回结构化特征规格，供 HyFormer 等模型初始化使用。

        必须在 .load() 之后调用，否则返回空 spec。

        Returns:
            包含 user_int_feature_specs, item_int_feature_specs,
            seq_vocab_sizes, user_ns_groups, item_ns_groups 等字段的字典。
        """
        if hasattr(self, "_hyformer_schema_meta"):
            return self._hyformer_schema_meta
        # .load() 未调用时返回最小合法 spec（无序列模式）
        logger.warning(
            "get_schema_metadata() called before .load(); "
            "返回最小 schema（无序列模式）",
        )
        return {
            "user_int_feature_specs": [(2, 0, 1)],
            "item_int_feature_specs": [(2, 0, 1)],
            "user_dense_dim": 0,
            "item_dense_dim": 0,
            "seq_vocab_sizes": {},
            "user_ns_groups": [[0]],
            "item_ns_groups": [[0]],
            "seq_domains": [],
            "num_users": self.num_users,
            "num_items": self.num_items,
        }

    # ----- iteration -----------------------------------------------------

    def get_item_pool_stats(self) -> Optional[Any]:
        """返回过滤后的物品池统计（用于负采样）。

        Returns
        -------
        Optional[ItemPoolStats]
            物品池统计对象，若数据集未加载则返回 None。
        """
        if not hasattr(self, "_all_items") or self._all_items is None:
            return None
        from recsys.data.negative_sampling import ItemPoolStats
        item_ids = np.array(sorted(self._all_items), dtype=np.int64)
        return ItemPoolStats(
            item_ids=item_ids,
            n_total_items=len(item_ids),
        )

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
