"""TAAC 2026 dataset adapter — Tencent Advertising Algorithm Competition 2026.

Lightweight sample datasets for local prototyping.
Two variants available:
    - TAAC2026/data_sample_1000          (1k rows, CTR/CVR prediction)
    - TAAC2026/second_round_sample_1000  (1k rows, richer sequence features)

Column groups:
    core       – user_id, item_id, label_type, label_time, timestamp
    user_feat  – user_int_feats_*, user_dense_feats_*, user_string_feats_*
    item_feat  – item_int_feats_*,  item_dense_feats_*,  item_string_feats_*
    domain seq – domain_{a,b,c,d}_seq_*   (only present in second_round_sample_1000)

Usage:
    ds = TAAC2026Dataset(
        variant="second_round",
        root_dir="./data",
        split_ratios=(0.8, 0.1, 0.1),
    ).load()
    train_loader = ds.get_dataloader("train", batch_size=256)

    # Or use the thin registered wrappers:
    from recsys.core import DATASET_REGISTRY
    ds_cls = DATASET_REGISTRY.get("taac2026_second_round")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from recsys.core.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# ---- HuggingFace repo & available variants ----
_TAAC2026_REPO = "TAAC2026"

_VARIANTS: Dict[str, Dict[str, str]] = {
    "data_sample": {
        "repo_suffix": "data_sample_1000",
        "label": "TAAC 2026 data sample 1000 — basic features",
        "has_domain_seq": False,
    },
    "second_round": {
        "repo_suffix": "second_round_sample_1000",
        "label": "TAAC 2026 second round sample 1000 — with domain sequences",
        "has_domain_seq": True,
    },
}

# ---- Pre-computed column groups for fast numeric extraction ----
_CORE_COLS = ["user_id", "item_id", "label_type", "label_time", "timestamp"]


def _detect_column_groups(columns: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Split column names into user / item / domain categories."""
    user_cols: List[str] = []
    item_cols: List[str] = []
    domain_cols: List[str] = []
    for col in columns:
        if col in _CORE_COLS:
            continue
        if col.startswith("user_"):
            user_cols.append(col)
        elif col.startswith("item_"):
            item_cols.append(col)
        elif col.startswith("domain_"):
            domain_cols.append(col)
    return user_cols, item_cols, domain_cols


def _safe_int(val: Any, default: int = 0) -> int:
    """Convert a value to int, falling back to *default*."""
    try:
        if val is None:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float_list(val: Any) -> List[float]:
    """Convert to list of floats if possible, else empty list."""
    if val is None:
        return []
    if isinstance(val, list):
        return [float(v) for v in val]
    try:
        return [float(val)]
    except (ValueError, TypeError):
        return []


def _safe_int_list(val: Any) -> List[int]:
    if val is None:
        return []
    if isinstance(val, list):
        return [int(v) for v in val]
    try:
        return [int(val)]
    except (ValueError, TypeError):
        return []


def _compute_max_feat_len(
    rows: List[Dict[str, Any]], cols: List[str]
) -> int:
    """预计算 user_feats / item_feats 在给定列集合下的最大展平长度。

    遍历所有行，对每列根据其前缀判断展平后的元素数：
    - user_int_ / item_int_: 标量 → 1 个元素
    - user_dense_ / item_dense_: list<float> → len(list) 个元素
    - user_string_ / item_string_: list<int> → len(list) 个元素

    Returns
    -------
    int
        所有行中最大的展平特征长度。
    """
    max_len = 0
    for row in rows:
        length = 0
        for col in cols:
            val = row.get(col)
            if col.startswith(("user_int_", "item_int_")):
                length += 1
            elif col.startswith(("user_dense_", "item_dense_")):
                length += len(_safe_float_list(val))
            elif col.startswith(("user_string_", "item_string_")):
                length += len(_safe_int_list(val))
        max_len = max(max_len, length)
    return max_len


def _pad_or_truncate(feats: List[float], target_len: int) -> List[float]:
    """将特征列表补零或截断到固定长度，保证 collate 一致性。

    对齐 dist/dataset.py 的 _pad_varlen_float_column 语义：
    不足则右侧补 0.0，超出则截断到 target_len。
    """
    if len(feats) < target_len:
        return feats + [0.0] * (target_len - len(feats))
    return feats[:target_len]


def _build_dense_id_map(values: List[int]) -> Dict[int, int]:
    """将一组原始（可能稀疏）ID 映射为从 1 开始的稠密连续 ID。

    保留 0 给 padding/OOV，不做映射。
    返回 ``{原始ID: 连续ID}`` 的字典。

    Parameters
    ----------
    values : List[int]
        原始 ID 集合（可重复，自动去重排序保证确定性）。

    Returns
    -------
    Dict[int, int]
        原始 ID 到稠密 ID 的映射，从 1 开始编号。
    """
    unique = sorted(set(values) - {0})
    return {raw_id: dense_id for dense_id, raw_id in enumerate(unique, start=1)}


def _remap_rows_inplace(
    rows: List[Dict[str, Any]],
    user_map: Dict[int, int],
    item_map: Dict[int, int],
) -> Tuple[int, int]:
    """原地将 rows 中的 user_id / item_id 替换为稠密 ID。

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        待 remap 的行列表（原地修改）。
    user_map : Dict[int, int]
        用户 ID 映射表。
    item_map : Dict[int, int]
        物品 ID 映射表。

    Returns
    -------
    Tuple[int, int]
        (remap 后唯一用户数, remap 后唯一物品数)
    """
    user_set: set = set()
    item_set: set = set()
    for row in rows:
        raw_uid = _safe_int(row.get("user_id"))
        raw_iid = _safe_int(row.get("item_id"))
        new_uid = user_map.get(raw_uid, 0)
        new_iid = item_map.get(raw_iid, 0)
        row["user_id"] = new_uid
        row["item_id"] = new_iid
        user_set.add(new_uid)
        item_set.add(new_iid)
    return len(user_set), len(item_set)


class _TabularSplit(Dataset[Dict[str, torch.Tensor]]):
    """Internal split that wraps a list of row dicts."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        user_cols: List[str],
        item_cols: List[str],
        domain_cols: List[str],
    ) -> None:
        self._rows = rows
        self._user_cols = user_cols
        self._item_cols = item_cols
        self._domain_cols = domain_cols
        # 预计算最大特征长度，保证所有行产出相同 shape 的 tensor。
        # 对齐 dist/dataset.py 的 schema.json dim 驱动的固定尺寸输出。
        self._max_user_feat_len = _compute_max_feat_len(rows, user_cols)
        self._max_item_feat_len = _compute_max_feat_len(rows, item_cols)
        # 预计算 domain seq 的最大序列长度，保证所有行产出相同 shape。
        self._max_domain_seq_len = 0
        for row in rows:
            for col in domain_cols:
                seq_len = len(_safe_int_list(row.get(col)))
                if seq_len > self._max_domain_seq_len:
                    self._max_domain_seq_len = seq_len

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self._rows[idx]

        user_id = _safe_int(row.get("user_id"))
        item_id = _safe_int(row.get("item_id"))
        label = _safe_int(row.get("label_type"))

        # Build user feature tensor from int / dense / string col groups
        user_feats: List[float] = []
        for col in self._user_cols:
            val = row.get(col)
            if col.startswith("user_int_"):
                user_feats.append(float(_safe_int(val)))
            elif col.startswith("user_dense_"):
                user_feats.extend(_safe_float_list(val))
            elif col.startswith("user_string_"):
                user_feats.extend(float(v) for v in _safe_int_list(val))
        user_feats = _pad_or_truncate(user_feats, self._max_user_feat_len)

        item_feats: List[float] = []
        for col in self._item_cols:
            val = row.get(col)
            if col.startswith("item_int_"):
                item_feats.append(float(_safe_int(val)))
            elif col.startswith("item_dense_"):
                item_feats.extend(_safe_float_list(val))
            elif col.startswith("item_string_"):
                item_feats.extend(float(v) for v in _safe_int_list(val))
        item_feats = _pad_or_truncate(item_feats, self._max_item_feat_len)

        domain_seqs: List[torch.Tensor] = []
        for col in self._domain_cols:
            val = row.get(col)
            seq = _safe_int_list(val)
            # Pad 到全局最大序列长度，保证 collate 时 shape 一致
            mlen = self._max_domain_seq_len
            seq = seq + [0] * (mlen - len(seq)) if len(seq) < mlen else seq[:mlen]
            domain_seqs.append(torch.as_tensor(seq, dtype=torch.long))

        return {
            "user_id": torch.as_tensor(user_id, dtype=torch.long),
            "item_id": torch.as_tensor(item_id, dtype=torch.long),
            "label": torch.as_tensor(label, dtype=torch.long),
            "user_feats": torch.as_tensor(user_feats, dtype=torch.float32),
            "item_feats": torch.as_tensor(item_feats, dtype=torch.float32),
            "domain_seqs": (
                # torch.stack(domain_seqs, dim=0)
                pad_sequence(domain_seqs, batch_first=True, padding_value=0)
                if domain_seqs
                else torch.empty(0, dtype=torch.long)
            ),
        }

    def iter_user_item_pairs_fast(self):
        """快速迭代 (user_id, item_id) 对 — O(rows)，零 tensor 分配。

        直接从原始字典提取，跳过 __getitem__ 的 tensor 构造开销。
        供实验管线 `_execute_nontrainable_path` 调用。
        """
        for row in self._rows:
            yield _safe_int(row.get("user_id")), _safe_int(row.get("item_id"))

    def extract_user_item_mapping_fast(self) -> dict:
        """快速提取 user_id → set(item_ids) 映射 — O(rows)。

        供实验管线 `_execute_nontrainable_path` 调用。
        与 TAAC2025 的 `_SequenceSplit.extract_user_item_mapping_fast` 对齐。

        Returns
        -------
        dict[int, set[int]]
            用户到物品集合的映射。
        """
        mapping: dict = defaultdict(set)
        for row in self._rows:
            mapping[_safe_int(row.get("user_id"))].add(
                _safe_int(row.get("item_id"))
            )
        return dict(mapping)


class TAAC2026Dataset(BaseDataset):
    """TAAC 2026 advertising recommendation sample dataset.

    Parameters
    ----------
    variant : str
        ``"data_sample"`` or ``"second_round"``.
    root_dir : str
        Local cache directory for HuggingFace datasets.
    split_mode : str
        数据切分方式：``"temporal"``（默认，按 timestamp 时序切分，
        用过去预测未来 — 推荐系统标准做法）或 ``"random"``
        （随机打乱后切分，用于非时序场景如 CTR/CVR 点式评估）。
    split_ratios : tuple
        (train, val, test) 比例，默认 (0.8, 0.1, 0.1)。
    """

    def __init__(
        self,
        variant: str = "second_round",
        root_dir: str = "./data",
        split_mode: str = "temporal",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        max_seq_len: int = 50,
        min_seq_len: int = 2,
        neg_sample_count: int = 4,
        **kwargs: Any,
    ) -> None:
        if variant not in _VARIANTS:
            raise ValueError(
                f"variant must be one of {list(_VARIANTS.keys())}, got '{variant}'"
            )
        if split_mode not in ("temporal", "random"):
            raise ValueError(
                f"split_mode must be 'temporal' or 'random', got '{split_mode}'"
            )
        self._variant = variant
        self._variant_info = _VARIANTS[variant]
        self._repo_id = f"{_TAAC2026_REPO}/{self._variant_info['repo_suffix']}"
        self._split_mode = split_mode
        # Dense remap 调试元信息（_prepare_splits 中填充）
        self._user_id_space: str = "raw"
        self._item_id_space: str = "raw"
        self._padding_idx: int = 0
        self._raw_user_id_max: int = 0
        self._raw_item_id_max: int = 0
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
        return f"taac2026_{self._variant}"

    @property
    def dataset_url(self) -> str:
        return f"https://huggingface.co/datasets/{self._repo_id}"

    @property
    def feature_cols(self) -> List[str]:
        return self._feature_cols

    @property
    def label_col(self) -> str:
        return "label_type"

    @property
    def num_users(self) -> int:
        return self._num_users

    @property
    def num_items(self) -> int:
        return self._num_items

    # ----- loading -------------------------------------------------------

    def _load_raw(self) -> Dict[str, Any]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "huggingface datasets is required. "
                "Install with: pip install datasets"
            ) from None

        logger.info("Loading %s from HuggingFace …", self._repo_id)
        ds = load_dataset(
            self._repo_id, "default", split="train", cache_dir=self.root_dir
        )
        return {"dataset": ds}

    def _prepare_splits(
        self, raw: Dict[str, Any]
    ) -> Tuple[Dataset[Any], Dataset[Any], Dataset[Any]]:
        ds = raw["dataset"]

        # Detect column groups from the actual dataset schema
        columns = list(ds.features.keys()) if hasattr(ds, "features") else ds.column_names
        self._feature_cols = [c for c in columns if c not in _CORE_COLS]
        user_cols, item_cols, domain_cols = _detect_column_groups(columns)

        logger.info(
            "TAAC 2026 %s: %d rows, %d user feat cols, %d item feat cols, %d domain seq cols.",
            self._variant,
            len(ds),
            len(user_cols),
            len(item_cols),
            len(domain_cols),
        )

        # Convert HF dataset rows to plain dicts & collect raw IDs for dense remap
        rows: List[Dict[str, Any]] = []
        raw_user_ids: List[int] = []
        raw_item_ids: List[int] = []
        for row in ds:
            d = dict(row)
            rows.append(d)
            raw_user_ids.append(_safe_int(row.get("user_id")))
            raw_item_ids.append(_safe_int(row.get("item_id")))

        # 保存原始 ID 统计信息（调试用）
        self._raw_user_id_max = max(raw_user_ids) if raw_user_ids else 0
        self._raw_item_id_max = max(raw_item_ids) if raw_item_ids else 0

        # ---- Dense remap: 原始 ID → 从 1 开始的连续 ID ----
        user_map = _build_dense_id_map(raw_user_ids)
        item_map = _build_dense_id_map(raw_item_ids)
        self._num_users, self._num_items = _remap_rows_inplace(
            rows, user_map, item_map,
        )
        self._user_id_space = "dense_1_based"
        self._item_id_space = "dense_1_based"

        logger.info(
            "Dense remap: users %d → %d, items %d → %d (padding_idx=%d)",
            len(raw_user_ids), self._num_users,
            len(raw_item_ids), self._num_items,
            self._padding_idx,
        )

        # Split rows by chosen strategy
        if self._split_mode == "temporal":
            # 按时序切分：timestamp 升序排列，train → val → test（推荐系统标准做法）
            rows.sort(key=lambda r: _safe_int(r.get("timestamp")))
        else:
            # 随机切分：shuffle 后按比例分割，适用于 CTR/CVR 点式评估
            import numpy as np
            rng = np.random.default_rng(42)
            rng.shuffle(rows)  # type: ignore[arg-type]

        n = len(rows)
        n_train = int(n * self.split_ratios[0])
        n_val = int(n * self.split_ratios[1])

        train_rows = rows[:n_train]
        val_rows = rows[n_train : n_train + n_val]
        test_rows = rows[n_train + n_val :]

        logger.info(
            "Splits: train=%d, val=%d, test=%d | users=%d, items=%d",
            len(train_rows), len(val_rows), len(test_rows),
            self._num_users, self._num_items,
        )

        return (
            _TabularSplit(train_rows, user_cols, item_cols, domain_cols),
            _TabularSplit(val_rows, user_cols, item_cols, domain_cols),
            _TabularSplit(test_rows, user_cols, item_cols, domain_cols),
        )

    # ----- iteration -----------------------------------------------------

    def __len__(self) -> int:
        if self._train is None:
            raise RuntimeError("Dataset not loaded. Call .load() first.")
        return sum(
            len(split)  # type: ignore[arg-type]
            for split in (self._train, self._val, self._test)
            if split is not None
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "TAAC2026Dataset uses per-split TabularSplit datasets."
            " Use .get_dataloader(split='train') or .get_split(split)."
        )


# ------------------------------------------------------------------
# Auto-register both variants via thin wrappers
# ------------------------------------------------------------------
from recsys.core.registry import DATASET_REGISTRY  # noqa: E402


@DATASET_REGISTRY.register(
    "taac2026_data_sample",
    family="deep_ctr",
    modality=("tabular", "time-series"),
    tasks=("ctr", "cvr"),
)
class TAAC2026DataSample(TAAC2026Dataset):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        super().__init__(variant="data_sample", **kwargs)


@DATASET_REGISTRY.register(
    "taac2026_second_round",
    family="deep_ctr",
    modality=("tabular", "time-series"),
    tasks=("ctr", "cvr"),
)
class TAAC2026SecondRound(TAAC2026Dataset):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        super().__init__(variant="second_round", **kwargs)
