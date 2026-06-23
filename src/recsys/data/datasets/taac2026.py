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
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from recsys.core.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# ---- HuggingFace repo & available variants ----
_TAAC2026_REPO = "TAAC2026"

_VARIANTS: Dict[str, Dict[str, Any]] = {
    "data_sample": {
        "repo_suffix": "data_sample_1000",
        "label": "TAAC 2026 data sample 1000 — basic features",
        "has_domain_seq": True,
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
    """Convert a value to non-negative int, falling back to *default*."""
    try:
        if val is None:
            return default
        return max(int(val), 0)
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
        return [max(int(v), 0) for v in val]
    try:
        return [max(int(val), 0)]
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


def _classify_feat_cols(cols: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """按前缀细分特征列为 int / dense / string。"""
    int_cols = [c for c in cols if c.endswith("_int_") or "_int_" in c
                or c.startswith(("user_int_", "item_int_"))]
    dense_cols = [c for c in cols if "_dense_" in c]
    string_cols = [c for c in cols if "_string_" in c]
    return int_cols, dense_cols, string_cols


def _classify_domain_cols(
    domain_cols: List[str],
) -> Dict[str, List[str]]:
    """将 domain_* 列按域名（domain_a, domain_b, ...）分组。

    列名如 domain_a_seq_item_id, domain_a_seq_cate → domain_a: [col1, col2]
    """
    groups: Dict[str, List[str]] = {}
    for col in domain_cols:
        if not col.startswith("domain_"):
            continue
        # domain_X_seq_... → X
        parts = col.split("_")
        if len(parts) >= 2:
            domain_name = parts[1]  # "a", "b", "c", "d"
            groups.setdefault(domain_name, []).append(col)
    return groups


def _build_feature_specs(
    rows: List[Dict[str, Any]],
    int_cols: List[str],
    string_cols: List[str],
    max_vocab_cap: int = 500_000,
) -> Tuple[List[Tuple[int, int, int]], int]:
    """预计算 PCVRHyFormer 所需的 feature_specs。

    feature_specs = [(vocab_size, offset, length), ...]
    顺序：int_cols（长度=1），然后 string_cols（多值，需 padding）。

    vocab_size 为 max_value + 1（Embedding 索引需要），但上限为 max_vocab_cap
    防止高基数特征（如 hash ID）导致内存溢出。超出上限的特征由模型侧
    emb_skip_threshold 控制跳过 Embedding 创建。

    Returns:
        (feature_specs, total_int_dim)
    """
    # 扫描最大词表大小和多值特征最大长度
    max_vals: Dict[str, int] = {}
    max_lens: Dict[str, int] = {}
    for row in rows:
        for col in int_cols:
            v = _safe_int(row.get(col))
            max_vals[col] = max(max_vals.get(col, 0), v)
        for col in string_cols:
            vals = _safe_int_list(row.get(col))
            if vals:
                max_vals[col] = max(max_vals.get(col, 0), max(vals))  # pylint: disable=nested-min-max
            max_lens[col] = max(max_lens.get(col, 0), len(vals))

    specs: List[Tuple[int, int, int]] = []
    offset = 0
    for col in int_cols:
        raw_vocab = max_vals.get(col, 0) + 1
        vocab = min(raw_vocab, max_vocab_cap + 1)
        if raw_vocab > max_vocab_cap:
            logger.debug(
                "Feature '%s' vocab capped: %d → %d", col, raw_vocab, vocab,
            )
        specs.append((vocab, offset, 1))
        offset += 1
    for col in string_cols:
        raw_vocab = max_vals.get(col, 0) + 1
        vocab = min(raw_vocab, max_vocab_cap + 1)
        if raw_vocab > max_vocab_cap:
            logger.debug(
                "Feature '%s' vocab capped: %d → %d", col, raw_vocab, vocab,
            )
        length = max_lens.get(col, 1)
        specs.append((vocab, offset, length))
        offset += length
    return specs, offset


def _build_dense_dim(
    rows: List[Dict[str, Any]],
    dense_cols: List[str],
) -> Tuple[int, Dict[str, int]]:
    """预计算 dense 特征总维度和每列最大长度。

    Returns:
        (total_dense_dim, col_max_lens)
    """
    col_max: Dict[str, int] = {}
    for row in rows:
        for col in dense_cols:
            vals = _safe_float_list(row.get(col))
            col_max[col] = max(col_max.get(col, 0), len(vals))
    total = sum(col_max.values())
    return total, col_max


def _build_seq_vocab_sizes(
    rows: List[Dict[str, Any]],
    domain_groups: Dict[str, List[str]],
    max_vocab_cap: int = 500_000,
) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, int]], Dict[str, int]]:
    """预计算每个域的 seq_vocab_sizes。

    vocab_size 为 max_value + 1（Embedding 索引需要），上限 max_vocab_cap。

    Returns:
        (seq_vocab_sizes, domain_max_vals, domain_max_lens)
        seq_vocab_sizes: {domain: [vocab_size_per_sideinfo_col, ...]}
    """
    seq_vocab: Dict[str, List[int]] = {}
    domain_max_vals: Dict[str, Dict[str, int]] = {}
    domain_max_lens: Dict[str, int] = {}
    for domain, cols in domain_groups.items():
        mv: Dict[str, int] = {}
        ml = 0
        for row in rows:
            for col in cols:
                vals = _safe_int_list(row.get(col))
                if vals:
                    mv[col] = max(mv.get(col, 0), max(vals))  # pylint: disable=nested-min-max
                ml = max(ml, len(vals))
        seq_vocab[domain] = [min(mv.get(col, 0) + 1, max_vocab_cap + 1) for col in cols]
        domain_max_vals[domain] = mv
        domain_max_lens[domain] = ml
    return seq_vocab, domain_max_vals, domain_max_lens


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
    """Internal split that wraps a list of row dicts.

    Produces both legacy flat keys (backwards-compatible) and structured
    feature keys for PCVRHyFormer-style models.
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        user_cols: List[str],
        item_cols: List[str],
        domain_cols: List[str],
        # 可选：跨 split 全局预计算的结构化规格
        global_specs: Optional[Dict[str, Any]] = None,
        max_seq_len: int = 50,
    ) -> None:
        self._rows = rows
        self._user_cols = user_cols
        self._item_cols = item_cols
        self._domain_cols = domain_cols
        self._max_seq_len = max_seq_len

        # 预计算最大特征长度，保证所有行产出相同 shape 的 tensor。
        self._max_user_feat_len = _compute_max_feat_len(rows, user_cols)
        self._max_item_feat_len = _compute_max_feat_len(rows, item_cols)
        # 预计算 domain seq 的最大序列长度，封顶 max_seq_len 防止注意力矩阵 OOM
        self._max_domain_seq_len = 0
        for row in rows:
            for col in domain_cols:
                seq_len = len(_safe_int_list(row.get(col)))
                if seq_len > self._max_domain_seq_len:
                    self._max_domain_seq_len = seq_len
        self._max_domain_seq_len = min(self._max_domain_seq_len, max_seq_len)
        if self._max_domain_seq_len > 0:
            logger.debug(
                "TabularSplit max_domain_seq_len=%d (cap=%d)",
                self._max_domain_seq_len, max_seq_len,
            )

        # ---- 结构化特征规格（PCVRHyFormer 所需）----
        user_int_cols, user_dense_cols, user_string_cols = _classify_feat_cols(user_cols)
        item_int_cols, item_dense_cols, item_string_cols = _classify_feat_cols(item_cols)
        self._domain_groups = _classify_domain_cols(domain_cols)

        if global_specs is not None:
            # 使用跨 split 全局计算的结构化规格，避免词表不一致导致的 IndexError
            self._user_int_feature_specs = global_specs["user_int_feature_specs"]
            self._user_int_dim = global_specs["user_int_dim"]
            self._item_int_feature_specs = global_specs["item_int_feature_specs"]
            self._item_int_dim = global_specs["item_int_dim"]
            self._user_dense_dim = global_specs["user_dense_dim"]
            self._user_dense_max_lens = global_specs["user_dense_max_lens"]
            self._item_dense_dim = global_specs["item_dense_dim"]
            self._item_dense_max_lens = global_specs["item_dense_max_lens"]
            self._seq_vocab_sizes = global_specs["seq_vocab_sizes"]
            self._domain_max_lens = global_specs["domain_max_lens"]
        else:
            self._user_int_feature_specs, self._user_int_dim = _build_feature_specs(
                rows, user_int_cols, user_string_cols,
            )
            self._item_int_feature_specs, self._item_int_dim = _build_feature_specs(
                rows, item_int_cols, item_string_cols,
            )
            self._user_dense_dim, self._user_dense_max_lens = _build_dense_dim(
                rows, user_dense_cols,
            )
            self._item_dense_dim, self._item_dense_max_lens = _build_dense_dim(
                rows, item_dense_cols,
            )
            self._seq_vocab_sizes, _, self._domain_max_lens = _build_seq_vocab_sizes(
                rows, self._domain_groups,
            )

        # 封顶 domain max lens（统一处理 global_specs 和非 global_specs 路径）
        for domain in list(self._domain_max_lens.keys()):
            if self._domain_max_lens[domain] > max_seq_len:
                self._domain_max_lens[domain] = max_seq_len

        # ns_groups：每个特征独立成组
        self._user_ns_groups = [[i] for i in range(len(self._user_int_feature_specs))]
        self._item_ns_groups = [[i] for i in range(len(self._item_int_feature_specs))]

        # 列引用
        self._user_int_cols = user_int_cols
        self._user_string_cols = user_string_cols
        self._item_int_cols = item_int_cols
        self._item_string_cols = item_string_cols
        self._user_dense_cols = user_dense_cols
        self._item_dense_cols = item_dense_cols

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self._rows[idx]

        user_id = _safe_int(row.get("user_id"))
        item_id = _safe_int(row.get("item_id"))
        label = _safe_int(row.get("label_type"))

        # ---- 旧键：展平特征（向后兼容）----
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
            mlen = self._max_domain_seq_len
            seq = seq + [0] * (mlen - len(seq)) if len(seq) < mlen else seq[:mlen]
            domain_seqs.append(torch.as_tensor(seq, dtype=torch.long))

        result: Dict[str, Any] = {
            "user_id": torch.as_tensor(user_id, dtype=torch.long),
            "item_id": torch.as_tensor(item_id, dtype=torch.long),
            "label": torch.as_tensor(label, dtype=torch.long),
            "user_feats": torch.as_tensor(user_feats, dtype=torch.float32),
            "item_feats": torch.as_tensor(item_feats, dtype=torch.float32),
            "domain_seqs": (
                pad_sequence(domain_seqs, batch_first=True, padding_value=0)
                if domain_seqs
                else torch.empty(0, dtype=torch.long)
            ),
        }

        # ---- 新键：结构化特征（PCVRHyFormer 消费）----
        # user_int_feats — 使用 spec_idx 定位列（offset 是张量布局偏移，不是列索引）
        user_int_vals = torch.zeros(self._user_int_dim, dtype=torch.long)
        for spec_idx, spec in enumerate(self._user_int_feature_specs):
            _vocab, offset, length = spec
            if length == 1:
                col = self._user_int_cols[spec_idx]
                user_int_vals[offset] = _safe_int(row.get(col))
            else:
                string_idx = spec_idx - len(self._user_int_cols)
                if 0 <= string_idx < len(self._user_string_cols):
                    col = self._user_string_cols[string_idx]
                    vals = _safe_int_list(row.get(col))
                    padd = (
                        vals + [0] * (length - len(vals))
                        if len(vals) < length
                        else vals[:length]
                    )
                    user_int_vals[offset:offset + length] = torch.as_tensor(
                        padd, dtype=torch.long,
                    )
        result["user_int_feats"] = user_int_vals

        # item_int_feats — 同理使用 spec_idx
        item_int_vals = torch.zeros(self._item_int_dim, dtype=torch.long)
        for spec_idx, spec in enumerate(self._item_int_feature_specs):
            _vocab, offset, length = spec
            if length == 1:
                col = self._item_int_cols[spec_idx]
                item_int_vals[offset] = _safe_int(row.get(col))
            else:
                string_idx = spec_idx - len(self._item_int_cols)
                if 0 <= string_idx < len(self._item_string_cols):
                    col = self._item_string_cols[string_idx]
                    vals = _safe_int_list(row.get(col))
                    padd = (
                        vals + [0] * (length - len(vals))
                        if len(vals) < length
                        else vals[:length]
                    )
                    item_int_vals[offset:offset + length] = torch.as_tensor(
                        padd, dtype=torch.long,
                    )
        result["item_int_feats"] = item_int_vals

        # user_dense_feats
        if self._user_dense_dim > 0:
            ud_vals = torch.zeros(self._user_dense_dim, dtype=torch.float32)
            doff = 0
            for col in self._user_dense_cols:
                clen = self._user_dense_max_lens.get(col, 0)
                vals = _safe_float_list(row.get(col))
                padd = vals + [0.0] * (clen - len(vals)) if len(vals) < clen else vals[:clen]
                ud_vals[doff:doff + clen] = torch.as_tensor(padd, dtype=torch.float32)
                doff += clen
            result["user_dense_feats"] = ud_vals

        # item_dense_feats
        if self._item_dense_dim > 0:
            id_vals = torch.zeros(self._item_dense_dim, dtype=torch.float32)
            doff = 0
            for col in self._item_dense_cols:
                clen = self._item_dense_max_lens.get(col, 0)
                vals = _safe_float_list(row.get(col))
                padd = vals + [0.0] * (clen - len(vals)) if len(vals) < clen else vals[:clen]
                id_vals[doff:doff + clen] = torch.as_tensor(padd, dtype=torch.float32)
                doff += clen
            result["item_dense_feats"] = id_vals

        # seq_data / seq_lens / seq_time_buckets（字典）
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_tb: Dict[str, torch.Tensor] = {}
        for domain, cols in sorted(self._domain_groups.items()):
            max_l = self._domain_max_lens.get(domain, 0)
            if max_l == 0:
                continue
            seqs = []
            for col in cols:
                vals = _safe_int_list(row.get(col))
                padd = vals + [0] * (max_l - len(vals)) if len(vals) < max_l else vals[:max_l]
                seqs.append(torch.as_tensor(padd, dtype=torch.long))
            # stack: (num_sideinfo, max_len)
            seq_data[domain] = torch.stack(seqs, dim=0)
            # 实际有效长度：取第一个 sideinfo 列的有效长度
            first_vals = _safe_int_list(row.get(cols[0]))
            seq_lens[domain] = torch.as_tensor(
                min(len(first_vals), max_l), dtype=torch.long,
            )
            seq_tb[domain] = torch.zeros(max_l, dtype=torch.long)
        if seq_data:
            result["seq_data"] = seq_data
            result["seq_lens"] = seq_lens
            result["seq_time_buckets"] = seq_tb

        return result

    def get_schema_metadata(self) -> Dict[str, Any]:
        """返回结构化特征规格（通用扩展点，供任意需要结构化特征的模型消费）。

        返回的字典包含：
        - user_int_feature_specs / item_int_feature_specs: [(vocab, offset, length), ...]
        - user_dense_dim / item_dense_dim: dense 特征总维度
        - seq_vocab_sizes: {domain: [vocab_per_sideinfo, ...]}
        - user_ns_groups / item_ns_groups: 特征分组
        - seq_domains: 域名列表
        """
        return {
            "user_int_feature_specs": self._user_int_feature_specs,
            "item_int_feature_specs": self._item_int_feature_specs,
            "user_dense_dim": self._user_dense_dim,
            "item_dense_dim": self._item_dense_dim,
            "seq_vocab_sizes": self._seq_vocab_sizes,
            "user_ns_groups": self._user_ns_groups,
            "item_ns_groups": self._item_ns_groups,
            "seq_domains": sorted(self._domain_groups.keys()),
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
        self.max_seq_len = max_seq_len
        # 在 _prepare_splits 中填充的属性，在此提前声明避免 pylint W0201
        self._feature_cols: List[str] = []
        self._num_users: int = 0
        self._num_items: int = 0
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

        # 跨 split 全局计算结构化特征规格（仿照 dist/train.py 的 FeatureSchema 思路）
        # 避免各 split 独立算词表导致 val/test 样本值越界 IndexError
        user_int_cols, user_dense_cols, user_string_cols = _classify_feat_cols(user_cols)
        item_int_cols, item_dense_cols, item_string_cols = _classify_feat_cols(item_cols)
        domain_groups = _classify_domain_cols(domain_cols)
        global_u_specs, global_u_dim = _build_feature_specs(
            rows, user_int_cols, user_string_cols,
        )
        global_i_specs, global_i_dim = _build_feature_specs(
            rows, item_int_cols, item_string_cols,
        )
        global_ud_dim, global_ud_max_lens = _build_dense_dim(rows, user_dense_cols)
        global_id_dim, global_id_max_lens = _build_dense_dim(rows, item_dense_cols)
        global_seq_vocab, _, global_domain_ml = _build_seq_vocab_sizes(rows, domain_groups)
        # 封顶 domain max lens，防止超长序列导致注意力矩阵 OOM
        for domain in global_domain_ml:
            if global_domain_ml[domain] > self.max_seq_len:
                logger.debug(
                    "Domain '%s' max_len capped: %d → %d",
                    domain, global_domain_ml[domain], self.max_seq_len,
                )
                global_domain_ml[domain] = self.max_seq_len
        global_specs = {
            "user_int_feature_specs": global_u_specs,
            "user_int_dim": global_u_dim,
            "item_int_feature_specs": global_i_specs,
            "item_int_dim": global_i_dim,
            "user_dense_dim": global_ud_dim,
            "user_dense_max_lens": global_ud_max_lens,
            "item_dense_dim": global_id_dim,
            "item_dense_max_lens": global_id_max_lens,
            "seq_vocab_sizes": global_seq_vocab,
            "domain_max_lens": global_domain_ml,
        }

        return (
            _TabularSplit(train_rows, user_cols, item_cols, domain_cols,
                          global_specs=global_specs, max_seq_len=self.max_seq_len),
            _TabularSplit(val_rows, user_cols, item_cols, domain_cols,
                          global_specs=global_specs, max_seq_len=self.max_seq_len),
            _TabularSplit(test_rows, user_cols, item_cols, domain_cols,
                          global_specs=global_specs, max_seq_len=self.max_seq_len),
        )

    # ----- iteration -----------------------------------------------------

    def get_schema_metadata(self) -> Dict[str, Any]:
        """返回模型所需的结构化特征规格（通用扩展点）。

        从 train split 提取 schema（所有 split 共享相同列结构）。
        需在 .load() 之后调用。任何需要结构化特征的模型均可消费此元信息。
        """
        if self._train is None:
            raise RuntimeError("Dataset not loaded. Call .load() first.")
        if hasattr(self._train, "get_schema_metadata"):
            return self._train.get_schema_metadata()  # type: ignore[union-attr]
        return {}

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
    """TAAC 2026 data sample 1000 — lightweight prototyping variant."""

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
    """TAAC 2026 second round sample 1000 — with domain sequence features."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        super().__init__(variant="second_round", **kwargs)
