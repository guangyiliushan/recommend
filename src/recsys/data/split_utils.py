"""序列数据集的通用 Split 实现。

从 TAAC2025 的 ``_SequenceSplit`` 提取为公共模块，供所有用户序列格式
数据集复用（TAAC2025、MovieLens-1M 等）。

``SequenceSplit`` 是惰性序列 split — 按需计算带标签的样本：

- 存储紧凑的用户→物品映射（O(num_users) 内存）
- 而不是预展开为扁平列表（O(total_interactions) 内存）
- ``__getitem__`` 使用二分查找定位用户，``O(log n)`` 索引
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceSplit(Dataset[Dict[str, torch.Tensor]]):
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
            candidate_items = torch.as_tensor(
                np.array([], dtype=np.int64), dtype=torch.long
            )

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
