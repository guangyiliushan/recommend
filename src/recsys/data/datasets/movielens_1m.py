"""MovieLens-1M 数据集适配器 (RecZoo)。

从 RecZoo 的 Movielens1M_m1 加载，数据格式为 JSON 二维数组：
``[[user0_items], [user1_items], ...]``。

预切分为 train/val/test 三个独立 JSON 文件，直接 json.load 加载
（不走 HF datasets API，因嵌套 JSON 格式导致 Viewer 解析失败）。

序列切分：复用 ``SequenceSplit`` 公共模块。

Usage:
    ds = Movielens1MDataset(root_dir="./data").load()
    train_loader = ds.get_dataloader("train", batch_size=64)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from recsys.core.base_dataset import BaseDataset
from recsys.core.registry import DATASET_REGISTRY
from recsys.data.split_utils import SequenceSplit

logger = logging.getLogger(__name__)

_REPO_URL = "https://huggingface.co/datasets/reczoo/Movielens1M_m1/resolve/main"
_FILES = {
    "train": "train_data.json",
    "val": "validation_data.json",
    "test": "test_data.json",
}


class Movielens1MDataset(BaseDataset):
    """MovieLens-1M (RecZoo) — 预切分的序列推荐数据集。

    每个用户对应一个交互物品序列。已有 train/val/test JSON 文件，
    不接受运行时 split_ratios 参数。

    Parameters
    ----------
    root_dir : str
        数据集缓存目录。
    max_seq_len : int
        序列最大长度（默认 50）。
    min_seq_len : int
        用户序列最小长度（短序列用户被过滤）。
    """

    dataset_name = "movielens_1m"
    dataset_url = "https://huggingface.co/datasets/reczoo/Movielens1M_m1"
    feature_cols = ["user_id", "item_id"]
    label_col = "label"

    def __init__(
        self,
        root_dir: str = "./data",
        split_ratios: tuple = (0.8, 0.1, 0.1),  # 忽略，预切分数据集
        max_seq_len: int = 50,
        min_seq_len: int = 2,
        neg_sample_count: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            split_ratios=split_ratios,  # 接受但忽略
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            neg_sample_count=neg_sample_count,
            **kwargs,
        )
        self._max_seq_len = max_seq_len
        self._min_seq_len = min_seq_len

    def _load_raw(self) -> dict:
        """加载三个 JSON 文件（优先本地缓存，其次 HF 下载）。

        Returns
        -------
        dict[str, list[list[int]]]
            {"train": [...], "val": [...], "test": [...]}
        """
        import urllib.request

        cache_dir = Path(self.root_dir) / "movielens_1m"
        cache_dir.mkdir(parents=True, exist_ok=True)

        raw: dict[str, Any] = {}
        for split, filename in _FILES.items():
            local_path = cache_dir / filename
            url = f"{_REPO_URL}/{filename}"

            # 优先本地缓存
            if local_path.exists():
                logger.info("Loading %s from cache: %s", split, local_path)
                raw[split] = json.loads(local_path.read_text(encoding="utf-8"))
                continue

            # 从 HF 下载
            logger.info("Downloading %s from HF ...", url)
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                local_path.write_text(
                    json.dumps(data), encoding="utf-8"
                )
                raw[split] = data
            except Exception as exc:
                raise RuntimeError(
                    f"无法加载 {split} split ({url}): {exc}"
                ) from exc

        return raw

    def _prepare_splits(self, raw: dict) -> tuple:
        """将 JSON 数据转换为 SequenceSplit。

        过滤序列长度不足 min_seq_len 的用户，
        从 train 序列构建全局物品池作为候选。
        """
        import numpy as np

        # 过滤短序列用户
        filtered: dict[str, list[np.ndarray]] = {}
        all_items: set[int] = set()
        user_counts: dict[str, int] = {}

        for split in ("train", "val", "test"):
            seqs = raw[split]
            filtered_seqs: list[np.ndarray] = []
            for seq in seqs:
                if len(seq) < self._min_seq_len:
                    continue
                arr = np.asarray(seq, dtype=np.int64)
                filtered_seqs.append(arr)
                if split == "train":
                    all_items.update(int(x) for x in seq)
            filtered[split] = filtered_seqs
            user_counts[split] = len(filtered_seqs)

        self._num_users = user_counts["train"]
        self._num_items = len(all_items)
        assert self._num_items > 0, "训练集中无有效物品"

        # 构建候选池（所有 train 物品，按 ID 排序）
        candidate_pool = torch.as_tensor(sorted(all_items), dtype=torch.long)

        logger.info(
            "MovieLens-1M: train=%d users, val=%d users, test=%d users, %d items",
            user_counts["train"], user_counts["val"],
            user_counts["test"], self._num_items,
        )

        splits = []
        for split_name in ("train", "val", "test"):
            seqs = filtered[split_name]
            n_users = len(seqs)
            user_ids = np.arange(n_users, dtype=np.int64)
            splits.append(
                SequenceSplit(
                    user_ids=user_ids,
                    item_sequences=seqs,
                    max_seq_len=self._max_seq_len,
                    neg_sample_count=self.neg_sample_count,
                    candidate_pool=candidate_pool,
                )
            )

        return tuple(splits)

    # ----- metadata ------------------------------------------------------

    @property
    def num_users(self) -> int:
        return self._num_users

    @property
    def num_items(self) -> int:
        return self._num_items

    # ----- iteration -----------------------------------------------------

    def __len__(self) -> int:
        if self._train is None:
            raise RuntimeError("Dataset not loaded. Call .load() first.")
        return sum(
            len(s)
            for s in (self._train, self._val, self._test)
            if s is not None
        )

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError(
            "Movielens1MDataset uses per-split SequenceSplit datasets."
            " Use .get_dataloader(split='train') or .get_split(split)."
        )

    # ---- EDA 兼容接口 ----

    def load_subset(
        self, subset: str, max_rows: int = 500_000, seed: int = 42
    ) -> tuple:
        """加载单个 split 为 DataFrame（EDA 兼容接口）。

        将 MovieLens 的预切分 split 展平为 ``user_id × item_id`` 长格式
        DataFrame，供 ``recsys-dataset-eda`` 统计模块消费。

        Parameters
        ----------
        subset : str
            Split 名称。支持 ``"train"`` / ``"val"`` / ``"test"``，
            也兼容 TAAC2025 子集名 ``"seq"`` / ``"behavior"``（均映射到 train）。
        max_rows : int
            最大行数（超出则预采样）。
        seed : int
            随机种子。

        Returns
        -------
        tuple[pd.DataFrame, Optional[dict]]
            (DataFrame with columns [user_id, item_id], load_meta)
        """
        import numpy as np
        import pandas as pd

        # subset 名称映射（兼容 TAAC2025 子集名）
        subset_map = {"seq": "train", "behavior": "train"}
        split_key = subset_map.get(subset, subset)

        raw = self._load_raw()
        if split_key not in raw:
            # 模糊匹配
            for k in raw:
                if split_key in k or k in split_key:
                    split_key = k
                    break
            else:
                raise ValueError(
                    f"Unknown subset '{subset}'. Available: {list(raw.keys())}"
                )

        seqs = raw[split_key]
        records = []
        for user_idx, items in enumerate(seqs):
            if len(items) < self._min_seq_len:
                continue
            for item_id in items:
                records.append({"user_id": user_idx, "item_id": int(item_id)})

        df = pd.DataFrame(records)

        load_meta = None
        if len(df) > max_rows:
            rng = np.random.default_rng(seed)
            indices = sorted(
                rng.choice(len(df), max_rows, replace=False).tolist()
            )
            df = df.iloc[indices].reset_index(drop=True)
            load_meta = {"original_rows": len(records), "sampled_at_load": True}
            logger.info(
                "load_subset pre-sampled: %d → %d rows.",
                len(records), max_rows,
            )

        return df, load_meta


# ------------------------------------------------------------------
# 注册
# ------------------------------------------------------------------

@DATASET_REGISTRY.register(
    "movielens_1m",
    family="classical",
    modality=("sequential",),
    tasks=("ranking",),
)
class Movielens1M(Movielens1MDataset):
    """轻量注册包装器。"""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("variant", None)
        super().__init__(**kwargs)
