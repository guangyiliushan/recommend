"""可控规模的合成交互数据集，用于性能基准测试（Part B）。

支持生成不同规模、不同稀疏度、显式/隐式的合成交互数据。
使用幂律分布模拟真实长尾，与 NegativeSamplingConfig.popularity_power 默认值一致。
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from recsys.core.base_dataset import BaseDataset
from recsys.core.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register(
    "synthetic",
    supports_multi_subset=False,
    supports_candidates=False,
    supports_vector_embeddings=False,
    default_eda_subset="train",
)
class SyntheticDataset(BaseDataset):
    """可控规模的合成交互数据集，用于性能基准测试。

    参数通过 data_config 传入：
        num_users: 用户数 (默认 10000)
        num_items: 物品数 (默认 5000)
        num_interactions: 交互总数 (默认 1000000)
        sparsity: 目标稀疏度 (可选，覆盖 num_interactions)
        rating_scale: None=隐式, (1,5)=显式评分
        seed: 随机种子
        popularity_power: 长尾分布幂次 (默认 0.75，与 NegativeSamplingConfig 一致)
    """

    dataset_name = "synthetic"
    dataset_url = "synthetic://local"
    feature_cols = ["user_id", "item_id"]
    label_col = "label"

    def __init__(
        self,
        root_dir: str = "./data",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        max_seq_len: int = 50,
        min_seq_len: int = 2,
        neg_sample_count: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            root_dir, split_ratios, max_seq_len, min_seq_len, neg_sample_count, **kwargs
        )

        # 数据集配置参数
        self._num_users = kwargs.get("num_users", 10000)
        self._num_items = kwargs.get("num_items", 5000)
        self._num_interactions = kwargs.get("num_interactions", 1_000_000)
        self._sparsity = kwargs.get("sparsity")  # 可选，覆盖 num_interactions
        self._rating_scale = kwargs.get("rating_scale")  # None=隐式, (min,max)=显式
        self._seed = kwargs.get("seed", 42)
        self._popularity_power = kwargs.get("popularity_power", 0.75)

        # 内部状态
        self._raw_interactions: Optional[List[Tuple[int, int]]] = None
        self._raw_ratings: Optional[Dict[Tuple[int, int], float]] = None

    @property
    def num_users(self) -> int:
        return self._num_users

    @property
    def num_items(self) -> int:
        return self._num_items

    def _load_raw(self) -> Any:
        """生成合成交互数据（无需下载，内存生成）。"""
        if self._raw_interactions is None:
            self._raw_interactions, self._raw_ratings = self._generate_synthetic_data()

        return {
            "interactions": self._raw_interactions,
            "ratings": self._raw_ratings,
        }

    def _generate_synthetic_data(
        self,
    ) -> Tuple[List[Tuple[int, int]], Optional[Dict[Tuple[int, int], float]]]:
        """生成合成交互数据。

        策略：
            1. 使用幂律分布生成物品流行度
            2. 使用幂律分布生成用户活跃度
            3. 根据流行度/活跃度生成交互数据
            4. 如果 rating_scale 不为 None，生成显式评分

        Returns
        -------
        interactions : list of (user_id, item_id)
            用户-物品交互对列表。
        ratings : dict of {(user_id, item_id): float}, optional
            显式评分数据（仅当 rating_scale 不为 None 时）。
        """
        np.random.seed(self._seed)

        # 如果指定了 sparsity，计算对应的 num_interactions
        if self._sparsity is not None:
            total_possible = self._num_users * self._num_items
            self._num_interactions = int(total_possible * self._sparsity)

        # 1. 生成物品流行度（幂律分布）
        # p(item) ∝ popularity^power
        item_probs = np.arange(1, self._num_items + 1) ** (-self._popularity_power)
        item_probs = item_probs / item_probs.sum()

        # 2. 生成用户活跃度（幂律分布）
        # p(user) ∝ activity^power
        user_probs = np.arange(1, self._num_users + 1) ** (-self._popularity_power)
        user_probs = user_probs / user_probs.sum()

        # 3. 生成交互数据
        interactions = []
        ratings = {} if self._rating_scale is not None else None

        # 使用重采样选择交互对
        selected_users = np.random.choice(
            self._num_users, size=self._num_interactions, p=user_probs
        )
        selected_items = np.random.choice(
            self._num_items, size=self._num_interactions, p=item_probs
        )

        for user_id, item_id in zip(selected_users, selected_items, strict=False):
            pair = (user_id, item_id)

            # 避免重复交互
            if pair not in interactions:
                interactions.append(pair)

                # 生成显式评分
                if ratings is not None and self._rating_scale is not None:
                    min_rating, max_rating = self._rating_scale
                    rating = np.random.uniform(min_rating, max_rating)
                    ratings[pair] = float(rating)

        # 如果去重后不够，用更宽松的策略补充
        while len(interactions) < self._num_interactions:
            # 随机选择用户和物品
            user_id = np.random.randint(0, self._num_users)
            item_id = np.random.randint(0, self._num_items)
            pair = (user_id, item_id)

            if pair not in interactions:
                interactions.append(pair)

                if ratings is not None and self._rating_scale is not None:
                    min_rating, max_rating = self._rating_scale
                    rating = np.random.uniform(min_rating, max_rating)
                    ratings[pair] = float(rating)

        return interactions, ratings

    def _prepare_splits(self, raw: Any) -> Tuple[Any, Any, Any]:
        """将原始数据划分为 train/val/test splits。

        按用户级别划分，确保每个用户在所有 splits 中都有数据。
        """
        interactions = raw["interactions"]
        raw["ratings"]

        # 按用户分组
        user_interactions: Dict[int, List[Tuple[int, int]]] = {}
        for user_id, item_id in interactions:
            if user_id not in user_interactions:
                user_interactions[user_id] = []
            user_interactions[user_id].append((user_id, item_id))

        # 按用户级别划分，确保每个用户在所有 splits 中都有数据
        train_data = []
        val_data = []
        test_data = []

        for _user_id, items in user_interactions.items():
            # 打乱物品顺序
            np.random.shuffle(items)

            # 按比例划分
            n_items = len(items)
            n_train = int(n_items * self.split_ratios[0])
            n_val = int(n_items * self.split_ratios[1])
            n_test = n_items - n_train - n_val

            train_items = items[:n_train]
            val_items = items[n_train : n_train + n_val]
            test_items = items[n_train + n_val : n_train + n_val + n_test]

            # 转换为字典格式
            for user_i, item_i in train_items:
                train_data.append({"user_id": user_i, "item_id": item_i, "label": 1})
            for user_v, item_v in val_items:
                val_data.append({"user_id": user_v, "item_id": item_v, "label": 1})
            for user_t, item_t in test_items:
                test_data.append({"user_id": user_t, "item_id": item_t, "label": 1})

        # 创建 dataset 包装器
        from recsys.core.base_dataset import SplitDataset

        train_dataset = SplitDataset(train_data)
        val_dataset = SplitDataset(val_data) if val_data else SplitDataset([])
        test_dataset = SplitDataset(test_data) if test_data else SplitDataset([])

        return train_dataset, val_dataset, test_dataset

    def __len__(self) -> int:
        """总样本数（包括所有 splits）。"""
        if self._raw_interactions is None:
            return self._num_interactions
        return len(self._raw_interactions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本（默认从 train split）。"""
        if self._train is None:
            self.load()
        return self._train[idx]
