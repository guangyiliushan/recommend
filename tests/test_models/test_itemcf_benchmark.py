"""ItemCF 微基准测试（Part D1）。

使用 pytest-benchmark 测试核心函数的性能：
- cosine 相似度构建
- weighted sum 预测
- 不同规模下的内存/耗时对比

运行方式:
    uv run pytest tests/test_models/test_itemcf_benchmark.py -v --benchmark-only
"""

from collections import defaultdict
from typing import Dict, Set

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 辅助函数：生成合成交互数据
# ---------------------------------------------------------------------------


def _generate_synthetic_interactions(
    num_users: int = 1000,
    num_items: int = 500,
    num_interactions: int = 20000,
    seed: int = 42,
) -> Dict[int, Set[int]]:
    """生成合成用户-物品交互数据。

    使用幂律分布模拟真实长尾。
    """
    np.random.seed(seed)

    # 幂律分布
    item_probs = np.arange(1, num_items + 1) ** (-0.75)
    item_probs = item_probs / item_probs.sum()
    user_probs = np.arange(1, num_users + 1) ** (-0.75)
    user_probs = user_probs / user_probs.sum()

    selected_users = np.random.choice(num_users, size=num_interactions, p=user_probs)
    selected_items = np.random.choice(num_items, size=num_interactions, p=item_probs)

    user_items: Dict[int, Set[int]] = defaultdict(set)
    seen = set()
    for uid, iid in zip(selected_users, selected_items, strict=False):
        pair = (uid, iid)
        if pair not in seen:
            seen.add(pair)
            user_items[uid].add(iid)

    return dict(user_items)


# ---------------------------------------------------------------------------
# 基准测试
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(min_rounds=3, warmup=True)
def test_benchmark_cosine_similarity_1k(benchmark):
    """1k 物品的 cosine 相似度矩阵构建（fit）耗时。"""
    from recsys.models.classical.item_based_cf import ItemBasedCF

    user_items = _generate_synthetic_interactions(
        num_users=1000, num_items=1000, num_interactions=50000, seed=42
    )

    def run_fit():
        model = ItemBasedCF(similarity="cosine", top_k_neighbors=50)
        model.fit(user_items_dict=user_items)

    benchmark(run_fit)


@pytest.mark.benchmark(min_rounds=3, warmup=True)
def test_benchmark_cosine_similarity_5k(benchmark):
    """5k 物品的 cosine 相似度矩阵构建（fit）耗时。"""
    from recsys.models.classical.item_based_cf import ItemBasedCF

    user_items = _generate_synthetic_interactions(
        num_users=3000, num_items=5000, num_interactions=100000, seed=42
    )

    def run_fit():
        model = ItemBasedCF(similarity="cosine", top_k_neighbors=50)
        model.fit(user_items_dict=user_items)

    benchmark(run_fit)


@pytest.mark.benchmark(min_rounds=3, warmup=True)
def test_benchmark_weighted_sum_prediction(benchmark):
    """Weighted sum 预测耗时（含 Σ|sim| 归一化）。"""
    from recsys.models.classical.item_based_cf import ItemBasedCF

    # 先 fit
    model = ItemBasedCF(similarity="cosine", top_k_neighbors=50)
    user_items = _generate_synthetic_interactions(
        num_users=1000, num_items=1000, num_interactions=50000, seed=42
    )
    model.fit(user_items_dict=user_items)

    # 生成测试用户的映射
    test_users = {uid: items for uid, items in list(user_items.items())[:100]}

    def run_predict():
        model.predict(user_train_items=test_users, user_test_items=test_users, k=10)

    benchmark(run_predict)


@pytest.mark.benchmark(min_rounds=3, warmup=True)
def test_benchmark_iuf_vs_cosine(benchmark):
    """对比 IUF vs Cosine 相似度的构建速度。"""
    from recsys.models.classical.item_based_cf import ItemBasedCF

    user_items = _generate_synthetic_interactions(
        num_users=1000, num_items=1000, num_interactions=50000, seed=42
    )

    def run_fit_iuf():
        model = ItemBasedCF(similarity="iuf", top_k_neighbors=50)
        model.fit(user_items_dict=user_items)

    benchmark(run_fit_iuf)


@pytest.mark.benchmark
def test_benchmark_top_k_variation(benchmark):
    """不同 Top-K 邻居数对性能的影响（k=20, k=100）。"""
    from recsys.models.classical.item_based_cf import ItemBasedCF

    user_items = _generate_synthetic_interactions(
        num_users=1000, num_items=1000, num_interactions=50000, seed=42
    )

    def run_fit_k100():
        model = ItemBasedCF(similarity="cosine", top_k_neighbors=100)
        model.fit(user_items_dict=user_items)

    benchmark(run_fit_k100)
