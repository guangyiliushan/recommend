"""Shared test fixtures for RecBench."""

from typing import Dict, List, Set

import numpy as np
import pytest
import torch

from recsys import auto_discover_models, get_model
from recsys.core.base_model import Batch
from recsys.core.prediction_bundle import PredictionBundle

# ---------------------------------------------------------------------------
# 模型发现
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def discover_models():
    """会话级自动发现模型。"""
    auto_discover_models()


# ---------------------------------------------------------------------------
# 合成数据 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_interactions() -> List[tuple]:
    """合成交互数据：5 个用户、8 个物品。

    Returns
    -------
    List[tuple]
        (user_id, item_id) 交互对列表。
    """
    return [
        (0, 0), (0, 1), (0, 2),
        (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 3), (2, 4),
        (3, 4), (3, 5), (3, 6),
        (4, 0), (4, 6), (4, 7),
    ]


@pytest.fixture
def user_train_items() -> Dict[int, Set[int]]:
    """用户训练集物品。

    Returns
    -------
    Dict[int, Set[int]]
        user_id -> {item_id, ...} 映射。
    """
    return {
        0: {0, 1, 2},
        1: {1, 2, 3},
        2: {0, 3, 4},
        3: {4, 5, 6},
        4: {0, 6, 7},
    }


@pytest.fixture
def user_test_items() -> Dict[int, Set[int]]:
    """用户测试集物品。

    Returns
    -------
    Dict[int, Set[int]]
        user_id -> {item_id, ...} 映射。
    """
    return {
        0: {3},
        1: {4},
        2: {1},
        3: {0},
        4: {5},
    }


# ---------------------------------------------------------------------------
# 模型 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def itemcf_model():
    """ItemCF 模型实例。

    Returns
    -------
    ItemBasedCF
        配置好的 ItemCF 模型。
    """
    itemcf_cls = get_model("itemcf")
    return itemcf_cls(similarity="cosine", top_k_neighbors=10, recommend_k=5)


@pytest.fixture
def fitted_itemcf_model(itemcf_model, synthetic_interactions):
    """已拟合的 ItemCF 模型。

    Returns
    -------
    ItemBasedCF
        已在合成数据上拟合的 ItemCF 模型。
    """
    itemcf_model.fit(synthetic_interactions)
    return itemcf_model


# ---------------------------------------------------------------------------
# Batch fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_batch() -> Batch:
    """合成 Batch 用于模型前向测试。

    Returns
    -------
    Batch
        包含 user_id, item_id, label 的合成 batch。
    """
    data = {
        "user_id": torch.tensor([0, 0, 1, 1, 2]),
        "item_id": torch.tensor([0, 1, 2, 3, 4]),
        "label": torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]),
        "group_id": torch.tensor([0, 0, 1, 1, 2]),
    }
    return Batch(data=data)


@pytest.fixture
def ranking_batch() -> Batch:
    """排序任务 Batch。

    Returns
    -------
    Batch
        包含 group_id, candidate_item_ids, label 的排序 batch。
    """
    data = {
        "group_id": torch.tensor([0, 0, 0, 1, 1, 1]),
        "candidate_item_ids": torch.tensor([[10, 20, 30], [11, 21, 31]]),
        "label": torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32),
    }
    return Batch(data=data)


# ---------------------------------------------------------------------------
# PredictionBundle fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ranking_prediction_bundle() -> PredictionBundle:
    """排序任务 PredictionBundle。

    Returns
    -------
    PredictionBundle
        包含 group_ids, y_score, candidate_ids 的排序预测结果。
    """
    return PredictionBundle(
        task_type="ranking",
        problem_type="implicit_ranking",
        group_ids=np.array([0, 0, 0, 1, 1, 1]),
        y_score=np.array([0.9, 0.5, 0.1, 0.8, 0.6, 0.2]),
        candidate_ids=np.array([10, 20, 30, 11, 21, 31]),
        y_true=np.array([1, 0, 0, 0, 1, 0]),
        score_type="raw_score",
    )


@pytest.fixture
def classification_prediction_bundle() -> PredictionBundle:
    """分类任务 PredictionBundle。

    Returns
    -------
    PredictionBundle
        包含 y_true, y_score 的分类预测结果。
    """
    return PredictionBundle(
        task_type="classification",
        problem_type="binary_classification",
        y_true=np.array([0, 0, 1, 1, 0, 1]),
        y_score=np.array([0.1, 0.2, 0.9, 0.85, 0.15, 0.95]),
        score_type="prob",
    )
