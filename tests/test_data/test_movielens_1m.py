"""MovieLens-1M 数据集适配器单元测试。

验证 JSON 加载、SequenceSplit 集成、ItemCF fit/predict 等核心路径。
"""

import json
from pathlib import Path

import numpy as np
import pytest

from recsys.data.datasets.movielens_1m import Movielens1MDataset
from recsys.data.split_utils import SequenceSplit


def _make_mock_raw() -> dict:
    """构造最小化 mock 数据（3 用户，每个有 4-5 个物品的序列）。"""
    return {
        "train": [
            [10, 20, 30, 40],
            [20, 30, 50],
            [10, 40, 60, 70, 80],
        ],
        "val": [
            [15, 25, 35],
            [45, 55],
        ],
        "test": [
            [18, 28],
            [48, 58, 68],
        ],
    }


def test_movielens_load_raw(tmp_path: Path):
    """验证 _load_raw 从缓存目录读取 JSON。"""
    cache_dir = tmp_path / "movielens_1m"
    cache_dir.mkdir(parents=True)

    raw_data = _make_mock_raw()
    for split, data in raw_data.items():
        fname = {
            "train": "train_data.json",
            "val": "validation_data.json",
            "test": "test_data.json",
        }[split]
        (cache_dir / fname).write_text(json.dumps(data), encoding="utf-8")

    ds = Movielens1MDataset(root_dir=str(tmp_path), min_seq_len=2)
    raw = ds._load_raw()

    assert len(raw["train"]) == 3
    assert len(raw["val"]) == 2
    assert len(raw["test"]) == 2
    assert raw["train"][0] == [10, 20, 30, 40]


def test_movielens_prepare_splits(tmp_path: Path):
    """验证 _prepare_splits 产出正确的 SequenceSplit。"""
    ds = Movielens1MDataset(root_dir=str(tmp_path), min_seq_len=2, max_seq_len=50)
    raw = _make_mock_raw()

    train, val, test = ds._prepare_splits(raw)

    assert isinstance(train, SequenceSplit)
    assert isinstance(val, SequenceSplit)
    assert isinstance(test, SequenceSplit)

    # train split: 3 用户, 序列长度 4+3+5=12, 位置数=(4-1)+(3-1)+(5-1)=9
    assert len(train) == 9

    # 验证 __getitem__ 产出正确键
    sample = train[0]
    assert "user_id" in sample
    assert "item_id" in sample
    assert "item_ids" in sample
    assert "labels" in sample
    assert "candidate_items" in sample


def test_movielens_metadata(tmp_path: Path):
    """验证 num_users / num_items 元数据。"""
    ds = Movielens1MDataset(root_dir=str(tmp_path), min_seq_len=2)
    raw = _make_mock_raw()

    ds._prepare_splits(raw)

    assert ds.num_users == 3
    assert ds.num_items == 8  # 物品 10,20,30,40,50,60,70,80


def test_movielens_itemcf_fit_predict(tmp_path: Path):
    """端到端：加载 mock 数据 → ItemCF fit → predict。"""
    from recsys.models.classical.item_based_cf import ItemBasedCF

    ds = Movielens1MDataset(root_dir=str(tmp_path), min_seq_len=2, max_seq_len=50)
    raw = _make_mock_raw()
    train_split, _, test_split = ds._prepare_splits(raw)

    # 提取用户-物品映射（直接使用返回的 split，不调用 get_split）
    train_mapping = train_split.extract_user_item_mapping_fast()
    test_mapping = test_split.extract_user_item_mapping_fast()

    # ItemCF
    model = ItemBasedCF(similarity="cosine", top_k_neighbors=10, recommend_k=5)
    model.fit(user_items_dict=train_mapping)

    # predict 不崩溃
    train_for_predict = {uid: set(items.tolist()) if hasattr(items, "tolist") else set(items)
                         for uid, items in train_mapping.items()}
    test_for_predict = {uid: set(items.tolist()) if hasattr(items, "tolist") else set(items)
                        for uid, items in test_mapping.items()}

    bundle = model.predict(
        user_train_items=train_for_predict,
        user_test_items=test_for_predict,
    )
    assert bundle is not None
    assert bundle.task_type == "ranking"
    # MovieLens mock 数据有跨用户共现，应产出非空结果
    assert len(bundle.group_ids) > 0


def test_sequence_split_basic():
    """验证 SequenceSplit 按 ((seq_len-1)) 计算位置数。"""
    result = SequenceSplit(
        user_ids=np.array([0, 1, 2], dtype=np.int64),
        item_sequences=[
            np.array([1], dtype=np.int64),              # 长度 1 → 0 个位置
            np.array([1, 2], dtype=np.int64),           # 长度 2 → 1 个位置
            np.array([1, 2, 3, 4], dtype=np.int64),     # 长度 4 → 3 个位置
        ],
        max_seq_len=50,
    )
    # 位置总数 = 0 + 1 + 3 = 4
    assert len(result) == 4


def test_movielens_empty_seq_filtered(tmp_path: Path):
    """min_seq_len 过滤后无用户的边界情况。"""
    ds = Movielens1MDataset(root_dir=str(tmp_path), min_seq_len=100)
    raw = {
        "train": [[1, 2]],
        "val": [[3]],
        "test": [[4, 5]],
    }
    with pytest.raises(AssertionError, match="训练集中无有效物品"):
        ds._prepare_splits(raw)
