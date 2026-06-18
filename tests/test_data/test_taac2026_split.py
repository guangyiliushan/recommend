"""TAAC2026 _TabularSplit 快速提取方法测试。"""

from recsys.data.datasets.taac2026 import _TabularSplit


def _make_split(rows: list):
    """构造最小化 _TabularSplit 实例。"""
    return _TabularSplit(rows, user_cols=[], item_cols=[], domain_cols=[])


def test_iter_user_item_pairs_fast():
    """验证 iter_user_item_pairs_fast 产出正确的 (user_id, item_id) 对。"""
    rows = [
        {"user_id": 10, "item_id": 100, "label_type": 1},
        {"user_id": 20, "item_id": 200, "label_type": 0},
        {"user_id": 30, "item_id": 300, "label_type": 1},
    ]
    split = _make_split(rows)

    pairs = list(split.iter_user_item_pairs_fast())

    assert pairs == [(10, 100), (20, 200), (30, 300)]
    assert len(pairs) == 3


def test_extract_user_item_mapping_fast():
    """验证 extract_user_item_mapping_fast 正确去重并构建映射。"""
    rows = [
        {"user_id": 1, "item_id": 10, "label_type": 1},
        {"user_id": 1, "item_id": 20, "label_type": 1},  # 同一用户，不同物品
        {"user_id": 2, "item_id": 10, "label_type": 0},  # 同一物品，不同用户
        {"user_id": 2, "item_id": 10, "label_type": 1},  # 重复交互（应去重）
    ]
    split = _make_split(rows)

    mapping = split.extract_user_item_mapping_fast()

    assert set(mapping.keys()) == {1, 2}
    assert mapping[1] == {10, 20}
    assert mapping[2] == {10}  # 重复 item_id 被 set 去重


def test_fast_path_consistency():
    """快速路径产出与 __getitem__ 迭代一致。"""
    rows = [
        {"user_id": 5, "item_id": 50, "label_type": 1},
        {"user_id": 6, "item_id": 60, "label_type": 1},
        {"user_id": 7, "item_id": 70, "label_type": 0},
    ]
    split = _make_split(rows)

    # 快速路径
    fast_pairs = list(split.iter_user_item_pairs_fast())
    fast_mapping = split.extract_user_item_mapping_fast()

    # 慢路径：通过 __getitem__ 逐个获取
    slow_pairs = []
    slow_mapping = {}
    for i in range(len(split)):
        sample = split[i]
        uid = int(sample["user_id"].item())
        iid = int(sample["item_id"].item())
        slow_pairs.append((uid, iid))
        slow_mapping.setdefault(uid, set()).add(iid)

    assert fast_pairs == slow_pairs
    assert fast_mapping == slow_mapping
