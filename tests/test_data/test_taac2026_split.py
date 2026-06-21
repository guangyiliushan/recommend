"""TAAC2026 _TabularSplit 快速提取方法 + ID dense remap 测试。"""

from recsys.data.datasets.taac2026 import (
    _TabularSplit,
    _build_dense_id_map,
    _remap_rows_inplace,
)


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


# ------------------------------------------------------------------
# ID dense remap 测试
# ------------------------------------------------------------------


def test_build_dense_id_map_sparse():
    """稀疏原始 ID 应映射为从 1 开始的连续 ID，0 被跳过。"""
    raw_ids = [0, 100, 0, 2000, 100, 9999, 0]
    mapping = _build_dense_id_map(raw_ids)
    # 0 不参与映射
    assert 0 not in mapping
    # 三个唯一非零 ID → 1, 2, 3
    assert mapping == {100: 1, 2000: 2, 9999: 3}


def test_remap_rows_inplace_shared_mapping():
    """remap 后 train/val/test 使用同一映射，原始 ID 在不同 split 中编出相同值。"""
    rows = [
        {"user_id": 100, "item_id": 500, "label_type": 1},
        {"user_id": 200, "item_id": 600, "label_type": 0},
        {"user_id": 100, "item_id": 600, "label_type": 1},  # 同一用户
        {"user_id": 300, "item_id": 500, "label_type": 0},  # 同一物品
    ]
    user_map = _build_dense_id_map([100, 200, 300])
    item_map = _build_dense_id_map([500, 600])
    n_users, n_items = _remap_rows_inplace(rows, user_map, item_map)

    # 验证计数
    assert n_users == 3
    assert n_items == 2

    # 验证 remap 后 ID 连续
    remapped_uids = set()
    remapped_iids = set()
    for row in rows:
        remapped_uids.add(row["user_id"])
        remapped_iids.add(row["item_id"])

    # 用户 ID 应为 {1, 2, 3}，物品 ID 应为 {1, 2}
    assert remapped_uids == {1, 2, 3}
    assert remapped_iids == {1, 2}

    # 同一原始用户 ID=100 在两行中映射为相同值
    assert rows[0]["user_id"] == rows[2]["user_id"]
    # 同一原始物品 ID=500 在两行中映射为相同值
    assert rows[0]["item_id"] == rows[3]["item_id"]


def test_dense_remap_max_id_equals_count():
    """remap 后 max(ID) 应等于唯一计数，确保 embedding 只需 num_users+1。"""
    user_map = _build_dense_id_map([7, 42, 999])
    item_map = _build_dense_id_map([1000, 2000, 3000])

    rows = [
        {"user_id": 7, "item_id": 1000, "label_type": 1},
        {"user_id": 42, "item_id": 2000, "label_type": 1},
        {"user_id": 999, "item_id": 3000, "label_type": 0},
    ]
    n_users, n_items = _remap_rows_inplace(rows, user_map, item_map)

    max_uid = max(r["user_id"] for r in rows)
    max_iid = max(r["item_id"] for r in rows)
    assert max_uid == n_users == 3
    assert max_iid == n_items == 3
