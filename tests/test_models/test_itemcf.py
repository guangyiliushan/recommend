"""ItemCF 单元测试：验证实例化、fit/predict 与 PredictionBundle 字段。"""

from recsys import auto_discover_models, get_model
from recsys.core.prediction_bundle import PredictionBundle


def test_itemcf_instantiation():
    """ItemCF 可正常实例化，默认参数正确。"""
    auto_discover_models()
    itemcf_cls = get_model("itemcf")
    model = itemcf_cls()
    assert model._similarity == "cosine"
    assert model._top_k_neighbors == 50
    assert model._recommend_k == 10
    assert model._normalize is False


def test_itemcf_fit_predict():
    """用合成数据调 fit() + predict()，返回 PredictionBundle。"""
    auto_discover_models()
    itemcf_cls = get_model("itemcf")
    model = itemcf_cls(similarity="cosine", top_k_neighbors=10, recommend_k=5)

    # 合成交互数据：5 个用户、8 个物品
    user_item_pairs = [
        (0, 0), (0, 1), (0, 2),
        (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 3), (2, 4),
        (3, 4), (3, 5), (3, 6),
        (4, 0), (4, 6), (4, 7),
    ]
    model.fit(user_item_pairs)

    user_train_items = {
        0: {0, 1, 2},
        1: {1, 2, 3},
        2: {0, 3, 4},
    }
    user_test_items = {
        0: {3},
        1: {4},
        2: {1},
    }
    bundle = model.predict(
        user_train_items=user_train_items,
        user_test_items=user_test_items,
    )
    assert isinstance(bundle, PredictionBundle)


def test_itemcf_prediction_bundle_fields():
    """PredictionBundle 包含 task_type、problem_type、group_ids 等必填字段。"""
    auto_discover_models()
    itemcf_cls = get_model("itemcf")
    model = itemcf_cls(similarity="cosine", top_k_neighbors=10, recommend_k=5)

    user_item_pairs = [
        (0, 0), (0, 1), (0, 2),
        (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 3), (2, 4),
    ]
    model.fit(user_item_pairs)

    user_train_items = {0: {0, 1, 2}, 1: {1, 2, 3}, 2: {0, 3, 4}}
    user_test_items = {0: {3}, 1: {4}, 2: {1}}

    bundle = model.predict(
        user_train_items=user_train_items,
        user_test_items=user_test_items,
    )

    assert bundle.task_type == "ranking"
    assert bundle.problem_type == "implicit_ranking"
    assert bundle.group_ids is not None
    assert len(bundle.group_ids) > 0
    assert bundle.y_score is not None
    assert bundle.candidate_ids is not None
    assert bundle.score_type == "raw_score"


def test_itemcf_weighted_sum_normalization():
    """验证 weighted sum 预测包含 Σ|sim| 分母归一化。

    构造 3 物品、2 用户的小案例，验证预测值在合理范围内。
    """
    from recsys.models.classical.item_based_cf import ItemBasedCF

    model = ItemBasedCF(similarity="cosine", top_k_neighbors=10, recommend_k=5)

    # 简单案例：2 个用户，3 个物品
    user_item_pairs = [
        (0, 0), (0, 1),  # 用户 0 交互了物品 0 和 1
        (1, 0), (1, 2),  # 用户 1 交互了物品 0 和 2
    ]
    model.fit(user_item_pairs)

    # 预测
    bundle = model.predict(
        user_train_items={0: {0, 1}},
        user_test_items={0: {2}},
        k=5,
    )

    assert bundle is not None
    assert len(bundle.y_score) > 0
    # 验证分数在 [0,1] 范围内（归一化的 weighted sum）
    for scores in bundle.y_score:
        for s in scores:
            assert 0.0 <= s <= 1.0, f"Prediction score {s} out of range [0,1]"


def test_itemcf_iuf_similarity():
    """IUF 加权余弦相似度可正常 fit 和 predict。"""
    from recsys.models.classical.item_based_cf import ItemBasedCF

    model = ItemBasedCF(similarity="iuf", top_k_neighbors=10, recommend_k=5)

    user_item_pairs = [
        (0, 0), (0, 1), (0, 2),
        (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 3), (2, 4),
    ]
    model.fit(user_item_pairs)

    assert model._fitted
    assert model._sim_matrix is not None


def test_itemcf_compute_backends():
    """Compute backend 工厂函数返回正确的后端实例。"""
    from recsys.models.classical.itemcf_backends import NumpyBackend, get_compute_backend

    backend = get_compute_backend("numpy")
    assert isinstance(backend, NumpyBackend)

    # auto 应该总是可用（至少回退到 numpy）
    backend_auto = get_compute_backend("auto")
    assert backend_auto is not None


def test_itemcf_partial_fit():
    """增量更新 partial_fit 可被调用（基础调用检查）。"""
    from recsys.models.classical.item_based_cf import ItemBasedCF

    model = ItemBasedCF(similarity="cosine", top_k_neighbors=10, recommend_k=5)

    user_item_pairs = [
        (0, 0), (0, 1), (0, 2),
        (1, 1), (1, 2), (1, 3),
    ]
    model.fit(user_item_pairs)

    # 增量更新
    new_pairs = [(0, 3), (1, 0)]
    model.partial_fit(new_pairs)

    assert model._fitted


def test_itemcf_empty_input():
    """空输入应抛出有意义错误。"""
    import pytest

    from recsys.models.classical.item_based_cf import ItemBasedCF

    model = ItemBasedCF()

    with pytest.raises(ValueError, match="必须提供"):
        model.fit()

    with pytest.raises(ValueError, match="user_item_pairs 不能为空"):
        model.fit(user_item_pairs=[])
