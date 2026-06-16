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
    assert model._normalize is True


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
