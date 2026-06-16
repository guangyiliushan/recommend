"""注册表冒烟测试：确保模型与数据集注册链通畅。"""

from recsys import auto_discover_models, get_model, list_models
from recsys.core.registry import DATASET_REGISTRY


def test_model_registry_discover():
    """auto_discover_models() 后 list_models() 包含 itemcf。"""
    auto_discover_models()
    models = list_models()
    assert "itemcf" in models, f"itemcf not found in registered models: {models}"


def test_dataset_registry_list():
    """DATASET_REGISTRY.list() 包含 taac2026_data_sample。"""
    import recsys.data.dataset_registry  # noqa: F401, E402

    datasets = DATASET_REGISTRY.list()
    assert "taac2026_data_sample" in datasets, (
        f"taac2026_data_sample not found in registered datasets: {datasets}"
    )


def test_get_model_itemcf():
    """get_model('itemcf') 返回 ItemBasedCF 类。"""
    auto_discover_models()
    model_cls = get_model("itemcf")
    from recsys.models.classical.item_based_cf import ItemBasedCF  # noqa: E402

    assert model_cls is ItemBasedCF
