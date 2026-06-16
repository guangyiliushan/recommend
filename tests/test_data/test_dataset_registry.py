"""Dataset registry 单元测试：验证数据集注册与发现。"""

import pytest

from recsys.core.registry import DATASET_REGISTRY


class TestDatasetRegistry:
    """数据集注册表测试。"""

    def test_list_datasets(self):
        """list() 返回已注册数据集列表。"""
        import recsys.data.dataset_registry  # noqa: F401

        datasets = DATASET_REGISTRY.list()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_taac2026_data_sample_registered(self):
        """taac2026_data_sample 已注册。"""
        import recsys.data.dataset_registry  # noqa: F401

        datasets = DATASET_REGISTRY.list()
        assert "taac2026_data_sample" in datasets, f"taac2026_data_sample not found in {datasets}"

    def test_taac2025_data_registered(self):
        """taac2025 数据集已注册。"""
        import recsys.data.dataset_registry  # noqa: F401

        datasets = DATASET_REGISTRY.list()
        # taac2025 有两个版本：taac2025_10M 和 taac2025_1M
        has_taac2025 = any("taac2025" in ds for ds in datasets)
        assert has_taac2025, f"No taac2025 dataset found in {datasets}"

    def test_get_dataset_info(self):
        """get() 返回数据集类。"""
        import recsys.data.dataset_registry  # noqa: F401

        dataset_cls = DATASET_REGISTRY.get("taac2026_data_sample")
        assert dataset_cls is not None
        # 返回的是数据集类
        assert hasattr(dataset_cls, "__name__")

    def test_get_nonexistent_dataset(self):
        """获取未注册数据集抛 KeyError。"""
        import recsys.data.dataset_registry  # noqa: F401

        with pytest.raises(KeyError):
            DATASET_REGISTRY.get("nonexistent_dataset_xyz")
