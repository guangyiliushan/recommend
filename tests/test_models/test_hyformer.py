"""HyFormer 模型测试。"""

import pytest
import torch

from recsys import auto_discover_models, get_model
from recsys.core.base_model import Batch, ModelOutput


@pytest.fixture(autouse=True)
def setup():
    """自动发现模型。"""
    auto_discover_models()


class TestHyFormerInstantiation:
    """HyFormer 实例化测试。"""

    def test_hyformer_instantiation(self):
        """测试 HyFormer 实例化。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 32, "emb_dim": 32},
            schema_metadata={"num_users": 100, "num_items": 500},
        )
        assert model.model_name == "hyformer"
        assert model.supports_training is True
        assert model.task_type == "pointwise"

    def test_hyformer_default_config(self):
        """测试默认配置。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls()
        assert model.d_model == 64
        assert model.emb_dim == 64
        assert model.num_heads == 4
        assert model.num_blocks == 2
        assert model.dropout == 0.1


class TestHyFormerForward:
    """HyFormer 前向传播测试。"""

    def test_hyformer_forward(self):
        """测试 HyFormer 前向传播。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 32, "emb_dim": 32},
            schema_metadata={"num_users": 100, "num_items": 500},
        )

        batch = Batch(
            data={
                "user_id": torch.randint(1, 100, (16,)),
                "item_id": torch.randint(1, 500, (16,)),
                "label": torch.randint(0, 2, (16,)).float(),
            }
        )

        output = model(batch)
        assert isinstance(output, ModelOutput)
        assert output.scores is not None
        assert output.scores.shape == (16,)

    def test_hyformer_forward_missing_fields(self):
        """测试缺少必要字段时抛出异常。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 32, "emb_dim": 32},
            schema_metadata={"num_users": 100, "num_items": 500},
        )

        batch = Batch(data={"user_id": torch.randint(1, 100, (16,))})

        with pytest.raises(ValueError, match="user_id 和 item_id"):
            model(batch)


class TestHyFormerSparseDenseParams:
    """HyFormer 稀疏/密集参数分离测试。"""

    def test_hyformer_sparse_dense_params(self):
        """测试稀疏/密集参数分离。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 32, "emb_dim": 32},
            schema_metadata={"num_users": 100, "num_items": 500},
        )

        sparse_params = model.get_sparse_params()
        dense_params = model.get_dense_params()

        assert len(sparse_params) == 2  # user_emb + item_emb
        assert len(dense_params) > 0

    def test_sparse_params_are_embeddings(self):
        """测试稀疏参数是 Embedding 权重。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 32, "emb_dim": 32},
            schema_metadata={"num_users": 100, "num_items": 500},
        )

        sparse_params = model.get_sparse_params()
        # 检查是否是 Embedding 权重
        assert sparse_params[0] is model.user_emb.weight
        assert sparse_params[1] is model.item_emb.weight


class TestHyFormerComputeLoss:
    """HyFormer 损失计算测试。"""

    def test_hyformer_compute_loss(self):
        """测试损失计算。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 32, "emb_dim": 32},
            schema_metadata={"num_users": 100, "num_items": 500},
        )

        batch = Batch(
            data={
                "user_id": torch.randint(1, 100, (16,)),
                "item_id": torch.randint(1, 500, (16,)),
                "label": torch.randint(0, 2, (16,)).float(),
            }
        )

        output = model(batch)
        loss_dict = model.compute_loss(batch, output)

        assert "loss" in loss_dict
        assert loss_dict["loss"].requires_grad

    def test_hyformer_compute_loss_with_labels_field(self):
        """测试使用 labels 字段计算损失。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 32, "emb_dim": 32},
            schema_metadata={"num_users": 100, "num_items": 500},
        )

        batch = Batch(
            data={
                "user_id": torch.randint(1, 100, (16,)),
                "item_id": torch.randint(1, 500, (16,)),
                "labels": torch.randint(0, 2, (16,)).float(),
            }
        )

        output = model(batch)
        loss_dict = model.compute_loss(batch, output)

        assert "loss" in loss_dict
        assert loss_dict["loss"].requires_grad

    def test_hyformer_compute_loss_missing_label(self):
        """测试缺少标签时抛出异常。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 32, "emb_dim": 32},
            schema_metadata={"num_users": 100, "num_items": 500},
        )

        batch = Batch(
            data={
                "user_id": torch.randint(1, 100, (16,)),
                "item_id": torch.randint(1, 500, (16,)),
            }
        )

        output = model(batch)

        with pytest.raises(ValueError, match="label 或 labels"):
            model.compute_loss(batch, output)


class TestHyFormerModelRegistry:
    """HyFormer 模型注册测试。"""

    def test_hyformer_registered(self):
        """测试 HyFormer 已注册。"""
        from recsys.core.registry import MODEL_REGISTRY

        assert "hyformer" in MODEL_REGISTRY
        assert MODEL_REGISTRY.get_metadata("hyformer")["family"] == "unified"
        assert MODEL_REGISTRY.get_metadata("hyformer")["task_type"] == "pointwise"
        assert MODEL_REGISTRY.get_metadata("hyformer")["supports_training"] is True
