"""HyFormer 模型测试。"""

import pytest
import torch

from recsys import auto_discover_models, get_model
from recsys.core.base_model import Batch, ModelOutput


@pytest.fixture(autouse=True)
def setup():
    """自动发现模型。"""
    auto_discover_models()


# ------------------------------------------------------------------
# 共享 mock schema
# ------------------------------------------------------------------


def _make_mock_schema():
    """构造模拟 HyFormer schema_metadata。"""
    # user: 1 个 int 特征 (vocab=10) + 1 个 string 特征 (vocab=10, max 2 values)
    user_int_feature_specs = [
        (10, 0, 1),   # vocab=10, offset=0, length=1
        (10, 1, 2),   # vocab=10, offset=1, length=2
    ]
    # 1 个 item int 特征
    item_int_feature_specs = [
        (20, 0, 1),
    ]
    # 1 个序列域 domain_a，有 2 个 sideinfo 特征
    seq_vocab_sizes = {"a": [30, 30]}

    return {
        "num_users": 100,
        "num_items": 500,
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": 0,
        "item_dense_dim": 0,
        "seq_vocab_sizes": seq_vocab_sizes,
        "user_ns_groups": [[0], [1]],
        "item_ns_groups": [[0]],
        "seq_domains": ["a"],
    }


def _make_batch(batch_size: int = 8) -> Batch:
    """构造标准 batch（含完整特征）。"""
    return Batch(data={
        "label": torch.randint(0, 2, (batch_size,)).float(),
        "user_int_feats": torch.randint(0, 9, (batch_size, 3), dtype=torch.long),
        "item_int_feats": torch.randint(0, 19, (batch_size, 1), dtype=torch.long),
        "seq_data": {"a": torch.randint(0, 29, (batch_size, 2, 5), dtype=torch.long)},
        "seq_lens": {"a": torch.randint(1, 6, (batch_size,), dtype=torch.long)},
    })


# ------------------------------------------------------------------
# 实例化测试
# ------------------------------------------------------------------


class TestHyFormerInstantiation:
    """HyFormer 实例化测试。"""

    def test_hyformer_instantiation(self):
        """测试 HyFormer 实例化。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 32, "emb_dim": 32},
            schema_metadata=_make_mock_schema(),
        )
        assert model.model_name == "hyformer"
        assert model.supports_training is True
        assert model.task_type == "pointwise"
        assert model.core is not None

    def test_hyformer_default_config(self):
        """测试默认配置（小数据集自动缩容）。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={},
            schema_metadata=_make_mock_schema(),
        )
        # 小数据集 (100 users, 500 items) 触发自动缩容
        assert model.d_model == 32
        assert model.emb_dim == 32
        assert model.num_heads == 2
        assert model.num_hyformer_blocks == 1
        assert model.dropout_rate == 0.1

    def test_hyformer_full_size_config(self):
        """显式指定 d_model 时跳过自动缩容。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 64, "emb_dim": 64},
            schema_metadata=_make_mock_schema(),
        )
        assert model.d_model == 64
        assert model.emb_dim == 64
        assert model.num_heads == 4
        assert model.num_hyformer_blocks == 2

    def test_raises_on_missing_schema(self):
        """缺少 user_int_feature_specs 时抛出 ValueError。"""
        hyformer_cls = get_model("hyformer")
        with pytest.raises(ValueError, match="user_int_feature_specs"):
            hyformer_cls(
                config={"d_model": 32},
                schema_metadata={"num_users": 100, "num_items": 500},
            )


# ------------------------------------------------------------------
# 前向传播测试
# ------------------------------------------------------------------


class TestHyFormerForward:
    """HyFormer 前向传播测试。"""

    def test_hyformer_forward(self):
        """测试前向传播。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={
                "d_model": 32, "emb_dim": 16, "num_heads": 4,
                "num_hyformer_blocks": 1, "num_queries": 1,
                "dropout_rate": 0.0, "rank_mixer_mode": "ffn_only",
                "use_rope": False, "ns_tokenizer_type": "group",
            },
            schema_metadata=_make_mock_schema(),
        )

        batch = _make_batch(16)
        output = model(batch)
        assert isinstance(output, ModelOutput)
        assert output.scores is not None
        assert output.scores.shape == (16,)
        assert output.probs is not None
        assert output.probs.shape == (16,)

    def test_hyformer_forward_missing_int_feats(self):
        """测试缺少 user_int_feats 时抛出异常。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={
                "d_model": 32, "emb_dim": 16, "num_heads": 4,
                "num_hyformer_blocks": 1, "num_queries": 1,
                "dropout_rate": 0.0, "rank_mixer_mode": "ffn_only",
                "use_rope": False, "ns_tokenizer_type": "group",
            },
            schema_metadata=_make_mock_schema(),
        )

        batch = Batch(data={"user_id": torch.randint(1, 100, (16,))})
        with pytest.raises(ValueError, match="user_int_feats.*item_int_feats"):
            model(batch)


# ------------------------------------------------------------------
# 稀疏/密集参数分离测试
# ------------------------------------------------------------------


class TestHyFormerSparseDenseParams:
    """HyFormer 稀疏/密集参数分离测试。"""

    def test_hyformer_sparse_dense_params(self):
        """测试稀疏/密集参数分离。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={
                "d_model": 32, "emb_dim": 16, "num_heads": 4,
                "num_hyformer_blocks": 1, "num_queries": 1,
                "dropout_rate": 0.0, "rank_mixer_mode": "ffn_only",
                "use_rope": False, "ns_tokenizer_type": "group",
            },
            schema_metadata=_make_mock_schema(),
        )

        sparse_params = model.get_sparse_params()
        dense_params = model.get_dense_params()

        # 完整模式至少 3 个 Embedding
        assert len(sparse_params) >= 3
        assert len(dense_params) > 0

    def test_sparse_dense_disjoint(self):
        """稀疏和密集参数不重叠。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={
                "d_model": 32, "emb_dim": 16, "num_heads": 4,
                "num_hyformer_blocks": 1, "num_queries": 1,
                "dropout_rate": 0.0, "rank_mixer_mode": "ffn_only",
                "use_rope": False, "ns_tokenizer_type": "group",
            },
            schema_metadata=_make_mock_schema(),
        )
        sparse = {p.data_ptr() for p in model.get_sparse_params()}
        dense = {p.data_ptr() for p in model.get_dense_params()}
        assert sparse.isdisjoint(dense)
        all_ptrs = {p.data_ptr() for p in model.parameters()}
        assert sparse | dense == all_ptrs


# ------------------------------------------------------------------
# 损失计算测试
# ------------------------------------------------------------------


class TestHyFormerComputeLoss:
    """HyFormer 损失计算测试。"""

    def test_hyformer_compute_loss(self):
        """测试损失计算。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={
                "d_model": 32, "emb_dim": 16, "num_heads": 4,
                "num_hyformer_blocks": 1, "num_queries": 1,
                "dropout_rate": 0.0, "rank_mixer_mode": "ffn_only",
                "use_rope": False, "ns_tokenizer_type": "group",
            },
            schema_metadata=_make_mock_schema(),
        )

        batch = _make_batch(16)
        output = model(batch)
        loss_dict = model.compute_loss(batch, output)

        assert "loss" in loss_dict
        assert loss_dict["loss"].requires_grad

    def test_hyformer_compute_loss_with_labels_field(self):
        """测试使用 labels 字段计算损失。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={
                "d_model": 32, "emb_dim": 16, "num_heads": 4,
                "num_hyformer_blocks": 1, "num_queries": 1,
                "dropout_rate": 0.0, "rank_mixer_mode": "ffn_only",
                "use_rope": False, "ns_tokenizer_type": "group",
            },
            schema_metadata=_make_mock_schema(),
        )

        batch = _make_batch(16)
        # 将 label 重命名为 labels
        batch.data["labels"] = batch.data.pop("label")
        output = model(batch)
        loss_dict = model.compute_loss(batch, output)

        assert "loss" in loss_dict
        assert loss_dict["loss"].requires_grad

    def test_hyformer_compute_loss_missing_label(self):
        """测试缺少标签时抛出异常。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={
                "d_model": 32, "emb_dim": 16, "num_heads": 4,
                "num_hyformer_blocks": 1, "num_queries": 1,
                "dropout_rate": 0.0, "rank_mixer_mode": "ffn_only",
                "use_rope": False, "ns_tokenizer_type": "group",
            },
            schema_metadata=_make_mock_schema(),
        )

        batch = _make_batch(16)
        batch.data.pop("label")
        output = model(batch)

        with pytest.raises(ValueError, match="label 或 labels"):
            model.compute_loss(batch, output)

    def test_hyformer_gradient(self):
        """测试梯度回传。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={
                "d_model": 32, "emb_dim": 16, "num_heads": 4,
                "num_hyformer_blocks": 1, "num_queries": 1,
                "dropout_rate": 0.0, "rank_mixer_mode": "ffn_only",
                "use_rope": False, "ns_tokenizer_type": "group",
            },
            schema_metadata=_make_mock_schema(),
        )

        batch = _make_batch(4)
        output = model(batch)
        loss_dict = model.compute_loss(batch, output)
        loss_dict["loss"].backward()

        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0


# ------------------------------------------------------------------
# reinit 测试
# ------------------------------------------------------------------


class TestHyFormerReinit:
    """HyFormer reinit_high_cardinality_params 测试。"""

    def test_reinit_high_cardinality(self):
        """测试 reinit_high_cardinality_params。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={
                "d_model": 32, "emb_dim": 16, "num_heads": 4,
                "num_hyformer_blocks": 1, "num_queries": 1,
                "dropout_rate": 0.0, "rank_mixer_mode": "ffn_only",
                "use_rope": False, "ns_tokenizer_type": "group",
            },
            schema_metadata=_make_mock_schema(),
        )
        # 用低阈值触发所有 Embedding 的重置
        reinit_ptrs = model.reinit_high_cardinality_params(cardinality_threshold=1)
        assert len(reinit_ptrs) > 0


# ------------------------------------------------------------------
# 模型注册测试
# ------------------------------------------------------------------


class TestHyFormerModelRegistry:
    """HyFormer 模型注册测试。"""

    def test_hyformer_registered(self):
        """测试 HyFormer 已注册。"""
        from recsys.core.registry import MODEL_REGISTRY

        assert "hyformer" in MODEL_REGISTRY
        assert MODEL_REGISTRY.get_metadata("hyformer")["family"] == "unified"
        assert MODEL_REGISTRY.get_metadata("hyformer")["task_type"] == "pointwise"
        assert MODEL_REGISTRY.get_metadata("hyformer")["supports_training"] is True


# ------------------------------------------------------------------
# 任务解析测试
# ------------------------------------------------------------------


def _make_mock_schema_no_seq():
    """构造无序列的 schema（用于无序列场景测试）。"""
    return {
        "user_int_feature_specs": [(10, 0, 1), (10, 1, 2)],
        "item_int_feature_specs": [(20, 0, 1)],
        "user_dense_dim": 0,
        "item_dense_dim": 0,
        "seq_vocab_sizes": {},
        "user_ns_groups": [[0], [1]],
        "item_ns_groups": [[0]],
        "seq_domains": [],
    }


class TestTaskParsing:
    """任务解析测试。"""

    def test_task_ctr_default(self):
        """默认 task=ctr → task_type=pointwise, problem_type=binary。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 16, "emb_dim": 8, "num_hyformer_blocks": 1,
                    "use_rope": False, "rank_mixer_mode": "ffn_only",
                    "ns_tokenizer_type": "group"},
            schema_metadata=_make_mock_schema(),
        )
        assert model.task_type == "pointwise"
        assert model.problem_type == "binary"
        assert model.action_num == 1

    def test_task_multiclass(self):
        """task=multiclass + num_classes=5 → action_num=5。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "multiclass", "num_classes": 5,
                    "d_model": 16, "emb_dim": 8, "num_hyformer_blocks": 1,
                    "use_rope": False, "rank_mixer_mode": "ffn_only",
                    "ns_tokenizer_type": "group"},
            schema_metadata=_make_mock_schema(),
        )
        assert model.task_type == "pointwise"
        assert model.problem_type == "multiclass"
        assert model.action_num == 5

    def test_task_regression(self):
        """task=regression → problem_type=regression。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "regression",
                    "d_model": 16, "emb_dim": 8, "num_hyformer_blocks": 1,
                    "use_rope": False, "rank_mixer_mode": "ffn_only",
                    "ns_tokenizer_type": "group"},
            schema_metadata=_make_mock_schema(),
        )
        assert model.task_type == "pointwise"
        assert model.problem_type == "regression"
        assert model.action_num == 1

    def test_task_multitask(self):
        """task=multitask + num_tasks=3 → task_type=multitask。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "multitask", "num_tasks": 3,
                    "d_model": 16, "emb_dim": 8, "num_hyformer_blocks": 1,
                    "use_rope": False, "rank_mixer_mode": "ffn_only",
                    "ns_tokenizer_type": "group"},
            schema_metadata=_make_mock_schema(),
        )
        assert model.task_type == "multitask"
        assert model.action_num == 3

    def test_task_ranking(self):
        """task=ranking → task_type=ranking。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "ranking",
                    "d_model": 16, "emb_dim": 8, "num_hyformer_blocks": 1,
                    "use_rope": False, "rank_mixer_mode": "ffn_only",
                    "ns_tokenizer_type": "group"},
            schema_metadata=_make_mock_schema(),
        )
        assert model.task_type == "ranking"
        assert model.problem_type == "implicit_ranking"
        assert model.action_num == 1

    def test_task_cvr_same_as_binary(self):
        """task=cvr 与 task=binary 等价。"""
        hyformer_cls = get_model("hyformer")
        m_bin = hyformer_cls(
            config={"task": "binary", "d_model": 16, "emb_dim": 8,
                    "num_hyformer_blocks": 1, "use_rope": False,
                    "rank_mixer_mode": "ffn_only", "ns_tokenizer_type": "group"},
            schema_metadata=_make_mock_schema(),
        )
        m_cvr = hyformer_cls(
            config={"task": "cvr", "d_model": 16, "emb_dim": 8,
                    "num_hyformer_blocks": 1, "use_rope": False,
                    "rank_mixer_mode": "ffn_only", "ns_tokenizer_type": "group"},
            schema_metadata=_make_mock_schema(),
        )
        assert m_cvr.task_type == m_bin.task_type
        assert m_cvr.problem_type == m_bin.problem_type
        assert m_cvr.action_num == m_bin.action_num

    def test_unknown_task_raises(self):
        """unknown task 别名抛 ValueError。"""
        hyformer_cls = get_model("hyformer")
        with pytest.raises(ValueError, match="未知 task 别名"):
            hyformer_cls(
                config={"task": "bogus"},
                schema_metadata=_make_mock_schema(),
            )


# ------------------------------------------------------------------
# 各任务前向传播测试
# ------------------------------------------------------------------


class TestTaskForward:
    """各任务类型前向传播测试。"""

    def test_binary_forward(self):
        """binary: scores (B,) + probs (B,)。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "binary", "d_model": 16, "emb_dim": 8,
                    "num_hyformer_blocks": 1, "use_rope": False,
                    "rank_mixer_mode": "ffn_only", "ns_tokenizer_type": "group",
                    "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema(),
        )
        batch = _make_batch(8)
        output = model(batch)
        assert output.scores is not None
        assert output.scores.shape == (8,)
        assert output.probs is not None
        assert output.probs.shape == (8,)

    def test_multiclass_forward(self):
        """multiclass: scores (B, C) + probs (B, C) + preds (B,)。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "multiclass", "num_classes": 3,
                    "d_model": 16, "emb_dim": 8, "num_hyformer_blocks": 1,
                    "use_rope": False, "rank_mixer_mode": "ffn_only",
                    "ns_tokenizer_type": "group", "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema(),
        )
        batch = _make_batch(8)
        output = model(batch)
        assert output.scores.shape == (8, 3)
        assert output.probs.shape == (8, 3)
        assert output.preds.shape == (8,)

    def test_regression_forward(self):
        """regression: scores (B,) + probs=None。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "regression", "d_model": 16, "emb_dim": 8,
                    "num_hyformer_blocks": 1, "use_rope": False,
                    "rank_mixer_mode": "ffn_only", "ns_tokenizer_type": "group",
                    "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema(),
        )
        batch = _make_batch(8)
        output = model(batch)
        assert output.scores.shape == (8,)
        assert output.probs is None

    def test_multitask_forward(self):
        """multitask: scores (B, T) + task_outputs 非空。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "multitask", "num_tasks": 2,
                    "d_model": 16, "emb_dim": 8, "num_hyformer_blocks": 1,
                    "use_rope": False, "rank_mixer_mode": "ffn_only",
                    "ns_tokenizer_type": "group", "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema(),
        )
        batch = _make_batch(8)
        output = model(batch)
        assert output.scores.shape == (8, 2)
        assert output.task_outputs is not None
        assert len(output.task_outputs) == 2

    def test_ranking_forward(self):
        """ranking: scores (B,) + probs (B,)。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "ranking", "d_model": 16, "emb_dim": 8,
                    "num_hyformer_blocks": 1, "use_rope": False,
                    "rank_mixer_mode": "ffn_only", "ns_tokenizer_type": "group",
                    "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema(),
        )
        batch = _make_batch(8)
        output = model(batch)
        assert output.scores.shape == (8,)
        assert output.probs.shape == (8,)


# ------------------------------------------------------------------
# 各任务损失计算测试
# ------------------------------------------------------------------


class TestTaskComputeLoss:
    """各任务类型损失计算测试。"""

    def test_binary_loss(self):
        """binary: BCEWithLogits 可 backprop。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "binary", "d_model": 16, "emb_dim": 8,
                    "num_hyformer_blocks": 1, "use_rope": False,
                    "rank_mixer_mode": "ffn_only", "ns_tokenizer_type": "group",
                    "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema(),
        )
        batch = _make_batch(4)
        output = model(batch)
        loss_dict = model.compute_loss(batch, output)
        assert "loss" in loss_dict
        loss_dict["loss"].backward()

    def test_multiclass_loss(self):
        """multiclass: CrossEntropy 可 backprop。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "multiclass", "num_classes": 3,
                    "d_model": 16, "emb_dim": 8, "num_hyformer_blocks": 1,
                    "use_rope": False, "rank_mixer_mode": "ffn_only",
                    "ns_tokenizer_type": "group", "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema(),
        )
        batch = _make_batch(4)
        batch.data["label"] = torch.randint(0, 3, (4,)).long()
        output = model(batch)
        loss_dict = model.compute_loss(batch, output)
        assert "loss" in loss_dict
        loss_dict["loss"].backward()

    def test_regression_loss(self):
        """regression: MSE 可 backprop。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "regression", "d_model": 16, "emb_dim": 8,
                    "num_hyformer_blocks": 1, "use_rope": False,
                    "rank_mixer_mode": "ffn_only", "ns_tokenizer_type": "group",
                    "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema(),
        )
        batch = _make_batch(4)
        output = model(batch)
        loss_dict = model.compute_loss(batch, output)
        assert "loss" in loss_dict
        loss_dict["loss"].backward()

    def test_multitask_loss(self):
        """multitask: 逐任务 BCE 可 backprop。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "multitask", "num_tasks": 2,
                    "d_model": 16, "emb_dim": 8, "num_hyformer_blocks": 1,
                    "use_rope": False, "rank_mixer_mode": "ffn_only",
                    "ns_tokenizer_type": "group", "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema(),
        )
        batch = _make_batch(4)
        batch.data["task_labels"] = torch.randint(0, 2, (4, 2)).float()
        batch.data.pop("label")
        output = model(batch)
        loss_dict = model.compute_loss(batch, output)
        assert "loss" in loss_dict
        loss_dict["loss"].backward()

    def test_ranking_loss(self):
        """ranking: BCEWithLogits 可 backprop。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"task": "ranking", "d_model": 16, "emb_dim": 8,
                    "num_hyformer_blocks": 1, "use_rope": False,
                    "rank_mixer_mode": "ffn_only", "ns_tokenizer_type": "group",
                    "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema(),
        )
        batch = _make_batch(4)
        output = model(batch)
        loss_dict = model.compute_loss(batch, output)
        assert "loss" in loss_dict
        loss_dict["loss"].backward()


# ------------------------------------------------------------------
# 无序列场景测试
# ------------------------------------------------------------------

def _make_batch_no_seq(batch_size: int = 8) -> Batch:
    """构造无序列的 batch。"""
    return Batch(data={
        "label": torch.randint(0, 2, (batch_size,)).float(),
        "user_int_feats": torch.randint(0, 9, (batch_size, 3), dtype=torch.long),
        "item_int_feats": torch.randint(0, 19, (batch_size, 1), dtype=torch.long),
    })


class TestNoSequenceMode:
    """无序列场景（num_sequences=0）测试。"""

    def test_no_seq_instantiation(self):
        """num_sequences=0 实例化成功。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 16, "emb_dim": 8, "num_hyformer_blocks": 1,
                    "use_rope": False, "rank_mixer_mode": "ffn_only",
                    "ns_tokenizer_type": "group", "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema_no_seq(),
        )
        assert model.core.num_sequences == 0
        assert model.core.query_generator is None
        assert model.core.blocks is None

    def test_no_seq_forward(self):
        """无 seq_data 前向成功。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 16, "emb_dim": 8, "num_hyformer_blocks": 1,
                    "use_rope": False, "rank_mixer_mode": "ffn_only",
                    "ns_tokenizer_type": "group", "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema_no_seq(),
        )
        batch = _make_batch_no_seq(8)
        output = model(batch)
        assert output.scores.shape == (8,)
        assert output.probs is not None

    def test_no_seq_gradient(self):
        """无序列模式梯度回传正常。"""
        hyformer_cls = get_model("hyformer")
        model = hyformer_cls(
            config={"d_model": 16, "emb_dim": 8, "num_hyformer_blocks": 1,
                    "use_rope": False, "rank_mixer_mode": "ffn_only",
                    "ns_tokenizer_type": "group", "dropout_rate": 0.0},
            schema_metadata=_make_mock_schema_no_seq(),
        )
        batch = _make_batch_no_seq(4)
        output = model(batch)
        loss_dict = model.compute_loss(batch, output)
        loss_dict["loss"].backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0
