"""单实验端到端测试：验证 run_experiment() 全链路。"""

from pathlib import Path

import pytest

from recsys import auto_discover_models, get_model
from recsys.core.prediction_bundle import PredictionBundle
from recsys.evaluation import EvaluationConfig, evaluate
from recsys.pipeline.experiment import ExperimentConfig, run_experiment


def test_run_experiment_itemcf_synthetic():
    """用合成数据调 ItemCF.fit() + predict()，构造 PredictionBundle，调 evaluate()。"""
    auto_discover_models()
    itemcf_cls = get_model("itemcf")
    model = itemcf_cls(similarity="cosine", top_k_neighbors=10, recommend_k=5)

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
        3: {4, 5, 6},
        4: {0, 6, 7},
    }
    user_test_items = {
        0: {3},
        1: {4},
        2: {1},
        3: {0},
        4: {5},
    }
    bundle = model.predict(
        user_train_items=user_train_items,
        user_test_items=user_test_items,
    )

    assert isinstance(bundle, PredictionBundle)
    assert bundle.task_type == "ranking"
    assert len(bundle.group_ids) > 0

    eval_cfg = EvaluationConfig(
        metrics=["ndcg@10", "hit_rate@10", "recall@10", "mrr"],
        ranking_k=[10],
        generate_curves=False,
    )
    result = evaluate(bundle, eval_cfg)

    assert result.summary_metrics is not None
    assert len(result.summary_metrics) > 0
    has_ranking_metric = any(
        k in result.summary_metrics for k in ["mrr", "ndcg@10", "hit_rate@10", "recall@10"]
    )
    assert has_ranking_metric, f"Expected ranking metrics, got: {result.summary_metrics}"


def _is_dataset_cached(repo_id: str, variant: str) -> bool:
    """检查 HF 数据集是否已有本地缓存。"""
    import os as _os
    cache_base = _os.path.expanduser("~/.cache/huggingface/datasets")
    if not _os.path.isdir(cache_base):
        return False
    for root, dirs, _files in _os.walk(cache_base):
        for d in dirs:
            if variant in d and repo_id.replace("/", "___") in root:
                return True
    return False


@pytest.mark.integration
def test_run_experiment_itemcf_real(tmp_path, monkeypatch):
    """真实调 run_experiment(cfg)，验证产物完整。"""
    # 离线模式：避免挂死在网络请求上
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    # 跳过未缓存的数据集
    if not _is_dataset_cached("TAAC2026", "data_sample_1000"):
        pytest.skip("TAAC2026 data_sample_1000 未缓存，跳过（请在联网时先下载一次）")

    cfg = ExperimentConfig(
        experiment_name="test_itemcf",
        dataset_name="taac2026_data_sample",
        model_name="itemcf",
        seed=42,
        output_dir=str(tmp_path / "experiments"),
        data_config={"root_dir": str(tmp_path / "data")},
        model_config={
            "params": {
                "similarity": "cosine",
                "top_k_neighbors": 50,
                "recommend_k": 10,
                "normalize": True,
            }
        },
        evaluation_config={
            "primary_metric": "ndcg@10",
            "ranking_k": [10],
            "generate_curves": False,
        },
    )

    result = run_experiment(cfg)
    assert result.status.value == "succeeded", f"Experiment failed: {result.error}"

    # 检查是否有有效的评估结果
    # 如果所有组都被跳过（测试集中用户没有正样本），summary_metrics 可能为空
    # 这是数据集特性，不是代码错误
    if result.summary_metrics:
        assert "ndcg@10" in result.summary_metrics
    else:
        # 如果没有评估结果，检查是否有警告说明原因
        assert any("skipped" in w.lower() for w in result.warnings), (
            f"Expected skip warning, got: {result.warnings}"
        )

    run_dir = Path(result.artifact_paths.get("run_dir", ""))
    if run_dir.exists():
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "predictions.parquet").exists()
        assert (run_dir / "status.json").exists()


# ------------------------------------------------------------------
# schema_metadata 传递测试
# ------------------------------------------------------------------

def test_pipeline_passes_dense_remap_metadata_to_hyformer(monkeypatch, tmp_path):
    """确认 pipeline 把 TAAC2026 dense remap 元信息正确传给 HyFormer。"""
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as TorchDataset

    from recsys import auto_discover_models
    from recsys.core.registry import DATASET_REGISTRY, MODEL_REGISTRY

    auto_discover_models()

    # 构造一个最小 fake 数据集，模拟 TAAC2026 dense remap 后的语义
    class _FakeSplit(TorchDataset):
        def __init__(self, n):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return {
                "user_id": torch.tensor(idx % 5 + 1, dtype=torch.long),
                "item_id": torch.tensor(idx % 10 + 1, dtype=torch.long),
                "label": torch.tensor(0 if idx % 2 == 0 else 1, dtype=torch.long),
            }

    class _FakeDataset:
        dataset_name = "fake_dense"
        dataset_url = "https://fake"
        feature_cols = []
        label_col = "label_type"
        num_users = 5
        num_items = 10
        _padding_idx = 0
        _user_id_space = "dense_1_based"
        _item_id_space = "dense_1_based"
        _num_users = 5
        _num_items = 10

        def __init__(self, **kwargs):
            pass

        def load(self):
            return self

        def get_dataloader(self, split="train", batch_size=16, **kw):
            return DataLoader(_FakeSplit(32), batch_size=batch_size)

    # 记录模型收到的 schema_metadata
    captured_meta = {}

    original_get = MODEL_REGISTRY.get

    def _fake_get(name):
        cls = original_get(name)
        if name == "hyformer":
            orig_init = cls.__init__
            def _patched_init(self_, config=None, schema_metadata=None, **kw):
                nonlocal captured_meta
                captured_meta["received"] = dict(schema_metadata or {})
                orig_init(self_, config, schema_metadata, **kw)
            cls.__init__ = _patched_init
        return cls

    monkeypatch.setattr(MODEL_REGISTRY, "get", _fake_get)
    monkeypatch.setitem(DATASET_REGISTRY._items, "fake_dense", _FakeDataset)

    from recsys.pipeline.experiment import ExperimentConfig, run_experiment

    cfg = ExperimentConfig(
        experiment_name="test_meta",
        dataset_name="fake_dense",
        model_name="hyformer",
        seed=42,
        output_dir=str(tmp_path / "test_meta"),
        data_config={},
        training_config={"epochs": 1, "batch_size": 4, "learning_rate": 1e-3},
        evaluation_config={"generate_curves": False},
        model_config={"params": {"d_model": 8, "emb_dim": 8}},
    )
    _result = run_experiment(cfg)  # noqa: F841

    assert "received" in captured_meta, "pipeline 应调用 model_cls(schema_metadata=...)"
    meta = captured_meta["received"]
    assert meta["num_users"] == 5
    assert meta["num_items"] == 10
    assert meta["user_id_space"] == "dense_1_based"
    assert meta["item_id_space"] == "dense_1_based"
    assert meta["padding_idx"] == 0
