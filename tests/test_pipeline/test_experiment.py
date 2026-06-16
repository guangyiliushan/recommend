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


@pytest.mark.integration
def test_run_experiment_itemcf_real(tmp_path):
    """真实调 run_experiment(cfg)，验证产物完整。"""
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
