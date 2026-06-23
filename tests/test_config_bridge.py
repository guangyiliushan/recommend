"""配置桥接函数 recbench_to_experiment_config 单元测试。

验证 RecBenchConfig → pipeline ExperimentConfig 的字段映射完整性。
"""

from recsys.utils.config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    RecBenchConfig,
    RuntimeConfig,
    TrainingConfig,
    recbench_to_experiment_config,
)


def _make_base_config() -> RecBenchConfig:
    """构造最小化 RecBenchConfig。"""
    return RecBenchConfig(
        experiment=ExperimentConfig(name="test_exp"),
        data=DataConfig(
            name="taac2026_data_sample",
            data_dir="./data",
            split_mode="temporal",
            min_action_type=1,
        ),
        model=ModelConfig(
            name="itemcf",
            family="classical",
            task_type="ranking",
            problem_type="implicit_ranking",
            params={"similarity": "cosine", "top_k_neighbors": 50},
        ),
        training=TrainingConfig(
            epochs=5,
            learning_rate=0.01,
            optimizer="adam",
        ),
        evaluation=EvaluationConfig(
            metrics=["ndcg@10", "recall@10"],
            ranking_k=[5, 10],
            generate_curves=False,
        ),
        runtime=RuntimeConfig(
            device="cpu",
            seed=43,
            log_level="DEBUG",
            output_root="./outputs",
        ),
    )


def test_recbench_to_experiment_config_basic():
    """基本字段映射：name / seed / output_dir 等。"""
    cfg = _make_base_config()
    exp = recbench_to_experiment_config(cfg)

    assert exp.experiment_name == "test_exp"
    assert exp.dataset_name == "taac2026_data_sample"
    assert exp.model_name == "itemcf"
    assert exp.seed == 43
    assert exp.output_dir == "./outputs/runs"


def test_recbench_to_experiment_config_split_mode():
    """split_mode 正确传递。"""
    cfg = _make_base_config()
    exp = recbench_to_experiment_config(cfg)

    assert exp.data_config["split_mode"] == "temporal"
    assert exp.data_config["min_action_type"] == 1
    assert exp.data_config["root_dir"] == "./data"


def test_recbench_to_experiment_config_metrics():
    """evaluation metrics 和 ranking_k 列表正确转换。"""
    cfg = _make_base_config()
    exp = recbench_to_experiment_config(cfg)

    assert exp.evaluation_config["metrics"] == ["ndcg@10", "recall@10"]
    assert exp.evaluation_config["ranking_k"] == [5, 10]
    assert exp.evaluation_config["generate_curves"] is False


def test_recbench_to_experiment_config_training():
    """training 超参完整传递。"""
    cfg = _make_base_config()
    exp = recbench_to_experiment_config(cfg)

    assert exp.training_config["epochs"] == 5
    assert exp.training_config["learning_rate"] == 0.01
    assert exp.training_config["optimizer"] == "adam"


def test_recbench_to_experiment_config_model_params():
    """model params 字典正确传递。"""
    cfg = _make_base_config()
    exp = recbench_to_experiment_config(cfg)

    assert exp.model_config["params"]["similarity"] == "cosine"
    assert exp.model_config["params"]["top_k_neighbors"] == 50


def test_recbench_to_experiment_config_runtime():
    """runtime 运行时字段映射。"""
    cfg = _make_base_config()
    exp = recbench_to_experiment_config(cfg)

    assert exp.runtime_config["device"] == "cpu"
    assert exp.runtime_config["seed"] == 43
    assert exp.runtime_config["log_level"] == "DEBUG"


def test_recbench_to_experiment_config_empty_params():
    """空模型参数的边界情况。"""
    cfg = _make_base_config()
    cfg.model.params = {}
    exp = recbench_to_experiment_config(cfg)

    assert exp.model_config["params"] == {}


def test_recbench_to_experiment_config_default_split_mode():
    """DataConfig split_mode 默认值是 temporal。"""
    d = DataConfig()
    assert d.split_mode == "temporal"
