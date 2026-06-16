"""最小单实验 demo：itemcf + taac2026_data_sample。

运行方式：
    uv run python examples/run_itemcf_demo.py

前置条件：
    - uv sync --extra dev 已执行
    - 网络可用（数据集从 HuggingFace Hub 下载并缓存到 ./data）
"""

import recsys.data.dataset_registry  # noqa: F401 触发数据集注册
from recsys import auto_discover_models, list_models
from recsys.core.registry import DATASET_REGISTRY
from recsys.pipeline.experiment import ExperimentConfig, run_experiment

# 1. 触发注册
auto_discover_models()

print("models:", list_models())
print("datasets:", DATASET_REGISTRY.list())

# 2. 构造配置并运行
cfg = ExperimentConfig(
    experiment_name="demo_itemcf",
    dataset_name="taac2026_data_sample",
    model_name="itemcf",
    seed=42,
    output_dir="./outputs/experiments",
    data_config={"root_dir": "./data"},
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
print(f"status: {result.status}")
print(f"metrics: {result.summary_metrics}")
print(f"artifacts: {result.artifact_paths}")
