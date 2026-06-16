---
title: Getting Started
description: 本地环境、常用命令与当前推荐入门路径
---

# Getting Started

## 环境约定

本仓库统一使用 `uv` 管理 Python 环境与依赖，以保持与 CI 和文档示例一致。

## 安装依赖

```bash
uv sync --extra dev
```

这一步会安装开发、测试与文档构建所需依赖。

## 常用命令

### 静态检查

```bash
uv run ruff check .
```

### 测试

```bash
uv run pytest -v
```

说明：当前测试覆盖仍在逐步建设中，测试通过不等于所有模块都已经完整落地。

### 构建文档站

```bash
uv run zensical build --strict --clean
```

## 当前推荐的阅读顺序

如果你想先理解项目，建议依次阅读：

1. `README.md`
2. `docs/index.md`
3. `docs/concepts/architecture.md`
4. `docs/concepts/configuration.md`
5. `docs/project/pipeline.md`
6. `docs/project/evaluation.md`

## 当前推荐的代码入口

如果你要从代码快速建立整体认知，建议优先看：

- `src/recsys/core/registry.py`
- `src/recsys/core/base_model.py`
- `src/recsys/core/prediction_bundle.py`
- `src/recsys/pipeline/experiment.py`
- `src/recsys/pipeline/benchmark.py`
- `src/recsys/evaluation/evaluator.py`

## 当前最适合验证的路径

当前仓库里最适合作为最小闭环理解入口的是：

1. 自动发现模型
2. 获取 `itemcf`
3. 理解 `run_experiment()` 的非训练路径
4. 理解 `evaluate()` 如何消费 `PredictionBundle`
5. 理解 `run_benchmark()` 与 `Reporter` 如何聚合结果

## Python API 快速示例

### 模型发现

```python
from recsys import auto_discover_models, get_model, list_models

auto_discover_models()
print(list_models())

ItemCF = get_model("itemcf")
model = ItemCF(similarity="cosine")
```

### 单实验入口

```python
from recsys.pipeline.experiment import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    experiment_name="demo_itemcf",
    dataset_name="taac2026_data_sample",
    model_name="itemcf",
    seed=42,
    output_dir="./outputs/experiments",
)

result = run_experiment(cfg)
print(result.status)
```

### 训练型模型入口

```python
from recsys.pipeline.experiment import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    experiment_name="demo_dssm",
    dataset_name="taac2026_data_sample",
    model_name="dssm",
    seed=42,
    output_dir="./outputs/experiments",
    data_config={"root_dir": "./data"},
    model_config={
        "params": {"embed_dim": 64, "hidden_dims": [128, 64]},
    },
    training_config={
        "epochs": 10,
        "batch_size": 256,
        "learning_rate": 1e-3,
    },
    evaluation_config={
        "metrics": ["roc_auc", "log_loss"],
    },
)

result = run_experiment(cfg)
print(result.status)
```

### CLI 入口

```bash
# 非训练模型
uv run python scripts/run_single.py --model itemcf --dataset taac2026_data_sample --seed 42

# 训练模型
uv run python scripts/run_single.py --model dssm --dataset taac2026_data_sample --seed 42 --epochs 10 --lr 3e-4

# 批量 Benchmark
uv run python scripts/run_benchmark.py --config configs/experiment/benchmark_classical.yaml
```

## 当前需要特别注意的现实边界

- 大多数模型目录文件仍是占位，当前最清晰可运行的模型是 `itemcf`（非训练）和 `dssm`（训练）
- `scripts/run_ablation.py`、`scripts/download_data.py` 等辅助脚本仍待完善

## 仓库重要目录

- `src/recsys`：核心源码
- `configs`：配置与实验矩阵
- `docs`：文档站源码
- `tests`：测试
- `outputs`：实验与 Benchmark 结果目录
- `.github/workflows`：自动化工作流
