---
title: Public API
description: 当前真实存在的公共 Python API、返回结构与边界说明
---

# Public API

## 目标

RecBench 当前没有 HTTP 服务型接口，因此"公共 API"主要指：

- 顶层 Python 包导出
- 注册表与基础契约
- 运行时主干 `run_experiment()` / `run_benchmark()`
- evaluator 与 training 的公共工厂接口

`scripts/` 下的 CLI 入口目前仍是骨架，不属于稳定公共 API。

## 顶层包导出

`src/recsys/__init__.py` 当前对外导出：

- 版本号 `__version__`
- 四个注册表：`MODEL_REGISTRY`、`DATASET_REGISTRY`、`METRIC_REGISTRY`、`LOSS_REGISTRY`
- 模型契约对象：`BaseRecommender`、`NeuralRecommender`、`Batch`、`ModelOutput`、`Capability`
- 预测产物：`PredictionBundle`
- 模型发现入口：`auto_discover_models()`、`get_model()`、`get_model_metadata()`、`list_models()`、`list_models_by_family()`、`list_models_by_task_type()`

示例：

```python
from recsys import auto_discover_models, get_model, list_models

auto_discover_models()
print(list_models())

ItemCF = get_model("itemcf")
model = ItemCF(similarity="cosine")
```

## 错误语义

当前项目的错误模型以 Python 异常和结构化错误对象为主，而不是 HTTP 状态码。

### 当前已经存在的错误类型

- `ConfigError`
- `DeviceError`
- `LoggingError`
- `ReproducibilityError`
- `ProfilingError`
- `ModelContractError`
- `ExperimentError`

### 当前结构化错误对象

`run_experiment()` 和 `run_benchmark()` 使用结构化错误信息时，重点字段包括：

- `code`
- `phase`
- `message`
- `details`

文档不应假设所有模块都已经统一成完全一致的错误码体系，但可以把上述字段视为当前主干的收敛方向。

## Registry API

## `Registry`

`src/recsys/core/registry.py` 提供统一注册表能力，用于模型、数据集、指标和 loss 的命名发现。

### 主要能力

- `register(name, **metadata)`
- `get(name)`
- `get_metadata(name)`
- `list()`
- `list_by(key, value)`
- `auto_discover(package_path)`

### 示例

```python
from recsys.core.registry import MODEL_REGISTRY

ModelCls = MODEL_REGISTRY.get("itemcf")
meta = MODEL_REGISTRY.get_metadata("itemcf")
print(meta["task_type"])
```

## 模型发现 API

`src/recsys/models/model_registry.py` 提供比直接访问 `MODEL_REGISTRY` 更面向用户的辅助函数。

### 当前公开函数

- `auto_discover_models()`
- `get_model(name)`
- `get_model_metadata(name)`
- `list_models()`
- `list_models_by_family(family)`
- `list_models_by_task_type(task_type)`
- `list_trainable_models()`
- `list_non_trainable_models()`

### 重要边界

- `auto_discover_models()` 负责导入模块并触发注册，不等于"这些模型都已经完成实现"
- 当前文档应把"已发现的模块"与"真正可运行的模型"区分开
- 当前可运行模型：`itemcf`（非训练）、`hyformer`（训练）

## Dataset API

## `BaseDataset`

`src/recsys/core/base_dataset.py` 定义 dataset adapter 的统一生命周期。

### 关键方法

- `load()`
- `get_split(split)`
- `get_dataloader(split, batch_size, num_workers, shuffle, **kwargs)`

### 当前使用方式

```python
from recsys.data.datasets.taac2026 import TAAC2026SecondRound

dataset = TAAC2026SecondRound(root_dir="./data", split_mode="temporal").load()
train_split = dataset.get_split("train")
train_loader = dataset.get_dataloader("train", batch_size=256, num_workers=4)
```

### 数据集元信息

数据集实例暴露以下公共元信息，供 pipeline 在构模时采集：

- `num_users` / `num_items`：dense remap 后唯一用户/物品数
- `feature_cols`：特征列名列表
- `label_col`：标签列名
- `_padding_idx`：padding 槽位索引（默认 0）
- `_user_id_space` / `_item_id_space`：ID 空间类型（`"dense_1_based"` 表示 1-based 连续，`"raw"` 表示未 remap）

### 当前边界

- 数据集需要先 `load()`，再访问 split
- split 名统一围绕 `train / val / test / full`
- dataset adapter 是当前真实存在的公共能力，不是占位设计
- TAAC 2026 数据集已内置 dense ID remap，`num_users` / `num_items` 反映的是 remap 后的连续索引数量

## 配置 API

`src/recsys/utils/config.py` 提供结构化配置加载与校验能力。

### 主要对象

- `RecBenchConfig`
- `ExperimentConfig`
- `DataConfig`
- `ModelConfig`
- `TrainingConfig`
- `EvaluationConfig`
- `RuntimeConfig`

### 主要函数

- `load_config(config_path=None, overrides=None)`
- `validate_config(config)`
- `get_config_snapshot(config)`
- `recbench_to_experiment_config(config)`：Hydra 层到 pipeline 层的桥接

### 示例

```python
from recsys.utils.config import load_config

cfg = load_config("configs/config.yaml")
print(cfg.model.name)
print(cfg.runtime.device)
print(cfg.data.split_mode)
```

## Evaluation API

`src/recsys/evaluation` 当前已经提供可直接使用的公共评估接口。

### evaluator 主入口

- `evaluate(bundle, config=None)`
- `evaluate_pointwise(bundle, config)`
- `evaluate_ranking(bundle, config)`
- `evaluate_multitask(bundle, config)`

### metrics / ranking / visualization 导出

- 分类指标函数位于 `metrics.py`
- 排序指标函数位于 `ranking.py`
- 曲线导出与可选绘图位于 `visualization.py`

### 输入与返回

输入统一以 `PredictionBundle` 为主。

返回 `EvaluationResult`，重点字段包括：

- `summary_metrics`
- `task_metrics`
- `group_metrics`
- `curve_artifacts`
- `metadata`
- `warnings`
- `errors`

示例：

```python
from recsys.evaluation import EvaluationConfig, evaluate

result = evaluate(bundle, EvaluationConfig(metrics=["ndcg@10", "mrr"]))
print(result.summary_metrics)
```

## Training API

`src/recsys/training` 当前已经具备可训练模型所需的基础设施。

### 训练主入口

- `LightningRecommender`
- `TrainerFactory`
- `create_trainer()`

### 辅助工厂

- `build_callbacks()`
- `build_optimizer()`
- `build_scheduler()`
- `resolve_strategy()`
- `get_loss()`

### 当前边界

- 训练基础设施已实现
- `run_experiment()` 中的训练型路径已通过 HyFormer 验证接通
- 它们是"公共训练 API"，已经通过单实验主干打通了完整运行路径

## Experiment Runtime API

## `run_experiment()`

`src/recsys/pipeline/experiment.py` 当前已经实现单实验主干。

### 当前签名

```python
run_experiment(config: ExperimentConfig) -> ExperimentResult
```

这里的 `ExperimentConfig` 是 pipeline 层自己的运行配置对象，不是 `utils.config.RecBenchConfig`。

### 当前已实现行为

- 冻结配置并生成 `run_id`
- 初始化输出目录和 `status.json`
- 实例化数据集与模型
- 按模型能力路由执行非训练式路径或训练式路径
- 调用 evaluator
- 写出 `metrics.json`、`predictions.parquet`、`curves/`、`checkpoints/`
- 返回结构化 `ExperimentResult`

### 返回对象重点字段

- `run_id`
- `status`
- `summary_metrics`
- `task_metrics`
- `artifact_paths`
- `error`
- `metadata`
- `warnings`

### 示例

```python
from recsys.pipeline.experiment import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    experiment_name="demo_itemcf",
    dataset_name="taac2026_data_sample",
    model_name="itemcf",
    seed=42,
    output_dir="./outputs/experiments",
    data_config={"root_dir": "./data", "split_mode": "temporal"},
)

result = run_experiment(cfg)
print(result.status)
print(result.summary_metrics)
```

## Benchmark API

## `run_benchmark()`

`src/recsys/pipeline/benchmark.py` 当前已经实现批量调度与恢复策略。

### 当前签名

```python
run_benchmark(bench_cfg: BenchmarkConfig) -> BenchmarkResult
```

### 当前已实现行为

- 展开 `models × datasets × seeds`
- 生成 `RunPlan`
- 支持 `successful_skip / failed_only / unfinished_only / force`
- 串行或受控并发执行实验
- 调用 `Reporter.generate()` 聚合结果
- 写出 `manifest.json`

### 返回对象重点字段

- `benchmark_name`
- `status`
- `runs`
- `summary_path`
- `leaderboard_path`
- `failures_path`
- `manifest_path`
- `report_path`
- `metadata`

### 示例

```python
from recsys.pipeline.benchmark import BenchmarkConfig, ResumeMode, run_benchmark

bench_cfg = BenchmarkConfig(
    benchmark_name="demo_benchmark",
    models=["itemcf"],
    datasets=["taac2026_data_sample"],
    seeds=[42, 43],
    resume_mode=ResumeMode.SUCCESSFUL_SKIP,
)

result = run_benchmark(bench_cfg)
print(result.status)
print(result.summary_path)
```

## Reporter API

`src/recsys/pipeline/reporter.py` 当前负责聚合单实验结果并输出可扫描文件。

### 当前产物

- `summary.csv`
- `leaderboard.csv`
- `failures.csv`
- `trend.csv`
- `stability.csv`
- `report.html`

### 当前入口

```python
Reporter(config).generate(results)
```

## CLI 边界

`scripts/run_single.py` 和 `scripts/run_benchmark.py` 当前已实现为基础可用的 CLI 入口。
`scripts/generate_report.py`、`scripts/run_ablation.py`、`scripts/download_data.py` 仍为引导提示脚本。

因此：

- 文档不应把它们写成生产级 CLI
- 示例可以给出 `run_single.py` / `run_benchmark.py` 的命令行用法
- 当前更适合推荐 Python API 调用方式作为主要交互途径

## 返回格式原则

当前公共运行时 API 基本遵守以下原则：

- 返回结构化对象，而不是裸字典或裸字符串
- 主结果与错误信息分离
- artifact 路径显式暴露
- 摘要指标集中在 `summary_metrics`

## 当前最重要的边界结论

- 顶层包导出、模型发现、配置、评估、训练基础设施、单实验与批量 Benchmark API 都已存在
- "公共 API 已存在"意味着对应能力已经接通并可验证（训练路径已通过 hyformer 验证）
- 当前应把 Python API 视为正式入口，把 CLI 视为未来补齐项
