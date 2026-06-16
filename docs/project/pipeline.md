---
title: Pipeline Guide
description: 当前单实验主干的真实实现、流程与边界
---

# Pipeline Guide

## 目标

RecBench 的 pipeline 层负责把配置、数据、模型、评估与 artifact 串成一条可复现的实验主干。

当前仓库里，这条主干已经不是设计草案，而是由 `src/recsys/pipeline/experiment.py` 实现的单实验入口。

## 当前主入口

当前单实验公共入口是：

```python
run_experiment(config: ExperimentConfig) -> ExperimentResult
```

这里的 `ExperimentConfig` 来自 `src/recsys/pipeline/experiment.py`，是 pipeline 层自己的运行配置对象，而不是 `utils.config.RecBenchConfig`。

## 当前已实现流程

`run_experiment()` 当前已经覆盖下面这些步骤：

1. 冻结配置并生成稳定 `run_id`
2. 创建实验目录并写初始 `status.json`
3. 写出 `config.yaml` 快照
4. 初始化运行时环境
5. 通过 registry 构建 dataset adapter 和 model
6. 按模型能力路由执行
7. 产出 `PredictionBundle`
8. 调用 evaluator
9. 写出指标、预测和曲线产物
10. 返回结构化 `ExperimentResult`

这条链路已经足以支撑最小闭环。

## 当前最小可运行路径

目前最稳定的路径是非训练式模型路径：

- 数据集加载
- `itemcf.fit()`
- `itemcf.predict()`
- ranking evaluator
- artifact 落盘

这也是当前文档与示例应优先围绕的主路径。

## 能力路由

RecBench 并不假设所有模型都要走梯度训练。

当前 pipeline 通过模型能力进行路由：

- 非训练模型：走 `_execute_nontrainable_path()`
- 可训练模型：原本预留 `_execute_trainable_path()`

### 非训练路径

当前已实现的非训练路径会：

- 从 `train` split 提取 `(user_id, item_id)` 交互对
- 调用模型 `fit()`
- 从 `train/test` split 提取用户历史与测试真值
- 调用模型 `predict()`
- 返回标准化 `PredictionBundle`

### 训练路径

当前训练路径仍未完成接线，代码会直接抛出 `NotImplementedError`。

因此，training 模块虽然已经具备 `LightningRecommender`、`TrainerFactory` 等基础设施，但 experiment 主流程还没有把训练型模型真正接通。

## 统一预测产物

无论模型来自哪条路径，pipeline 最终都要求拿到统一的 `PredictionBundle`，这样 evaluator 才能和模型解耦。

对当前仓库来说，这层统一产物已经是既有契约，不是未来规划。

常见字段包括：

- `task_type`
- `problem_type`
- `y_true`
- `y_score`
- `y_pred`
- `group_ids`
- `candidate_ids`
- `task_outputs`
- `task_labels`
- `metadata`

## Evaluator 调用

拿到 `PredictionBundle` 后，pipeline 会调用 `evaluate()`。

当前 evaluator 已经支持：

- pointwise 路由
- ranking 路由
- multitask 路由
- 结构化结果组装
- 曲线产物索引

因此 pipeline 自己不应实现指标逻辑，它只负责把预测产物交给评估层。

## 当前 artifact 写出

单实验当前已写出的关键产物包括：

- `config.yaml`
- `status.json`
- `metrics.json`
- `predictions.parquet`
- `curves/*.json`
- `logs/stderr.log`

如果后续训练型路径接通，`checkpoints/` 将成为更重要的常规产物；但就当前实现而言，它还不是单实验主路径的一部分。

## 推荐目录结构

当前文档应以如下结构说明单实验目录：

```text
outputs/experiments/{run_id}/
|-- config.yaml
|-- status.json
|-- metrics.json
|-- predictions.parquet
|-- curves/
`-- logs/
    `-- stderr.log
```

## 返回结果

`run_experiment()` 返回的是结构化对象，而不是扁平字典。

当前最重要的字段包括：

- `run_id`
- `status`
- `summary_metrics`
- `task_metrics`
- `artifact_paths`
- `error`
- `metadata`
- `warnings`

这让 benchmark 层可以稳定消费单实验结果，而不必了解训练或评估内部细节。

## 最小示例

```python
from recsys.pipeline.experiment import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    experiment_name="demo_itemcf",
    dataset_name="taac2026_data_sample",
    model_name="itemcf",
    seed=42,
    output_dir="./outputs/experiments",
    data_config={"root_dir": "./data"},
    evaluation_config={
        "primary_metric": "ndcg@10",
        "ranking_k": [10],
        "generate_curves": False,
    },
)

result = run_experiment(cfg)
print(result.status)
print(result.summary_metrics)
print(result.artifact_paths)
```

## 错误处理

当前单实验主干已经有明确阶段边界，失败时会收敛到结构化 `ExperimentError`。

文档中应把错误阶段理解为以下几类：

- 配置阶段
- 数据阶段
- 模型阶段
- 训练阶段
- 评估阶段
- artifact 阶段

这比简单宣传“统一抛异常”更符合当前实现。

## 与 training 的边界

当前边界应描述为：

- pipeline 决定什么时候需要 trainer
- training 决定如何训练神经模型
- training 不负责 experiment 目录结构和 artifact 契约
- pipeline 不负责 optimizer、scheduler 和 callback 细节

## 与 benchmark 的边界

当前边界应描述为：

- `run_experiment()` 负责一个配置组合
- `run_benchmark()` 负责多个配置组合的展开和调度
- `Reporter` 负责聚合结果，而不是执行实验

这三个边界在代码里已经拆开，文档中也应保持一致。

## 当前限制

当前文档必须显式保留下面这些限制：

- 可训练路径尚未接通
- 当前真正清晰可运行的模型主要是 `itemcf`
- CLI 入口仍未完成
- benchmark 预设与 experiment 预设的完整解析能力尚未全部打通

## 当前最重要的结论

RecBench 的 pipeline 当前已经具备“单实验主干 + 标准预测产物 + 结构化 artifact”的核心能力。文档应把它写成“已实现的基础运行时”，同时明确可训练模型路径仍是当前最主要的未完成项。
