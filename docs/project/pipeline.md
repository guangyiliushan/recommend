---
title: Pipeline Guide
description: 一次实验从配置、数据、模型、训练、评估到 artifact 的全流程
---

# Pipeline Guide

## 目标

RecBench 的 pipeline 层是整个项目的中枢。它的目标不是新增一层“胶水代码”，而是把下面这些稳定边界串起来：

- 配置加载
- 数据集实例化
- 模型实例化
- 训练执行或旁路拟合
- 评估执行
- artifact 落盘

当前仓库已经有以下关键骨架：

- `src/recsys/pipeline/experiment.py`
- `src/recsys/training/trainer.py`
- `scripts/run_single.py`

其中 `experiment.py` 的注释已经给出单实验的核心步骤，因此这份文档重点说明“应该如何把这些步骤设计成一个可维护的单实验主干”。

## Pipeline 为什么重要

如果没有稳定 pipeline，后面每加一个模型或数据集都会遇到这些问题：

- 入口命令不统一
- 配置覆盖方式不一致
- 结果目录不一致
- 评估和训练无法复用
- 出错时不知道问题落在配置、数据、模型还是训练阶段

因此 pipeline 不是附属模块，而是整个 benchmark 平台的主干。

## 当前目标边界

对当前项目来说，单次实验 pipeline 至少应负责：

1. 解析并冻结配置
2. 设置随机种子与 runtime
3. 构建 dataset adapter
4. 构建 model
5. 判断是否需要 trainer
6. 执行训练或非训练式 fit
7. 生成预测结果
8. 执行 evaluator
9. 写出 artifact
10. 返回结构化结果

## 推荐的单实验流程

### Step 1. 解析配置

第一步应始终是把配置解析为一份完整、可落盘的 experiment config。

推荐这里完成：

- Hydra 组合
- dataclass schema 校验
- 路径规范化
- 默认值展开

这一步的输出应是一份 fully resolved config，而不是仍带大量隐式默认值的半成品对象。

## 为什么要先冻结配置

因为配置是一次实验的“真相源”。

如果不先冻结，后面会出现：

- 运行中途修改默认值导致不可复现
- 恢复任务时拿不到一致配置
- benchmark 汇总时无法比较不同 run

因此单实验目录里必须落一份最终配置快照。

### Step 2. 初始化运行环境

在正式构建数据和模型前，pipeline 应负责统一设置：

- `seed`
- `device`
- `deterministic`
- `log_level`
- `output_dir`

这一步应发生在最前面，避免不同组件各自偷偷设置环境。

## Step 3. 构建 Dataset Adapter

这一步应该：

- 从 registry 或配置中解析数据集名称
- 实例化对应 dataset adapter
- 调用 `load()`
- 获取 `train/val/test` split
- 记录 dataset metadata

这里不应该：

- 做模型专属 batch hack
- 做训练器相关副作用初始化

因为这些职责分别属于 model contract 与 trainer。

## Step 4. 校验数据与任务类型

在模型构建前，pipeline 最好先做一层轻量语义校验，例如：

- 数据集是否支持当前任务类型
- 必需字段是否存在
- split 是否齐全
- ranking 任务是否有分组语义
- multitask 是否有对应标签

这一步很重要，因为越晚发现问题，报错越难定位。

## Step 5. 构建 Model

模型构建应基于：

- 模型 registry
- 模型配置
- 数据集暴露的必要元信息

推荐只向模型传入：

- 配置对象
- 必需的 schema / metadata

不推荐直接把整个 dataset adapter 对象无边界注入模型，否则后续耦合会迅速扩大。

## Step 6. 判断训练路径

这个项目的一个关键特征是，不是所有模型都走梯度训练。

至少存在两类路径：

- 可训练模型：走 trainer 主路
- 非训练式模型：走直接 `fit/predict` 或纯推理路径

例如：

- `ItemCF` 这类经典方法不应强行包装成完整的 Lightning 训练循环
- `DeepFM` 这类神经模型则更适合进入 trainer

因此 pipeline 必须先判断模型能力，而不是默认所有模型都需要同一训练框架。

## 推荐的能力判断方式

更推荐基于 model contract 或 registry metadata 判断：

- `supports_training`
- `supports_validation`
- `supports_checkpointing`

而不是依赖文件名或类名猜测。

## Step 7. 执行训练或旁路 fit

### 可训练模型路径

如果模型支持训练，pipeline 应把它交给 trainer 处理：

- 构造 Lightning adapter
- 配置 callbacks / loggers
- 调用 `fit`
- 视情况调用 `validate` / `test`

### 非训练式模型路径

如果模型不需要梯度训练，pipeline 应提供旁路：

- 用训练 split 做 `fit`
- 用 test split 做 `predict`
- 生成标准化预测结果

这一步的关键不是具体框架，而是两条路径最后都必须产出统一预测契约。

## Step 8. 统一预测产物

这是 pipeline 成败的关键节点。

无论模型来自哪一种训练路径，最终都应产出统一结构，例如：

- `task_type`
- `y_true`
- `y_score`
- `y_pred`
- `group_ids`
- `candidate_ids`
- `task_outputs`
- `task_labels`
- `metadata`

这就是前面 `evaluation.md` 中提到的 `PredictionBundle`。

如果没有这层统一产物，后面 evaluator 只能和每个模型逐个适配。

## Step 9. 执行 Evaluator

在拿到统一预测产物后，pipeline 应：

- 根据 `task_type` 选择 pointwise、ranking 或 multitask evaluator
- 生成结构化指标结果
- 生成必要曲线数据
- 返回用于 benchmark 汇总的 summary metrics

这里的重点是：

- pipeline 负责调用 evaluator
- evaluator 负责计算指标

不要让 pipeline 自己实现指标逻辑。

## Step 10. 落盘 Artifact

一次实验至少应落盘以下 artifact：

- `config.yaml`
- `status.json`
- `metrics.json`
- `logs/`

如果任务支持，建议进一步保存：

- `predictions.parquet`
- `curves/`
- `checkpoints/`
- `model_summary.txt`

这些 artifact 不只是为了看结果，更是为了：

- 恢复运行
- 后续 benchmark 聚合
- 结果审计

## 推荐的单实验目录结构

建议长期统一为：

```text
outputs/experiments/{run_id}/
|-- config.yaml
|-- status.json
|-- metrics.json
|-- predictions.parquet
|-- logs/
|-- checkpoints/
`-- curves/
```

## Step 11. 返回结构化结果

`run_experiment()` 的返回值不应只是一个 `Dict[str, float]`。

更推荐返回结构化 experiment result，至少包含：

- `run_id`
- `status`
- `summary_metrics`
- `artifact_paths`
- `error`

这样 benchmark 层才能稳定消费。

## Pipeline 与 Trainer 的边界

当前 `trainer.py` 的骨架已经说明它希望成为：

- LightningRecommender
- TrainerFactory

这是合理的，但要守住边界：

- pipeline 决定什么时候需要 trainer
- trainer 决定如何训练
- trainer 不决定 benchmark 调度
- trainer 不决定 experiment 目录结构

如果 trainer 承担太多职责，后面所有非标准模型都会变得难接入。

## Pipeline 与 Dataset 的边界

pipeline 应依赖 dataset adapter 的稳定接口，例如：

- `load()`
- `get_split()`
- `get_dataloader()`
- metadata

但不应依赖某个具体数据集的内部类细节。

例如，pipeline 不应写：

- “如果是 TAAC2025 就自己拼一个特殊序列字段”

这种特判应该通过 batch schema 或 dataset contract 解决。

## Pipeline 与 Model 的边界

pipeline 只应知道：

- 如何通过 registry 找到模型
- 模型是否需要 trainer
- 模型如何返回标准预测结果

它不应知道：

- 某个模型内部 loss 怎么算
- 某个模型 embedding 如何更新
- 某个模型有哪些私有训练技巧

## Pipeline 与 Benchmark 的边界

这两个边界必须明确：

- pipeline 负责一次实验
- benchmark 负责多次实验

也就是说：

- `run_experiment()` 只关心一个 config
- `run_benchmark()` 负责展开多个 config 并循环调用 `run_experiment()`

不要让 `run_experiment()` 同时知道“整个 benchmark 要跑多少组组合”。

## 错误处理策略

单实验 pipeline 需要有清晰的阶段性错误边界。

推荐按阶段记录：

- `config_error`
- `data_error`
- `model_error`
- `training_error`
- `evaluation_error`
- `artifact_error`

这样出问题时能够快速定位，而不是所有失败都变成一个模糊的 `RuntimeError`。

## 恢复策略

单实验 pipeline 也应支持最基本恢复能力，至少考虑：

- checkpoint resume
- 已完成指标结果跳过
- 已存在 artifact 的幂等写入

不过当前项目阶段，优先级应是：

1. 先写清楚状态文件
2. 再做 checkpoint 恢复

不要在主干还没稳定时先上复杂恢复系统。

## 日志与可观测性

一次实验至少应记录：

- 入口配置摘要
- 数据集名称与 split 大小
- 模型名称与参数摘要
- 训练开始/结束时间
- 主指标结果
- artifact 输出位置

这部分信息最好同时存在于：

- 终端输出
- `logs/`
- `status.json`

## 当前仓库的建议落地顺序

结合现有骨架，我建议按下面顺序推进：

1. 定义 `run_experiment()` 的输入输出结构
2. 固定单实验目录结构与状态文件
3. 接通 dataset -> model -> evaluator 的最小闭环
4. 再接 trainer 的 Lightning 路径
5. 最后完善 CLI 与 resume 能力

## 与现有文档的关系

这条主线应和已有文档形成闭环：

- [Configuration Guide](../concepts/configuration.md)：决定实验如何组合与覆盖
- [Dataset Guide](datasets.md)：决定数据如何进入 pipeline
- [Evaluation Guide](evaluation.md)：决定预测结果如何被消费
- [Benchmarking Guide](benchmarking.md)：决定多个 pipeline 如何被批量调度

## 需要避免的反模式

请尽量避免：

- `run_experiment()` 直接耦合 benchmark 全局状态
- pipeline 自己实现训练细节和指标计算
- 没有标准预测产物就直接接 evaluator
- 所有模型都强行走同一种 trainer
- 结果只打印到终端，不写结构化 artifact
- 出错时不记录阶段信息

## 一句话总结

对 RecBench 来说，最佳实践不是“把所有模块串起来就算 pipeline”，而是：

- 用配置冻结一次实验的真相源
- 用 dataset / model / trainer / evaluator 的稳定边界串起主干
- 用统一预测产物和 artifact 让单实验可复现、可调试、可被 benchmark 复用
