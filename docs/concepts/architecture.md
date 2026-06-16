---
title: Architecture
description: 当前架构边界、主干实现与最小闭环说明
---

# Architecture

## 架构目标

RecBench 的目标不是把所有推荐算法塞进一个统一的大脚本，而是在推荐实验中稳定住四条核心边界：

- Config：定义实验选择、组件参数和运行时环境
- Dataset Adapter：把原始数据转换为统一 split 与样本语义
- Model Contract：定义模型能力、输入输出与预测产物
- Experiment Runtime：把训练、评估、产物落盘和批量调度串成主干

围绕这四条边界，项目进一步拆分出 registry、training、evaluation、pipeline 等层，避免数据、模型、训练与评估相互侵入。

## 当前分层

### `src/recsys/core`

核心层负责稳定契约：

- `registry.py`：统一的模型、数据集、指标、loss 注册表
- `base_model.py`：`BaseRecommender`、`NeuralRecommender`、`Capability`、`Batch`、`ModelOutput`
- `prediction_bundle.py`：统一评估输入 `PredictionBundle`
- `base_dataset.py`：dataset adapter 的生命周期与 split 接口

这一层的职责是定义“系统中对象应该如何协作”，而不是承载具体模型或实验逻辑。

### `src/recsys/data`

数据层负责适配数据源：

- `datasets/`：当前已存在 TAAC 2025 与 TAAC 2026 适配器
- `dataset_registry.py`：导入并触发数据集注册

数据层当前已经能提供真实 adapter 样例，但预处理、负采样、特征工程仍未形成独立稳定模块，因此文档中只应描述已存在的数据适配能力，不应把预留文件写成已支持的数据管线。

### `src/recsys/models`

模型层按家族组织：

- `classical`
- `deep_ctr`
- `sequence`
- `feature_cross`
- `pcvr`
- `unified`
- `generative`

当前模型层最重要的架构决定是“最薄基类 + 能力接口”：

- `BaseRecommender` 只保留通用契约
- `NeuralRecommender` 负责神经网络模型的 `forward` / `compute_loss`
- 通过 `Capability` 显式声明模型是否可训练、是否可排序、是否可推荐

这保证了经典非训练模型与 Lightning 训练模型能够在同一主干内共存，而不必共享一套臃肿万能基类。

### `src/recsys/training`

训练层负责“可训练模型如何训练”，而不是“什么时候训练”：

- `trainer.py`：`LightningRecommender`、`TrainerFactory`、`create_trainer`
- `callbacks.py`：checkpoint、early stopping、梯度/显存监控与 run 摘要
- `losses.py`：loss 工厂与注册
- `optimizers.py`：optimizer 与参数组策略
- `schedulers.py`：scheduler 工厂
- `distributed.py`：设备与分布式策略解析

这一层已经完成了可训练模型的训练基础设施，已通过 `_execute_trainable_path()` 接入 experiment 主流程。

### `src/recsys/evaluation`

评估层负责“消费预测产物并产出结构化指标”：

- `metrics.py`：分类与点预测指标
- `ranking.py`：按组排序指标
- `visualization.py`：曲线 JSON / CSV 导出与可选绘图
- `evaluator.py`：pointwise、ranking、multitask 路由与结果组装

评估层当前已经不是占位骨架，而是可被 pipeline 直接调用的独立模块。

### `src/recsys/pipeline`

pipeline 层负责运行时编排：

- `experiment.py`：单实验主干、状态管理、artifact 落盘
- `benchmark.py`：实验矩阵展开、恢复策略、串行或受控并发调度
- `reporter.py`：聚合 CSV 与 HTML 报告

这里需要特别区分：

- `training` 决定如何训练
- `pipeline` 决定何时构建数据、实例化模型、评估与写结果
- `benchmark` 决定如何批量调度多个实验

## 当前最小闭环

当前仓库已经能打通的最小闭环是：

1. 加载已注册的数据集 adapter
2. 通过模型注册表获取 `itemcf`（非训练）或 `dssm`（训练）
3. 走 `run_experiment()` 的非训练式路径或训练式路径
4. 用 `evaluate()` 计算指标
5. 落盘标准 artifact
6. 用 `run_benchmark()` 和 `Reporter` 聚合多次 run

这个最小闭环的意义是验证：

- 非训练模型与可训练神经网络模型均能在统一主干内共存
- 评估层能直接消费标准化预测产物
- artifact 契约足够支撑批量聚合和恢复策略

## 当前边界与限制

当前必须明确写清楚的边界有：

- 当前可运行模型主要为 `itemcf`（非训练 + ranking）和 `dssm`（训练 + pointwise）
- 模型家族目录已齐全，但实际可运行模型仍很少
- `scripts/run_ablation.py`、`scripts/download_data.py` 等辅助脚本仍待完善

因此，当前文档应避免把“架构预留”写成“功能已交付”。

## 关键设计原则

### 1. 先稳定共享契约

优先稳定下面这些接口，再扩展模型数量：

- 数据 split 语义
- `Batch` / `PredictionBundle`
- 模型能力声明
- evaluator 输入输出
- artifact 协议

### 2. 让不同范式共存

推荐系统里会同时出现：

- 非训练式经典方法
- 神经网络点预测模型
- 序列推荐模型
- 多任务模型

因此主干必须围绕“能力声明”和“标准产物”构建，而不是假设所有模型都走统一训练循环。

### 3. 保持职责单一

- dataset adapter 不应承担模型逻辑
- trainer 不应承担 benchmark 调度
- evaluator 不应承担训练逻辑
- reporter 不应承担单实验执行

## 现阶段的推荐推进顺序

1. 继续稳定 `PredictionBundle`、artifact 与错误模型
2. 扩展一个训练型样板模型，接通 experiment 主路径
3. 再逐步补充更多模型家族
4. 最后再完善 CLI、预设解析和更高级调度能力
