---
title: Architecture
description: 推荐项目的建议分层、边界和最小闭环
---

# Architecture

## 目标架构

RecBench 的目标不是把所有算法塞进一个大脚本，而是建立一个可扩展的 benchmark 平台。建议的稳定边界如下：

- Config: 定义运行参数与对象实例化方式
- Registry: 提供模型、数据集、指标等对象的命名发现能力
- Dataset Adapter: 负责原始数据读取、切分和 batch schema
- Model Contract: 定义模型输入输出与训练能力
- Trainer: 负责可训练模型的训练生命周期
- Evaluator: 负责预测结果汇总和指标计算
- Experiment Pipeline: 负责单次实验编排
- Benchmark Pipeline: 负责批量实验调度与聚合

## 当前目录与职责

### `src/recsys/core`

- `registry.py`: 注册表机制
- `base_model.py`: 模型统一契约的骨架
- `base_dataset.py`: 数据加载与 split 抽象

### `src/recsys/data`

- `datasets/`: 数据集适配器
- `preprocessor.py`: 预处理占位
- `negative_sampling.py`: 负采样占位
- `feature_engineering.py`: 特征工程占位

### `src/recsys/models`

按模型家族分目录：

- `classical`
- `deep_ctr`
- `sequence`
- `feature_cross`
- `pcvr`
- `unified`
- `generative`

### `src/recsys/pipeline`

- `experiment.py`: 单实验编排
- `benchmark.py`: 批量 benchmark 编排
- `reporter.py`: 结果汇总与报告

## 最佳实践建议

### 1. 先稳定契约，再扩实现

优先完成以下统一接口：

- 数据 batch schema
- 模型输出 schema
- evaluator 输入格式
- artifact 输出目录结构

### 2. 不要让单个抽象承担过多职责

例如：

- `BaseDataset` 不应同时承担所有具体 split dataset 的行为细节
- `Trainer` 不应接管 benchmark 调度
- `Evaluator` 不应强依赖 Lightning

### 3. 允许多种模型范式共存

这个项目包含：

- 非训练式经典方法
- 基于梯度训练的神经方法
- 多任务建模
- 可能依赖生成式解码的推荐方法

因此顶层接口应该强调“能力契约”，而不是强行让所有模型遵守同一种内部实现方式。

## 推荐 MVP 路线

第一阶段建议先打通一个最小闭环：

1. 一个真实可用数据集
2. 一个非神经 baseline
3. 一个可训练 baseline
4. 一套核心指标
5. 一次单实验运行流程

等最小闭环稳定后，再逐步扩展模型家族与数据集矩阵。
