---
title: Dataset Guide
description: 当前 dataset adapter、split 语义与数据层边界
---

# Dataset Guide

## 目标

RecBench 的数据层负责把原始数据源转换成可被模型和 pipeline 消费的 dataset adapter。

当前仓库中，这一层已经有真实实现，重点包括：

- `BaseDataset`
- `dataset_registry.py`
- `taac2025.py`
- `taac2026.py`

因此本页应围绕当前真实数据适配能力来写，而不是把尚未稳定的预处理或负采样方向写成既有功能。

## 当前代码结构

当前数据层核心文件包括：

- `src/recsys/core/base_dataset.py`
- `src/recsys/data/dataset_registry.py`
- `src/recsys/data/datasets/taac2025.py`
- `src/recsys/data/datasets/taac2026.py`

## 当前已实现的数据集样例

当前最明确的 dataset adapter 样例是：

- TAAC 2025 相关数据适配器
- TAAC 2026 相关数据适配器

文档应把它们写成“真实存在的数据集样例”，而不是把 Criteo、MovieLens 等尚未接通的数据源写成当前默认可用数据集。

## `BaseDataset` 当前契约

当前 `BaseDataset` 负责：

- `load()`
- `get_split()`
- `get_dataloader()`

子类通常需要负责：

- 原始数据加载
- split 切分
- 样本输出格式
- 数据集元信息

从当前实现看，`BaseDataset` 更接近“dataset adapter 容器”，而不是直接充当最终训练样本集本体。

## Split 语义

当前推荐和实现对齐的 split 名称是：

- `train`
- `val`
- `test`
- `full`

它们表达的是实验语义切分，而不是任意子集切片。

## 当前 batch 风格

从 TAAC 2025 和 TAAC 2026 的适配器样例可以看出，当前数据层已经默认采用“字典形式 batch”作为样本接口。

常见字段类型包括：

- 标识字段：`user_id`、`item_id`
- 监督字段：`label` 或 `labels`
- 特征字段：`user_feats`、`item_feats`
- 序列字段：`domain_seqs`、`item_ids`

这类设计与当前 `Batch` 视图、training 和 evaluator 层的使用方式一致。

## 数据层当前边界

dataset adapter 当前应该负责：

- 原始数据读取
- split 构造
- 样本格式定义
- 元信息暴露

当前不应把下面这些方向写成已稳定实现：

- 统一的独立负采样模块
- 完整的特征工程模块
- 完整的预处理流水线
- 完整的多模态 batch 标准层

这些方向在架构上是合理的，但当前仓库还没有作为稳定子系统落地。

## 当前已知限制

当前数据层仍有一些需要文档保留的限制：

- split 策略仍偏 MVP，未必适合所有严格时序实验
- 负采样能力还没有独立成稳定公共模块
- 多模态语义在部分数据集中已有元信息基础，但训练 batch 契约尚未完全统一

## 示例

```python
from recsys.data.datasets.taac2026 import TAAC2026DataSample

dataset = TAAC2026DataSample(root_dir="./data").load()
train_split = dataset.get_split("train")
train_loader = dataset.get_dataloader("train", batch_size=256, num_workers=4)
```

## 当前最重要的结论

RecBench 的数据层已经拥有真实可用的 dataset adapter 体系，但文档应聚焦于 TAAC 2025 / 2026 这些当前已存在的适配器与 `BaseDataset` 契约本身，不应把尚未落地的预处理、负采样和通用多模态机制写成已完成能力。
