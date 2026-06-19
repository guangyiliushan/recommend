---
title: Project Structure
description: 当前仓库目录结构、职责边界与新增文件落点规则
---

# Project Structure

## 目标

RecBench 的目录结构需要同时服务于：

- 运行时边界清晰
- 文档与代码同步维护
- 新增模型、数据集和实验时的可扩展性

本页描述的是当前仓库已经采用的结构与实际职责，而不是一套脱离实现的理想目录图。

## 顶层目录

当前顶层结构可概括为：

```text
.
|-- configs/
|-- docs/
|-- scripts/
|-- src/recsys/
|-- tests/
|-- outputs/
|-- .github/workflows/
|-- pyproject.toml
|-- uv.lock
|-- README.md
`-- CONTRIBUTING.md
```

## 顶层职责

### `configs/`

- 保存主配置与实验矩阵
- 保存 dataset / model / training / evaluation / runtime 相关配置

不应放：

- Python 业务实现
- 实验产物
- 原始数据文件

### `docs/`

- 保存文档站源码
- 保存项目概念、工程、实验、运维与指南文档

不应放：

- 自动生成结果
- 与站点无关的临时草稿

### `scripts/`

- 承载 CLI 实验入口与工具脚本
- `run.py` — Hydra 入口（YAML 组合 + CLI 覆盖）
- `run_single.py` — 单实验 argparse 入口
- `run_benchmark.py` — 批量 benchmark 入口
- `run_ablation.py` — 消融实验矩阵
- `download_data.py` — HuggingFace 数据集下载
- `generate_report.py` — 从 summary.csv 生成对比报告
- `benchmark_data_pipeline.py` — 数据存储格式基准
- `train_hyformer.py` — HyFormer 模型训练（模拟数据 / 调试用）

### `src/recsys/`

- 项目唯一的 Python 包真相源
- 所有核心契约与运行时主干都应落在这里

### `tests/`

- 契约测试
- 回归测试
- 聚焦行为验证

### `outputs/`

- 单实验结果
- 批量 Benchmark 聚合结果

默认不建议纳入版本控制。

## `src/recsys/` 分层

当前主要分层如下：

```text
src/recsys/
|-- core/
|-- data/
|-- evaluation/
|-- models/
|-- pipeline/
|-- training/
`-- utils/
```

## `src/recsys/core`

负责：

- 注册表
- 基础模型契约
- `PredictionBundle`
- dataset adapter 基础抽象

这一层只定义通用边界，不承载具体模型与具体数据集。

## `src/recsys/data`

负责：

- 数据集适配器
- 数据集注册入口

当前真实已存在的重点是 `datasets/` 与 `dataset_registry.py`。预处理、负采样、特征工程等方向目前还没有形成完整稳定模块，因此文档不应把这些预留方向写成已交付的数据子系统。

## `src/recsys/evaluation`

负责：

- 分类指标计算
- 排序指标计算
- evaluator 路由
- 曲线导出与可选绘图

当前该层已经具备实际实现，而不是占位。

## `src/recsys/models`

负责：

- 模型家族目录组织
- 模型注册与发现入口

当前要特别注意：

- 家族目录完整，不等于家族内模型都已实现
- 当前真正适合写入文档的可运行模型主要是 `itemcf`

## `src/recsys/pipeline`

负责：

- 单实验主干
- 批量 Benchmark 调度
- 聚合报告

当前边界已经清晰拆成：

- `experiment.py`
- `benchmark.py`
- `reporter.py`

## `src/recsys/training`

负责：

- Lightning 训练封装
- callbacks
- loss 工厂
- optimizer 工厂
- scheduler 工厂
- 分布式策略解析

这一层已经完成训练基础设施，但 experiment 主流程尚未接通训练型模型路径。

## `src/recsys/utils`

负责横切关注点：

- 配置
- 设备
- 日志
- profiling
- 可复现性

这类代码应保持与具体模型家族解耦。

## 文档目录

当前文档结构为：

```text
docs/
|-- index.md
|-- getting-started.md
|-- concepts/
|-- project/
|-- experiments/
|-- guides/
|-- papers/
`-- operations/
```

### `docs/concepts`

- 稳定概念和设计原则

### `docs/project`

- 当前仓库的工程主干与公共契约

### `docs/experiments`

- 实验矩阵、基线方案与复现说明

### `docs/guides`

- 工程实践与调优建议

### `docs/papers`

- 论文背景与工程映射

### `docs/operations`

- 仓库维护、文档站与结果目录运维

## 命名规范

### Markdown 文件

- 使用小写
- 使用连字符
- 名称语义明确

### Python 文件

- 一个主文件对应一个主职责
- 模型文件名与模型名保持稳定映射
- 公共块可用 `common.py`、`blocks.py`、`types.py`

## 新文件落点规则

新增文件前建议按顺序判断：

1. 这是概念、工程、实验、指南、论文还是运维文档
2. 这是公共抽象还是具体实现
3. 这是单实验逻辑还是批量 Benchmark 逻辑
4. 这是数据、模型、训练、评估还是横切工具

## 常见新增内容的落点

### 新数据集

- 代码：`src/recsys/data/datasets/{dataset_name}.py`
- 文档：`docs/project/datasets.md`

### 新模型

- 代码：`src/recsys/models/{family}/{model_name}.py`
- 文档：`docs/project/models.md`
- 如需实验说明，再补 `docs/experiments/{model_name}.md`

### 新实验套件

- 配置：`configs/experiment/{suite_name}.yaml`
- 文档：`docs/experiments/{suite_name}.md`

### 新运行时工具

- 代码：`src/recsys/utils/` 或 `src/recsys/training/`
- 文档：`docs/guides/` 或 `docs/operations/`

## 需要避免的反模式

- 在 `scripts/` 中堆放核心业务逻辑
- 在 `models/` 中写数据预处理
- 在 `data/` 中写模型专属逻辑
- 把实验产物混进源码目录
- 因目录预留而误写功能状态

## 当前最重要的结论

RecBench 的目录结构已经体现出清晰的工程边界。文档中最需要避免的错误，是把“目录已经建好”误写成“对应能力已经全部实现”。
