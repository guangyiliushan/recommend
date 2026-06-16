---
title: Project Structure
description: 标准化项目目录结构、职责边界与命名规范
---

# Project Structure

## 目标

RecBench 的目录结构不应只是“文件摆放位置”，而应承担三项作用：

- 让新成员快速判断代码和文档应该放在哪里
- 让运行时边界、工程边界与职责边界保持一致
- 让后续扩模型、扩数据、扩文档时不发生目录污染

这份文档定义项目的标准目录结构、职责定位、命名规范和新增文件落点规则。

## 顶层目录标准

推荐长期保持如下顶层结构：

```text
.
|-- configs/
|-- docs/
|-- scripts/
|-- src/recsys/
|-- tests/
|-- outputs/              # 运行产物目录，通常不入库
|-- .github/workflows/
|-- pyproject.toml
|-- uv.lock
|-- README.md
`-- CONTRIBUTING.md
```

## 顶层目录职责

### `configs/`

职责：

- Hydra 配置入口
- 实验组合定义
- 组件级配置预设

不应放：

- Python 业务逻辑
- 运行结果
- 数据集原始文件

### `docs/`

职责：

- 面向开发、维护和研究协作的文档站源码
- 项目架构、规范、实验、运维、论文与技术指南

不应放：

- 自动生成的大型二进制结果
- 与站点无关的临时笔记

### `scripts/`

职责：

- 用户可直接执行的 CLI 入口
- 薄封装脚本

不应放：

- 复杂业务实现
- 大量核心逻辑

推荐原则：

- `scripts/` 只做参数解析和入口转发
- 真正逻辑落在 `src/recsys/`

### `src/recsys/`

职责：

- 主业务代码
- 项目稳定 API 与运行时主干

这是整个仓库唯一的 Python 包真相源。

### `tests/`

职责：

- 契约测试
- 回归测试
- 聚焦行为验证

不应放：

- 仅用于人工调试的脚本
- 文档示例代码碎片

### `outputs/`

职责：

- 单实验产物
- benchmark 聚合结果
- logs、checkpoints、metrics、predictions

默认不建议纳入版本控制。

## `src/recsys/` 分层规范

推荐长期保持以下子模块分层：

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

职责：

- 提供基础契约
- 提供注册表
- 定义通用抽象边界

当前典型文件：

- `base_dataset.py`
- `base_model.py`
- `registry.py`

不应放：

- 某个具体模型实现
- 某个具体数据集下载脚本

## `src/recsys/data`

职责：

- 数据集适配器
- 预处理
- 负采样
- 特征工程

推荐子结构：

```text
data/
|-- datasets/
|-- preprocessor.py
|-- negative_sampling.py
|-- feature_engineering.py
`-- dataset_registry.py
```

边界要求：

- `datasets/` 只放 dataset adapter
- 负采样策略统一落在 `negative_sampling.py` 或其子模块
- 特征工程逻辑不要散落到模型目录

## `src/recsys/evaluation`

职责：

- 指标计算
- Evaluator
- ranking 评估
- 曲线与可视化

典型文件：

- `metrics.py`
- `evaluator.py`
- `ranking.py`
- `visualization.py`

边界要求：

- 数值计算与可视化分离
- evaluator 依赖统一预测产物，而不是依赖具体模型类

## `src/recsys/models`

职责：

- 各模型家族实现
- 与模型家族相关的局部组件

推荐按模型家族分目录：

- `classical/`
- `deep_ctr/`
- `sequence/`
- `feature_cross/`
- `pcvr/`
- `unified/`
- `generative/`

边界要求：

- 每个文件原则上只对应一个模型主实现
- 家族公共模块可以在家族目录内新增 `common.py`、`blocks.py` 等文件
- 不要把跨家族公用逻辑放回模型目录，应该上移到 `core/` 或 `utils/`

## `src/recsys/pipeline`

职责：

- 单实验编排
- 批量 benchmark 调度
- 聚合报告生成

典型文件：

- `experiment.py`
- `benchmark.py`
- `reporter.py`

边界要求：

- `experiment.py` 只关心一次实验
- `benchmark.py` 只关心多实验矩阵
- `reporter.py` 只关心结果聚合与输出

## `src/recsys/training`

职责：

- Trainer 封装
- callbacks
- losses
- optimizers
- schedulers

边界要求：

- 训练生命周期集中在这里
- 不要让模型目录承担通用训练基础设施

## `src/recsys/utils`

职责：

- 配置管理
- 设备管理
- 日志
- profiling
- 可复现性

边界要求：

- 这里放横切关注点
- 不放与单个模型强耦合的逻辑

## `docs/` 文档结构规范

当前推荐文档结构为：

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

## 各文档目录职责

### `docs/concepts`

职责：

- 稳定概念层说明
- 架构、配置、设计原则

适合放：

- `architecture.md`
- `configuration.md`

### `docs/project`

职责：

- 工程主干文档
- 数据、评估、pipeline、benchmark、结构、API、契约

这是当前最核心的工程文档目录。

### `docs/experiments`

职责：

- benchmark 套件
- baseline 页面
- 各实验线说明

适合放：

- `benchmark-classical.md`
- `baseline.md`
- `rankup.md`
- `interformer.md`

### `docs/guides`

职责：

- 技术指南
- 运用建议
- 工程实践文章

适合放：

- 学习率调度
- 梯度检查点
- Host Device Info
- Online Dataset EDA

### `docs/papers`

职责：

- 论文解读
- 数据管道论文
- 方法背景与文献映射

### `docs/operations`

职责：

- 运维、工具、站点生成、日志、缓存清理

## 文件命名规范

统一使用：

- 小写字母
- 连字符分词
- 明确语义

推荐示例：

- `project-structure.md`
- `api-contracts.md`
- `artifacts.md`
- `benchmarking.md`
- `learning-rate-scheduling.md`

不推荐示例：

- `ProjectStructure.md`
- `API_final_v2.md`
- `temp-note.md`

## Python 文件命名规范

推荐：

- 一个主文件对应一个主职责
- 家族模型文件名与模型名保持稳定映射

示例：

- `item_based_cf.py`
- `matrix_factorization.py`
- `wide_deep.py`

如果需要补公共块，优先使用：

- `common.py`
- `blocks.py`
- `types.py`

## 新文件落点决策规则

新增文件前，建议按下面顺序判断：

1. 这是概念文档、工程文档、实验文档、指南、论文还是运维文档
2. 这是公共抽象还是具体实现
3. 这是单次实验逻辑还是批量 benchmark 逻辑
4. 这是数据层、模型层、训练层、评估层还是横切工具

只有先判断边界，才能决定放在哪个目录。

## 常见新增内容的推荐落点

### 新数据集

代码：

- `src/recsys/data/datasets/{dataset_name}.py`

文档：

- `docs/project/datasets.md`
- 如需专项说明，可新增 `docs/project/dataset-{name}.md`

### 新模型

代码：

- `src/recsys/models/{family}/{model_name}.py`

文档：

- `docs/project/models.md`
- 若是重点实验模型，再新增 `docs/experiments/{model_name}.md`

### 新实验套件

配置：

- `configs/experiment/{suite_name}.yaml`

文档：

- `docs/experiments/{suite_name}.md`

### 新运行时规范

代码：

- `src/recsys/utils/` 或 `src/recsys/training/`

文档：

- `docs/guides/`
- 或 `docs/operations/`

## 需要避免的结构反模式

请尽量避免：

- 在 `scripts/` 中实现大量核心逻辑
- 在 `models/` 中写数据预处理
- 在 `data/` 中写模型专属逻辑
- 把实验结果混入源码目录
- 用临时命名文件长期留在仓库中
- 同一职责在多个目录重复出现

## 一句话总结

对 RecBench 来说，最佳实践不是“目录越多越好”，而是：

- 顶层目录按工程边界划分
- `src/recsys/` 按运行时职责分层
- `docs/` 按文档受众与用途分组
- 新文件按职责边界而不是个人习惯决定落点
