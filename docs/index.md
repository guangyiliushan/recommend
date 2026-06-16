---
title: RecBench Documentation
description: 项目概览、当前实现状态与文档导航
---

# RecBench Documentation

RecBench 是一个面向推荐系统研究与工程实践的基准框架。项目的目标不是简单堆叠模型目录，而是在统一配置、数据适配、模型契约、训练基础设施、评估与产物协议下，逐步形成可复现、可比较、可维护的推荐实验平台。

## 当前实现状态

当前代码仓库已经完成的主干能力包括：

- `src/recsys/core` 中的注册表、模型契约与 `PredictionBundle`
- `src/recsys/utils` 中的结构化配置、设备探测、日志、可复现性与 profiling 工具
- `src/recsys/training` 中的 Lightning 训练封装、callbacks、loss、optimizer、scheduler 与分布式策略解析
- `src/recsys/evaluation` 中的 pointwise、ranking、multitask 评估与曲线/可视化导出
- `src/recsys/pipeline/experiment.py` 中的单实验主干
- `src/recsys/pipeline/benchmark.py` 与 `src/recsys/pipeline/reporter.py` 中的批量调度、恢复策略与聚合报告
- `src/recsys/models/classical/item_based_cf.py` 中的 `itemcf` 基线模型

当前仍需明确保留的边界包括：

- 单实验中的可训练模型路径尚未接通 `trainer`
- 当前真正完成并可用于最小闭环的模型主要是 `itemcf`
- 大部分模型家族文件仍是目录预留或占位实现
- `scripts/` 下的 CLI 入口仍是骨架，不应作为现成命令行工具宣传

因此，RecBench 当前不是“全模型已完工的 Benchmark 套件”，而是“核心运行时已成形、可从 `itemcf` 最小闭环继续向外扩展”的推荐系统工程框架。

## 建议阅读顺序

建议按下面顺序阅读：

1. [Getting Started](getting-started.md)
2. [Architecture](concepts/architecture.md)
3. [Configuration Guide](concepts/configuration.md)
4. [Project Structure](project/structure.md)
5. [Public API](project/api-contracts.md)
6. [Pipeline Guide](project/pipeline.md)
7. [Evaluation Guide](project/evaluation.md)
8. [Benchmarking Guide](project/benchmarking.md)
9. [Persistence Contracts](project/artifacts.md)
10. [Dataset Guide](project/datasets.md)
11. [Model Integration Guide](project/models.md)
12. [Development Guide](project/development.md)
13. [Experiments](experiments/index.md)
14. [Guides](guides/index.md)
15. [Papers](papers/index.md)
16. [Operations Overview](operations/overview.md)
17. [Maintenance Guide](operations/maintenance.md)

## 文档分组

- `concepts/`：解释稳定的概念边界与设计原则
- `project/`：描述当前仓库中已经存在的工程主干、契约与运行时能力
- `experiments/`：沉淀实验矩阵、基线方案与复现实验说明
- `guides/`：沉淀工程实践、调优建议与专题指南
- `papers/`：记录论文背景与工程映射
- `operations/`：维护文档站、结果目录与仓库日常运维

## 核心原则

- 文档只描述当前仓库真实存在的实现能力
- 已实现、部分实现、未实现必须显式区分
- 配置、数据、模型、评估与运行时保持边界清晰
- 所有运行结果都优先通过结构化 artifact 表达
- 扩展模型前，先稳定共享契约与最小可运行闭环
