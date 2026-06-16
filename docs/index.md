---
title: RecBench Documentation
description: 项目概览、目标边界和当前状态
---

# RecBench Documentation

RecBench 是一个面向推荐系统研究与工程的 Benchmark 项目，目标是在统一的工程约束下比较经典协同过滤、深度 CTR/CVR、序列推荐、特征交叉、多任务建模和生成式推荐方法。

## 文档适用范围

这套文档主要面向三类读者：

- 想快速了解项目目标与边界的使用者
- 想接入数据集、模型或评估逻辑的开发者
- 需要维护 CI、文档站点和发布流程的维护者

## 当前仓库状态

当前仓库已经具备以下基础：

- 明确的源码分层，位于 `src/recsys`
- 多个模型家族的目录骨架
- TAAC 2025 与 TAAC 2026 的数据集适配示例
- 配置目录、CI、发布与文档部署工作流

当前仓库仍在建设中的部分：

- 统一的实验编排闭环
- 标准化的训练适配层
- 完整可运行的 benchmark runner
- 全量模型的实际实现

因此，本项目目前更接近“工程骨架与基线平台”，而不是已经完全落地的“全模型基准测试套件”。

## 文档阅读顺序

建议按下面顺序阅读：

1. [Getting Started](getting-started.md)
2. [Concepts: Architecture](concepts/architecture.md)
3. [Concepts: Configuration](concepts/configuration.md)
4. [Project: Structure](project/structure.md)
5. [Project: API Contracts](project/api-contracts.md)
6. [Project: Artifacts](project/artifacts.md)
7. [Project: Dataset Guide](project/datasets.md)
8. [Project: Evaluation Guide](project/evaluation.md)
9. [Project: Pipeline Guide](project/pipeline.md)
10. [Project: Benchmarking Guide](project/benchmarking.md)
11. [Project: Development Guide](project/development.md)
12. [Project: Model Integration](project/models.md)
13. [Experiments](experiments/index.md)
14. [Guides](guides/index.md)
15. [Papers](papers/index.md)
16. [Operations: Overview](operations/overview.md)
17. [Operations: Maintenance](operations/maintenance.md)

## 文档分组

当前文档体系按以下维度组织：

- `concepts/`: 架构、配置与稳定概念层
- `project/`: 工程主干、API、契约、数据、评估、pipeline 与 benchmark
- `experiments/`: 实验套件、baseline 与模型实验页面
- `guides/`: 技术指南与工程实践专题
- `papers/`: 论文解读与方法背景映射
- `operations/`: 运维、工具、站点生成与维护流程

## 核心原则

- 文档必须反映仓库真实状态
- 先稳定共享契约，再扩大量模型
- 配置、数据、模型、评估和运行时要保持边界清晰
- 优先考虑可维护性、可测试性和可复现性
