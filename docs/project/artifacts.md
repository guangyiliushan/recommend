---
title: Persistence Contracts
description: 当前单实验与批量 Benchmark 的真实 artifact 协议
---

# Persistence Contracts

## 目标

RecBench 当前已经把配置快照、状态文件、指标结果、预测文件与 Benchmark 聚合表落成真实文件输出。

因此本页重点不是继续讨论“将来应该保存什么”，而是明确当前代码已经保存了什么、保存到哪里、怎样被恢复与聚合逻辑消费。

## 输出根目录

当前输出目录分为两层：

```text
outputs/
|-- experiments/
|   `-- {run_id}/
`-- benchmarks/
    `-- {benchmark_name}/
```

- `experiments/`：单实验原子结果
- `benchmarks/`：多实验聚合结果

## 单实验产物

## `config.yaml`

### 当前作用

- 保存单实验配置快照
- 作为 run 的复现依据
- 用于恢复逻辑中的配置一致性判断

### 当前路径

- `outputs/experiments/{run_id}/config.yaml`

### 当前要求

- 与 run 目录唯一对应
- 在运行开始阶段写出
- 应能被恢复逻辑重新解析

## `status.json`

### 当前作用

- 记录当前 run 的状态
- 支撑恢复与幂等跳过
- 承接结构化错误信息

### 当前路径

- `outputs/experiments/{run_id}/status.json`

### 当前状态值

- `pending`
- `running`
- `succeeded`
- `failed`
- `skipped`

### 当前使用场景

- Benchmark 在恢复时会优先检查它
- 成功 run 的跳过判断依赖它与 `metrics.json`

## `metrics.json`

### 当前作用

- 保存结构化评估结果
- 提供 `summary_metrics` 给 Benchmark 聚合
- 提供 `task_metrics`、`group_metrics` 与 `curve_artifacts` 给下游分析

### 当前路径

- `outputs/experiments/{run_id}/metrics.json`

### 当前主要字段

- `summary_metrics`
- `task_metrics`
- `group_metrics`
- `curve_artifacts`
- `metadata`

## `predictions.parquet`

### 当前作用

- 保存预测明细
- 支撑离线复盘与误差分析

### 当前路径

- `outputs/experiments/{run_id}/predictions.parquet`

### 当前注意事项

- 并非所有任务未来都必须完全同列，但字段语义应围绕 `PredictionBundle` 保持稳定
- ranking 场景应保留分组语义

## `curves/`

### 当前作用

- 保存结构化曲线数据

### 当前路径

- `outputs/experiments/{run_id}/curves/`

### 当前典型内容

- ROC 曲线 JSON
- PR 曲线 JSON
- Top-K 指标曲线 JSON
- threshold sweep JSON

当前实现优先保存“原始结构化曲线数据”，图片并不是主产物。

## `logs/`

### 当前作用

- 保存单实验日志
- 至少用于失败排查

### 当前路径

- `outputs/experiments/{run_id}/logs/`

### 当前已明确存在的文件

- `stderr.log`

注意：文档不应把 `run.log` 或 `trainer.log` 写成当前单实验主干的既有产物，除非代码真正写出了这些文件。

## `checkpoints/`

### 当前状态

训练基础设施支持 checkpoint callback，但由于训练型 experiment 路径尚未接通，`checkpoints/` 目前不应被写成单实验主路径上的既有常规产物。

文档中更准确的表达是：

- 训练层已具备 checkpoint 机制
- 当训练型路径接通后，`checkpoints/` 会成为重要产物目录

## 批量 Benchmark 产物

## `manifest.json`

### 当前作用

- 记录一次 Benchmark 的矩阵范围与产物索引

### 当前路径

- `outputs/benchmarks/{benchmark_name}/manifest.json`

### 当前典型字段

- `benchmark_name`
- `created_at`
- `runs`
- `models`
- `datasets`
- `seeds`
- `total`
- `succeeded`
- `failed`
- `summary_path`
- `failures_path`
- `leaderboard_path`
- `report_path`

## `summary.csv`

### 当前作用

- 保存逐 run 的展平摘要

### 当前路径

- `outputs/benchmarks/{benchmark_name}/summary.csv`

### 当前用途

- 查看每次实验的状态与摘要指标
- 作为后续聚合或导出分析表的基础输入

## `leaderboard.csv`

### 当前作用

- 基于成功 run 聚合后形成排序视图

### 当前路径

- `outputs/benchmarks/{benchmark_name}/leaderboard.csv`

### 当前用途

- 查看同一模型在同一数据集上的均值表现
- 结合 `std` 和 `num_runs` 做稳定性比较

## `failures.csv`

### 当前作用

- 记录失败 run 的结构化信息

### 当前路径

- `outputs/benchmarks/{benchmark_name}/failures.csv`

### 当前用途

- 快速定位失败组合、阶段和错误信息

## `trend.csv`

### 当前作用

- 保存逐 run 的趋势信息

### 当前路径

- `outputs/benchmarks/{benchmark_name}/trend.csv`

### 当前用途

- 分析多 seed 波动和主指标趋势

## `stability.csv`

### 当前作用

- 保存聚合稳定性统计

### 当前路径

- `outputs/benchmarks/{benchmark_name}/stability.csv`

### 当前用途

- 查看均值、标准差、变异系数等稳定性指标

## `report.html`

### 当前作用

- 提供人工浏览友好的聚合摘要页

### 当前路径

- `outputs/benchmarks/{benchmark_name}/report.html`

## 当前恢复与一致性逻辑

当前恢复逻辑主要依赖下面几项一致性检查：

1. `status.json` 是否存在
2. `metrics.json` 是否存在
3. `status.json` 是否为 `succeeded`
4. `config.yaml` 中的配置 hash 是否与当前请求匹配

这意味着 artifact 契约当前已经直接参与运行控制，不只是“供人工查看”。

## 当前推荐的一致性检查

更新文档或调试产物时，建议至少检查：

1. `status.json` 与 `metrics.json` 是否对应
2. `manifest.json` 中的 run 数是否与 `summary.csv` 一致
3. `leaderboard.csv` 是否只基于成功 run 聚合
4. `config.yaml` 是否可重新解析
5. `predictions.parquet` 与任务类型语义是否匹配

## 当前需要避免的误写

文档不应把下面这些内容写成“已经稳定存在”：

- 单实验默认会写 `checkpoints/`
- 单实验默认会写 `run.log`
- 所有任务都已拥有统一的最终预测明细列集合
- 所有 artifact 都已版本化并完全支持迁移

这些方向是合理的，但当前仓库实现还没有全部走到那一步。

## 当前最重要的结论

RecBench 已经拥有一套真实可用的 artifact 主干：单实验写配置、状态、指标、预测与曲线，批量 Benchmark 写 manifest、summary、leaderboard、failures、trend、stability 和 HTML 报告。文档应以这套真实产物为准，而不是继续用泛化的“建议保存”口径替代当前实现。
