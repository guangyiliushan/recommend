---
title: Persistence Contracts
description: 关键持久化契约、存储格式、生命周期、一致性要求与版本规则
---

# Persistence Contracts

## 目标

RecBench 的运行结果、配置快照和聚合产物必须具备稳定持久化契约，否则会出现：

- benchmark 结果不可复现
- 失败恢复无法判断状态
- 旧结果与新代码无法兼容
- 自动化聚合和报告生成缺少统一输入

这份文档定义项目级关键持久化对象的存储格式、字段语义、生命周期、一致性约束和版本演进规则。

## 持久化对象范围

当前项目最关键的持久化对象包括：

- 配置快照
- 运行状态文件
- 单实验指标结果
- 单实验预测结果
- checkpoint
- benchmark 汇总结果
- leaderboard
- 失败记录
- 日志与可视化产物

## 总体原则

所有持久化对象都应遵守：

- 结构化优先于截图或纯文本
- 原始结果优先于只保留汇总结果
- 文件名稳定且可预测
- 能支持自动扫描、恢复与版本升级

## 标准输出根目录

推荐长期采用如下目录分层：

```text
outputs/
|-- experiments/
|   `-- {run_id}/
`-- benchmarks/
    `-- {benchmark_name}/
```

### 语义说明

- `experiments/`：单次实验的原子结果目录
- `benchmarks/`：多实验聚合视图目录

## 契约 1：`config.yaml`

### 作用

- 保存一次实验最终解析后的完整配置快照
- 作为实验复现的真相源

### 推荐路径

- `outputs/experiments/{run_id}/config.yaml`

### 推荐格式

- YAML

### 最低要求

- 必须是 fully resolved config
- 必须包含 dataset、model、training、evaluation、runtime、experiment 信息
- 不应只保存 CLI 覆盖片段

### 生命周期

- 创建时机：实验开始前
- 更新策略：只写一次，不在运行中途重写
- 保留策略：与 run 同生命周期

### 一致性要求

- `config.yaml` 与当前 run 的目录、状态和指标必须一一对应
- benchmark 聚合时若复用历史 run，必须能根据该文件判断配置一致性

## 契约 2：`status.json`

### 作用

- 记录单实验当前状态
- 支撑失败恢复和幂等执行

### 推荐路径

- `outputs/experiments/{run_id}/status.json`

### 推荐格式

- JSON

### 推荐结构

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `run_id` | `string` | 是 | 唯一实验标识 |
| `status` | `string` | 是 | `pending` / `running` / `succeeded` / `failed` / `skipped` |
| `started_at` | `string` | 否 | ISO 8601 时间 |
| `finished_at` | `string` | 否 | ISO 8601 时间 |
| `dataset` | `string` | 是 | 数据集名 |
| `model` | `string` | 是 | 模型名 |
| `seed` | `integer` | 否 | 随机种子 |
| `primary_metric` | `string \| null` | 否 | 主指标名 |
| `primary_metric_value` | `number \| null` | 否 | 主指标值 |
| `error` | `object \| null` | 否 | 结构化错误信息 |
| `resume_supported` | `boolean` | 否 | 是否支持恢复 |

### 生命周期

- 创建时机：实验启动时
- 更新时机：阶段切换、成功结束、失败退出时
- 保留策略：与 run 同生命周期

### 一致性要求

- `status = succeeded` 时，应存在 `metrics.json`
- `status = failed` 时，应存在 `error`
- 不允许 `finished_at` 早于 `started_at`

## 契约 3：`metrics.json`

### 作用

- 保存单实验结构化评估结果

### 推荐路径

- `outputs/experiments/{run_id}/metrics.json`

### 推荐格式

- JSON

### 推荐结构

| 字段 | 类型 | 说明 |
|---|---|---|
| `summary_metrics` | `object` | 主指标与摘要指标 |
| `task_metrics` | `object` | 多任务或分任务指标 |
| `group_metrics` | `object` | 分组诊断结果 |
| `curve_artifacts` | `object` | 曲线文件或曲线数据索引 |
| `metadata` | `object` | 任务类型、阈值、K 值、样本规模等 |

### 生命周期

- 创建时机：评估结束后
- 更新策略：原则上一次写入；若允许增量写入，必须保证原子性

### 一致性要求

- `summary_metrics` 中应包含 benchmark 使用的主指标
- `metadata.task_type` 应与配置与模型契约一致

## 契约 4：`predictions.parquet`

### 作用

- 保存单实验预测明细
- 支撑复盘、误差分析、统计检验与二次聚合

### 推荐路径

- `outputs/experiments/{run_id}/predictions.parquet`

### 推荐格式

- Parquet

### 推荐字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_id` | `string \| int` | 样本标识 |
| `user_id` | `string \| int \| null` | 用户标识 |
| `item_id` | `string \| int \| null` | 物品标识 |
| `group_id` | `string \| int \| null` | ranking 分组标识 |
| `task_name` | `string \| null` | 多任务任务头 |
| `y_true` | `number \| bool` | 真实标签 |
| `y_score` | `number` | 连续预测分数 |
| `y_pred` | `number \| bool \| null` | 离散预测 |
| `split` | `string` | 预测来源 split |

### 生命周期

- 创建时机：预测结果生成后
- 保留策略：可选，但 benchmark 级统计和误差分析强烈建议保留

### 一致性要求

- pointwise 可无 `group_id`
- ranking 必须有 `group_id`
- multitask 应通过 `task_name` 或多文件拆分明确任务头

## 契约 5：`checkpoints/`

### 作用

- 保存模型训练中间状态或最终状态

### 推荐路径

- `outputs/experiments/{run_id}/checkpoints/`

### 推荐内容

- `best.ckpt`
- `last.ckpt`
- 可选的 `epoch={n}.ckpt`

### 生命周期

- 创建时机：训练过程和训练结束时
- 清理策略：根据保留策略决定是否只保留 best/last

### 一致性要求

- checkpoint 必须对应当前 `config.yaml`
- checkpoint 元信息中应包含模型名、epoch、global step

## 契约 6：`logs/`

### 作用

- 保存运行日志、训练日志、错误日志

### 推荐路径

- `outputs/experiments/{run_id}/logs/`

### 推荐内容

- `run.log`
- `stderr.log`
- `trainer.log`

### 一致性要求

- 关键错误应能在日志中定位到阶段
- `status.json` 中的失败摘要应能对应日志内容

## 契约 7：`curves/`

### 作用

- 保存 ROC、PR、Top-K、训练曲线等图或数据文件

### 推荐路径

- `outputs/experiments/{run_id}/curves/`

### 推荐内容

- `roc_curve.json`
- `pr_curve.json`
- `ndcg_at_k.json`
- `training_curve.csv`

推荐优先存“原始曲线数据”，图片可以作为附加产物。

## 契约 8：`manifest.json`

### 作用

- 作为 benchmark 级索引文件
- 记录整批实验矩阵和参与 run

### 推荐路径

- `outputs/benchmarks/{benchmark_name}/manifest.json`

### 推荐结构

| 字段 | 类型 | 说明 |
|---|---|---|
| `benchmark_name` | `string` | benchmark 名称 |
| `created_at` | `string` | 创建时间 |
| `runs` | `array` | run_id 列表 |
| `models` | `array` | 模型集合 |
| `datasets` | `array` | 数据集集合 |
| `seeds` | `array` | seed 集合 |
| `summary_path` | `string` | 汇总文件路径 |
| `failures_path` | `string` | 失败列表路径 |

### 一致性要求

- `runs` 中每个 run 必须能在 `experiments/` 下找到
- manifest 中的配置范围必须和 summary/leaderboard 一致

## 契约 9：`summary.csv`

### 作用

- 保存 benchmark 中所有 run 的展平摘要

### 推荐路径

- `outputs/benchmarks/{benchmark_name}/summary.csv`

### 推荐列

- `run_id`
- `dataset`
- `model`
- `seed`
- `status`
- `primary_metric`
- 关键指标列

### 一致性要求

- 一行对应一个 run
- `run_id` 应唯一
- 失败 run 可以保留，但需标明 `status`

## 契约 10：`leaderboard.csv`

### 作用

- 保存基于主指标排序后的聚合视图

### 推荐路径

- `outputs/benchmarks/{benchmark_name}/leaderboard.csv`

### 推荐列

- `model`
- `dataset`
- `primary_metric`
- `mean`
- `std`
- `rank`
- `num_runs`

### 一致性要求

- 聚合方法必须在同版本文档中明确
- 不同任务类型不要混在同一排行榜中直接比较

## 契约 11：`failures.csv`

### 作用

- 保存 benchmark 中所有失败 run 的结构化信息

### 推荐路径

- `outputs/benchmarks/{benchmark_name}/failures.csv`

### 推荐列

- `run_id`
- `dataset`
- `model`
- `seed`
- `phase`
- `error_code`
- `error_message`

### 一致性要求

- 每一条失败记录应能映射到对应 run 目录
- `error_code` 应与项目统一错误模型保持一致

## 数据模型版本化

所有关键持久化对象建议包含 schema 版本信息。

### 推荐字段

- `schema_version`

### 推荐规则

- 新增可选字段：小版本兼容升级
- 删除或重命名字段：大版本升级
- 改变核心语义：必须升级版本并提供迁移说明

### 推荐命名方式

- `v1`
- `v1.1`

或语义化版本：

- `1.0.0`

## 生命周期管理规则

推荐按以下层级定义生命周期：

### 短生命周期

- 中间训练日志
- 临时缓存

### 中生命周期

- 单实验 checkpoint
- 单实验预测文件

### 长生命周期

- 配置快照
- `metrics.json`
- `summary.csv`
- `leaderboard.csv`
- `manifest.json`

长期留存对象应优先确保格式稳定与可迁移。

## 一致性校验规则

文档与实现应至少支持以下校验：

1. `status.json` 与 `metrics.json` 是否对应
2. `manifest.json` 与 `summary.csv` 的 run 数是否一致
3. `leaderboard.csv` 是否只基于成功 run 聚合
4. `predictions.parquet` 的列是否满足任务类型要求
5. `config.yaml` 是否存在且可解析

## 演进规则

当持久化契约需要调整时，应遵循以下顺序：

1. 先在文档中声明变更
2. 增加新字段或新版本支持
3. 保留旧格式兼容窗口
4. 提供迁移策略
5. 最后再移除旧格式

不要先改代码再让文档“事后补票”。

## 与其他文档的关系

这份契约文档与以下页面配合使用：

- [Pipeline Guide](pipeline.md)：定义单实验如何产出 artifact
- [Benchmarking Guide](benchmarking.md)：定义 benchmark 如何聚合这些 artifact
- [API Contracts](api-contracts.md)：定义运行时接口与错误语义

## 需要避免的反模式

请尽量避免：

- 只保存图片，不保存结构化原始结果
- 同一文件名在不同版本代表不同语义
- 没有状态文件就尝试做恢复
- 不保留配置快照
- 结果列名随不同模型任意变化

## 一句话总结

对 RecBench 来说，最佳实践不是“先把结果写出来”，而是：

- 先定义哪些结果必须被稳定持久化
- 再定义它们的格式、状态、一致性和演进规则
- 用结构化契约让实验、benchmark 与恢复流程真正可落地
