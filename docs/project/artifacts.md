---
title: Persistence Contracts
description: 单实验与批量 Benchmark 的真实产物协议 — 文件、格式、字段语义与恢复逻辑
---

# Persistence Contracts

## 概述

RecBench 当前已经将配置快照、状态文件、指标结果、预测文件与 Benchmark 聚合表落成真实的文件输出。本页描述这些产物的具体格式、字段语义、存储路径以及它们在恢复逻辑中的角色。所有描述基于 Pipeline 层（`experiment.py`、`benchmark.py`）和 Reporter 层（`reporter.py`）的实际写入逻辑。

## 输出根目录

所有产物分为两层目录结构：

```
outputs/
├── experiments/                # 单实验原子结果
│   └── {run_id}/
└── benchmarks/                 # 批量 Benchmark 聚合结果
    └── {benchmark_name}/
```

此外，数据管线性能基准测试工具使用独立输出目录：

```
outputs/data-benchmarks/{run_id}/    # 数据管线性能基准
```

## 单实验产物

### config.yaml — 配置快照

配置快照在实验开始时（CONFIG 阶段）写入，内容包含冻结后的完整配置树以及三条元信息：`_meta.schema_version`（快照结构版本号）、`_meta.resolved_at`（解析完成时间 ISO 8601 格式）和 `_meta.config_hash`（配置内容的 MD5 短哈希）。

**路径**：`outputs/runs/{run_id}/config.yaml`

**恢复角色**：Benchmark 的恢复判断会读取此文件中的 `config_hash` 与当前请求的配置哈希对比。若两者不一致，即使 `status.json` 为 `succeeded`，该 run 也会被标记为需要重新执行。

### status.json — 运行状态

状态文件在实验开始时写入初始状态（`running`），在结束时更新为最终状态（`succeeded` 或 `failed`）。

**路径**：`outputs/runs/{run_id}/status.json`

**状态值**：`pending`（初始）、`running`（执行中）、`succeeded`（成功完成）、`failed`（执行失败）、`skipped`（被 Benchmark 跳过）。

**关键字段**：`run_id`（运行标识）、`status`（当前状态）、`started_at` 和 `finished_at`（ISO 8601 时间戳）、`dataset` 和 `model`（注册键名）、`seed`（随机种子）、`primary_metric` 和 `primary_metric_value`（主指标名称与数值，仅在成功时填充）、`resume_supported`（是否支持恢复继续）。失败时包含 `error` 字段，内嵌错误码（`code`）、失败阶段（`phase`）、错误消息（`message`）和可选修复提示（`hint`）。

**恢复角色**：Benchmark 的四种恢复策略均以此为第一判断依据。`failed_only` 模式只重试 status 为 `failed` 的 run；`unfinished_only` 模式查询 status 不是 `succeeded` 或 `failed` 的 run。

### metrics.json — 评估结果

评估结果在 ARTIFACT 阶段写入，内容来自 `EvaluationResult` 的序列化输出。

**路径**：`outputs/runs/{run_id}/metrics.json`

**关键字段**：`summary_metrics`（主汇总指标字典）、`task_metrics`（分任务指标，multitask 场景每个任务头独立一组）、`group_metrics`（分段诊断结果）、`curve_artifacts`（曲线数据引用，numpy 数组已转为列表）、`metadata`（评估元信息）、`warnings` 和 `errors`（评估过程中的警告与错误）。

**恢复角色**：Benchmark 恢复判断中，`metrics.json` 是否存在是完成性检查的必要条件之一。跳过 run 的指标重建也从此文件读取。

### predictions.parquet — 预测明细

预测结果以 Parquet 列式格式保存，列结构根据任务类型自动调整。

**路径**：`outputs/runs/{run_id}/predictions.parquet`

**列结构按任务类型**：

- **pointwise**：每样本一行，包含 `sample_id`（样本序号）、`y_score`（预测分数）、`y_true`（真实标签）、`y_pred`（离散预测值，可选）、`group_id`（分组标识，可选）、`split`（固定为 `"test"`）
- **ranking**：每分组一行，包含 `group_id`（分组标识）、`split`（固定为 `"test"`），每个候选位置展开为 `score_{j}` 列，以及 `y_true_count`（正样本计数）和 `num_candidates`（候选总数）
- **multitask**：每任务每样本一行，包含 `task_name`（任务名）、`y_score`、`y_true`、`y_mask`（有效标记）和 `split`

若 bundle 数据无法展开为表格格式（数据为空或结构不兼容），写入过程会静默跳过，不阻塞实验流程。

### curves/ — 曲线数据

曲线目录在 ARTIFACT 阶段写入，每个曲线一个 JSON 文件。

**路径**：`outputs/runs/{run_id}/curves/`

**典型内容**：
- `roc_curve.json`：ROC 曲线的 x/y 数据点数组和标签
- `pr_curve.json`：PR 曲线的 x/y 数据点数组和标签
- `threshold_sweep.json`：阈值扫描的指标变化数据
- `ndcg_at_k.json` 等：Ranking 指标按 K 值展开的数据

所有曲线数据以原始结构化 JSON 为主产物，图片不是第一公民。numpy 数组在序列化前转为 Python 列表。

### checkpoints/ — 模型检查点

检查点目录仅训练型路径产生（如 HyFormer），由 PyTorch Lightning 的 `ModelCheckpoint` 回调自动管理。

**路径**：`outputs/runs/{run_id}/checkpoints/`

**内容**：最佳模型权重文件（基于监控指标 `val_loss`，文件名格式为 `best-{epoch}-{monitor}.ckpt`）和可选的最后一个 epoch 的检查点文件。检查点目录由 `build_callbacks()` 在训练开始时创建。

### logs/ — 运行日志

日志目录由 `setup_logging()` 函数在 BOOTSTRAP 阶段初始化，贯穿实验全过程。

**路径**：`outputs/runs/{run_id}/logs/`

**文件清单**：

| 文件         | 级别     | 说明                                        |
| :----------- | :------- | :------------------------------------------ |
| `run.log`    | DEBUG    | 完整运行日志，含结构化上下文（run_id 嵌入） |
| `stderr.log` | WARNING+ | 错误日志，仅记录 WARNING 及以上级别         |

日志文件配置了 100 MB 轮转和 30 天保留策略。若启用了 TensorBoard 或 WandB 追踪器，日志目录下还会产生对应的追踪器子目录。

## 批量 Benchmark 产物

### manifest.json — 产物索引

Benchmark 元信息索引，由 `run_benchmark()` 在收集完所有实验结果后写入，是 Benchmark 目录的入口文件。

**路径**：`outputs/benchmarks/{benchmark_name}/manifest.json`

**关键字段**：`benchmark_name`（Benchmark 名称）、`created_at`（创建时间 ISO 8601）、`runs`（所有运行 ID 列表）、`models`、`datasets`、`seeds`（矩阵维度）、`total`、`succeeded`、`failed`（汇总计数）、`summary_path`、`failures_path`、`leaderboard_path`、`report_path`（各产物路径引用）。

### summary.csv — 逐 run 展平表

由 Reporter 的 `extract_summary_row()` 从每个 `ExperimentResult` 中提取。

**路径**：`outputs/benchmarks/{benchmark_name}/summary.csv`

**列内容**：`run_id`、`dataset`、`model`、`seed`、`status`、`primary_metric`（主指标值）、以及所有评估指标的值列。每行对应一次实验运行。`generate_report.py` 脚本可独立消费此文件生成 Markdown 性能对比报告。

### leaderboard.csv — 聚合排序表

由 Reporter 的 `aggregate_leaderboard()` 生成，仅基于状态为 `succeeded` 的运行。

**路径**：`outputs/benchmarks/{benchmark_name}/leaderboard.csv`

**列内容**：`model`、`dataset`、`primary_metric_name`、`mean`（均值）、`std`（标准差）、`rank`（按均值降序排列的名次）、`num_runs`（参与的运行数）。用于快速比较同一模型在不同数据集上的平均表现和稳定性。

### failures.csv — 失败排查表

由 Reporter 的 `extract_failure_row()` 从状态为 `FAILED` 的 `ExperimentResult` 中提取。

**路径**：`outputs/benchmarks/{benchmark_name}/failures.csv`

**列内容**：`run_id`、`dataset`、`model`、`seed`、`phase`（失败的实验阶段）、`error_code`（错误码）、`error_message`（错误消息）。用于快速定位失败组合和失败原因。

### trend.csv — 趋势分析（v2）

由 Reporter 的 v2 版本新增，展示同一模型×数据集组合下不同种子运行的指标变化趋势。

**路径**：`outputs/benchmarks/{benchmark_name}/trend.csv`

**列内容**：`model`、`dataset`、`seed`、`task_type`、`primary_metric_name`、`primary_metric_value`、`duration_seconds`、`num_users`、`num_items`。用于观察不同随机初始化下的性能波动。

### stability.csv — 稳定性分析（v2）

由 Reporter 的 v2 版本新增，跨种子聚合的指标稳定性统计。

**路径**：`outputs/benchmarks/{benchmark_name}/stability.csv`

**列内容**：`model`、`dataset`、`task_type`、`primary_metric_name`、`mean`（均值）、`std`（标准差）、`cv`（变异系数，标准差除以均值的绝对值）、`min_val`、`max_val`、`num_seeds`。变异系数越小说明模型在随机性下越稳定。

### report.html — 交互式摘要页

由 Reporter 生成的供人工浏览的 HTML 报告，内联 JavaScript 实现表格排序，无需外部 Web 服务器或依赖。

**路径**：`outputs/benchmarks/{benchmark_name}/report.html`

## 数据管线性能 Benchmark 产物

`scripts/benchmark_data_pipeline.py` 独立于模型 Benchmark 运行，产出以下文件：

| 产物文件       | 内容说明                                                                   |
| :------------- | :------------------------------------------------------------------------- |
| `summary.csv`  | 逐组合的详细测量（格式/压缩/后端/行数/耗时/文件大小/压缩比/缓存命中/错误） |
| `formats.csv`  | 按格式×压缩聚合的均值和最优值                                              |
| `backends.csv` | 按后端聚合的均值统计和错误计数                                             |
| `report.md`    | 人类可读的 Markdown 对比报告                                               |
| `report.json`  | 机器可读的完整 JSON 报告                                                   |

## 恢复与一致性逻辑

当前 Benchmark 的恢复判断依赖以下产物的一致性检查：

1. `status.json` 文件是否存在且可解析
2. `metrics.json` 文件是否存在
3. `status.json` 中的 `status` 字段是否等于 `"succeeded"`
4. `config.yaml` 中的 `config_hash` 是否与当前请求的配置哈希一致（可选检查，配置变更时强制重跑）

这意味着产物协议不是仅供人工查看的参考文件，而是直接参与运行控制的核心组件。配置变更（如修改了 `split_ratios` 或 `evaluation.metrics`）会被哈希对比捕获并触发重新执行。

## 产物一致性校验建议

更新文档或调试产物时，建议执行以下检查：

1. `status.json` 与 `metrics.json` 的状态是否对应（成功 run 必须有两者）
2. `manifest.json` 中的 run 数是否与 `summary.csv` 的行数一致
3. `leaderboard.csv` 是否仅基于状态为 `succeeded` 的 run 聚合
4. `config.yaml` 是否可被 YAML 解析器重新解析
5. `predictions.parquet` 的列结构与 `task_type` 语义是否匹配（pointwise 应有 `y_score`/`y_true` 列，ranking 应有 `group_id` 列）
6. `failures.csv` 中的 `error_code` 是否与 Pipeline 层的 `ExperimentError.code` 枚举值对齐

## 未来展望

以下产物能力已在设计规划中：

- **LaTeX 表格导出**：Reporter v3 规划的学术论文级别的指标表格输出，格式适配常见的 LaTeX 模板（booktabs 风格）
- **统计显著性注释**：在 leaderboard 或 LaTeX 表格中自动标注统计检验结果（如上标 `*`、`†` 标记显著性水平）
- **predictions.parquet 的 schema version 字段**：当前版本信息通过 `ExperimentRunMeta.prediction_schema_version` 记录在内存中，但未嵌入 Parquet 文件的元数据块
- **曲线数据的通用格式版本标记**：当前 curves/ 下的 JSON 文件未包含统一的 schema 版本字段，跨版本解析时需额外兼容处理

## 参考

- [Pipeline 源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/pipeline/experiment.py)
- [Benchmark 源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/pipeline/benchmark.py)
- [Reporter 源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/pipeline/reporter.py)
- [Pipeline 文档](pipeline.md)
- [Benchmark 文档](benchmarking.md)
