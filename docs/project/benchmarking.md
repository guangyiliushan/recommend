---
title: Benchmarking Guide
description: 批量 Benchmark 矩阵展开、恢复调度、受控并发与聚合产物全指南
---

# Benchmarking Guide

## 概述

RecBench 的 Benchmark 层负责将多组单实验以矩阵方式组织、按恢复策略调度、隔离失败并聚合结果。它不实现训练逻辑、评估逻辑或指标计算——这些职责分别属于 Pipeline 层和评估层。

当前实现由两个模块构成：`src/recsys/pipeline/benchmark.py`（矩阵展开与调度）和 `src/recsys/pipeline/reporter.py`（结果聚合与报告生成）。

## 入口与配置

### 主入口

批量 Benchmark 的公共入口是 `run_benchmark()` 函数，接收 `BenchmarkConfig` 对象，返回 `BenchmarkResult` 对象。

### BenchmarkConfig 字段

`BenchmarkConfig` 定义了要执行的实验矩阵和调度参数：

- `benchmark_name`：Benchmark 名称，用于输出目录和产物命名
- `models`：模型注册键名列表
- `datasets`：数据集注册键名列表
- `seeds`：随机种子列表（默认为 `[42]`）
- `resume_mode`：恢复策略（支持四种模式）
- `max_concurrent_runs`：最大并发运行数（默认为 1，即串行）
- `output_root` 和 `experiment_output_dir`：输出路径控制

此外保留了四个预设字段（`experiment_preset`、`training_preset`、`evaluation_preset`、`runtime_preset`）用于引用可复用的配置预设，当前阶段作为可选占位，后续通过 Hydra 统一解析。

### CLI 入口

`scripts/run_benchmark.py` 提供命令行接口，支持三种运行模式：

- **配置文件模式**：通过 `--config` 参数指定 YAML 矩阵配置文件（如 `configs/experiment/benchmark_classical.yaml`），自动从中解析模型列表、数据集列表和 Benchmark 名称
- **命令行模式**：通过 `--models` 和 `--datasets` 直接指定参与矩阵的组件列表
- **Hydra 模式**：通过 `--hydra` 标记启用，从 `configs/config.yaml` 加载默认参数（含 `split_mode`、训练超参数等），注入每个实验组合

支持的关键 CLI 参数包括：

- `--seeds`：覆盖种子列表
- `--resume-mode`：恢复策略（`successful_skip` / `failed_only` / `unfinished_only` / `force`）
- `--max-concurrent`：控制并行执行的最大实验数
- `--output-root` / `--experiment-output-dir`：输出路径控制

## 与单实验的边界

当前代码中已明确拆分三层职责：

- **`run_experiment()`**：负责一次实验的完整执行流程（config → data → model → execute → predict → evaluate → artifact）
- **`run_benchmark()`**：负责多组实验的矩阵展开、调度和恢复
- **`Reporter.generate()`**：负责消费已完成的 `ExperimentResult` 列表，生成聚合产物

Benchmark 层不直接接触模型、数据集或训练器对象，所有实验级别的操作均通过 `run_experiment()` 间接完成。

## 实验矩阵展开

`expand_benchmark_config()` 函数将 Benchmark 配置展开为单实验配置列表。展开维度为：

```
模型数量 × 数据集数量 × 种子数量
```

每个组合生成一份独立的 `ExperimentConfig` 对象，包含 Benchmark 名称、数据集注册键名、模型注册键名、种子和输出路径。预设字段信息写入运行时配置字典，供后续统一解析。

## 恢复策略

Benchmark 层在执行前通过 `plan_runs()` 函数为每个实验组合生成执行计划（`RunPlan`），并根据恢复策略决定该组合是执行还是跳过。

### 四种恢复模式

| 模式       | 枚举值            | 行为                                                      |
| :--------- | :---------------- | :-------------------------------------------------------- |
| 成功跳过   | `successful_skip` | 跳过 status=succeeded 且产物完整的 run（默认）            |
| 仅重试失败 | `failed_only`     | 只重新执行 status=failed 的 run                           |
| 继续未完成 | `unfinished_only` | 只执行没有 status 文件或 status 为 pending/running 的 run |
| 强制全量   | `force`           | 忽略所有已有状态，重新执行全部组合                        |

### 恢复判断依据

`check_run_completion()` 函数检查一个实验目录是否已完成，判断依据为：

1. `status.json` 文件存在且其中 `status` 字段为 `succeeded`
2. `metrics.json` 文件存在
3. 可选：`config.yaml` 中的 `config_hash` 与当前配置哈希一致（若配置已变更则视为需要重新运行）

满足全部条件时视为已完成，根据恢复策略决定跳过或重试。

## 调度执行

Benchmark 层支持两种执行方式。

### 串行执行

默认方式（`max_concurrent_runs=1`），通过 `execute_runs()` 函数依次处理每个执行计划。对于标记为跳过的 plan，从已有产物文件重建 `ExperimentResult`（读取 `status.json` 和 `metrics.json`）；对于需要执行的 plan，调用 `run_experiment()` 并捕获异常。每个 run 的失败不会阻塞后续 run。

### 受控并发执行

通过 `execute_runs_parallel()` 函数实现，使用 Python 标准库的 `ThreadPoolExecutor`，并发粒度在实验级别（每个 run 一个执行单元，不做更细粒度的 batch 或 step 级调度）。并发执行时自动设置环境变量 `RECSYS_BENCHMARK_MODE=1`，触发所有子进程的进度条静默模式，避免多进程进度条交错。

### 进度显示

执行过程中通过 tqdm 显示进度条（仅在 tqdm 可用且非静默模式下），显示当前完成的 run 数量和总 run 数量。tqdm 为可选依赖，缺失时静默回退为无进度条模式。

## 失败隔离

单个 run 的失败不会阻塞整批 Benchmark。执行过程中发生的原生异常被捕获后封装为带结构化错误的 `ExperimentResult`（状态为 `FAILED`，包含错误码、失败阶段和错误消息），后续由 Reporter 统一聚合到 `failures.csv` 中。

被跳过的 run 以 `SKIPPED` 状态出现，Benchmark 试图从已有的 `status.json` 和 `metrics.json` 重建其指标。

## 结果聚合与 Reporter

Benchmark 执行完毕后，`run_benchmark()` 委托 Reporter 生成聚合产物。Reporter 的输入是实验结果的纯数据列表，不依赖任何运行时状态，可离线重建。同时 Benchmark 自身还额外写出 `manifest.json` 作为所有产物的索引清单。

### 聚合产物清单

| 产物文件          | 内容说明                                                                    |
| :---------------- | :-------------------------------------------------------------------------- |
| `summary.csv`     | 每轮实验一行，包含运行 ID、数据集、模型、种子、状态、主指标值和所有评估指标 |
| `leaderboard.csv` | 按模型×数据集×主指标聚合，仅包含成功的运行，含均值、标准差和排名            |
| `failures.csv`    | 仅包含失败的运行，含失败阶段、错误码和错误信息                              |
| `manifest.json`   | Benchmark 元信息索引，记录名称、创建时间、所有运行的 ID、矩阵维度和产物路径 |
| `trend.csv`       | 同一模型×数据集组合下不同种子的指标变化趋势（v2）                           |
| `stability.csv`   | 跨种子的指标稳定性分析，含均值、标准差和变异系数（v2）                      |
| `report.html`     | 交互式 HTML 摘要页，内联 JavaScript 排序，无需外部依赖                      |

### Reporter 数据提取逻辑

- **summary 行提取**：从 `ExperimentResult` 中提取运行 ID、数据集名、模型名、种子、状态、主指标值和完整指标字典
- **leaderboard 聚合**：筛选成功的 run，按模型×数据集分组，计算主指标的均值和标准差，按均值降序排列并赋予排名
- **failure 行提取**：只提取状态为 `FAILED` 的 run，从结构化错误中获取失败阶段、错误码和消息
- **trend 行提取**：成功 run 的模型×数据集组合按种子展开，含指标值、耗时、数据规模

### 独立报告生成

`scripts/generate_report.py` 可独立于 Benchmark 管线运行。它从任意 `summary.csv` 加载数据，按模型分组计算耗时和指标的均值、标准差、最小值和最大值，生成包含速度对比表（含加速比）、准确性对比表和内存使用比较表三部分的 Markdown 性能对比报告。

## Benchmark 结果结构

`run_benchmark()` 返回的 `BenchmarkResult` 包含以下关键字段：

- `benchmark_name`：Benchmark 名称
- `status`：整体状态（`succeeded`：全部成功；`partial_success`：部分成功；`failed`：全部失败）
- `runs`：所有 `ExperimentResult` 的列表（可通过 `succeeded_runs` 和 `failed_runs` 属性筛选）
- `summary_path`：summary.csv 的绝对路径
- `leaderboard_path`：leaderboard.csv 的绝对路径
- `failures_path`：failures.csv 的绝对路径
- `manifest_path`：manifest.json 的绝对路径
- `report_path`：report.html 的绝对路径（若生成）
- `metadata`：包含总运行数、成功数、失败数和恢复模式等运行时元信息

## Benchmark 输出目录结构

```
outputs/benchmarks/{benchmark_name}/
├── manifest.json          # 所有产物索引
├── summary.csv            # 逐 run 展平表
├── leaderboard.csv        # 聚合排序表
├── failures.csv           # 失败排查表
├── trend.csv              # 趋势分析（v2）
├── stability.csv          # 稳定性分析（v2）
└── report.html            # 交互式摘要页
```

每个 run 的独立产物仍位于 `outputs/runs/{run_id}/` 下。

## 数据管线性能 Benchmark

除模型 Benchmark 外，`scripts/benchmark_data_pipeline.py` 提供了一个独立的存储管线性能基准测试工具。它生成推荐风格的合成数据集（可配置行数、用户数、物品数和特征列数），然后对所有存储格式×压缩算法×计算后端的组合矩阵进行读写性能测量。测量项包括：

- 读取耗时和写入耗时
- 输出文件大小和压缩比
- 缓存命中耗时（重复运行的加速效果）
- 峰值内存使用（通过 psutil 采集 RSS）

输出产物包括 `summary.csv`（逐组合展平）、`formats.csv`（按格式×压缩聚合）、`backends.csv`（按后端聚合）、`report.md`（人类可读报告）和 `report.json`（机器可读报告）。

## 当前限制

- `run_benchmark()` 依赖 `run_experiment()`，后者已同时支持训练型和非训练型路径，但可运行模型范围仍受限于已注册激活的 itemcf 和 hyformer
- Benchmark 预设名（`experiment_preset` 等）当前只作为字段值传递，尚未形成完整的 Hydra 统一配置解析体系——Hydra 模式下通过 `_run_benchmark_hydra_mode()` 特殊路径注入默认参数
- `configs/experiment/` 下的 benchmark yaml 文件已清理：三个活跃配置仅引用已注册模型和数据集，四个 planned 配置内容已被注释
- CLI 的 `--hydra` 模式在 benchmark 中走独立执行路径（`_run_benchmark_hydra_mode()`），尚未与 `run_benchmark()` 的 Reporter 聚合流程完全打通
- 并发执行的粒度在实验级别，不支持更细粒度的资源调度或优先级控制

## 未来展望

- **Hydra 模式与 Reporter 的完整融合**：当前 `--hydra` 模式走独立路径且绕过 Reporter 聚合，未来应将 Hydra 的默认参数注入统一到 `expand_benchmark_config()` 中，保持单一聚合出口
- **配置文件驱动的预设解析**：`BenchmarkConfig` 中的四个 preset 字段可解析为对应的完整训练/评估/运行时参数子树
- **LaTeX 表格导出与统计显著性检验**：Reporter v3 规划的能力，用于学术论文级别的结果输出
- **更细粒度的失败分析**：当前 failure 仅记录 run 级别的失败，未来可关联到具体的评估阶段（如某个指标计算失败而不是整个 run 失败）

## 参考

- [Benchmark 源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/pipeline/benchmark.py)
- [Reporter 源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/pipeline/reporter.py)
- [Pipeline 文档](pipeline.md)
- [配置系统文档](../concepts/configuration.md)
