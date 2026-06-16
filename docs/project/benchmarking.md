---
title: Benchmarking Guide
description: 当前批量 Benchmark 实现、恢复策略与聚合产物
---

# Benchmarking Guide

## 目标

RecBench 的 Benchmark 层负责组织多个单实验、隔离失败、支持恢复，并把结果聚合成可比较的 CSV 与 HTML 报告。

当前仓库中，这一层已经由 `src/recsys/pipeline/benchmark.py` 和 `src/recsys/pipeline/reporter.py` 落地实现。

## 当前入口

批量 Benchmark 的公共入口是：

```python
run_benchmark(bench_cfg: BenchmarkConfig) -> BenchmarkResult
```

## 单实验与批量 Benchmark 的边界

当前代码已经明确拆分：

- `run_experiment()`：负责一次实验
- `run_benchmark()`：负责多次实验矩阵展开与调度
- `Reporter.generate()`：负责聚合已完成的实验结果

这三个层次在文档中也必须保持清晰边界，避免把 Benchmark 写成“包办训练、评估、调度、聚合的一体化黑盒”。

## 当前已实现能力

### 1. 实验矩阵展开

`BenchmarkConfig` 当前支持：

- `benchmark_name`
- `models`
- `datasets`
- `seeds`
- `experiment_preset`
- `training_preset`
- `evaluation_preset`
- `runtime_preset`
- `resume_mode`
- `max_concurrent_runs`
- `output_root`
- `experiment_output_dir`

矩阵展开规则是：

- `models × datasets × seeds`

每个组合会生成一份独立的 `ExperimentConfig`。

### 2. 恢复策略

当前已经实现四种恢复模式：

- `successful_skip`
- `failed_only`
- `unfinished_only`
- `force`

恢复判断会结合：

- `status.json`
- `metrics.json`
- `config.yaml`
- `config_hash`

这意味着恢复逻辑当前已经是代码实现，不再只是文档约定。

### 3. 调度执行

当前 Benchmark 层支持两种执行方式：

- 串行执行
- 受控并发执行

并发粒度是“实验组合级别”，而不是更细粒度的 step 或 batch 级别调度。

### 4. 失败隔离

当前一个 run 失败不会阻塞整批 Benchmark。

失败会被收集到结构化 `ExperimentResult` 中，后续再由 `Reporter` 聚合为 `failures.csv`。

### 5. 聚合报告

Reporter 当前已生成：

- `summary.csv`
- `leaderboard.csv`
- `failures.csv`
- `trend.csv`
- `stability.csv`
- `report.html`

Benchmark 层自身还会额外写出：

- `manifest.json`

## 当前目录结构

当前文档应围绕如下目录结构说明：

```text
outputs/
|-- experiments/
|   `-- {run_id}/
|       |-- config.yaml
|       |-- status.json
|       |-- metrics.json
|       |-- predictions.parquet
|       |-- curves/
|       `-- logs/
`-- benchmarks/
    `-- {benchmark_name}/
        |-- manifest.json
        |-- summary.csv
        |-- leaderboard.csv
        |-- failures.csv
        |-- trend.csv
        |-- stability.csv
        `-- report.html
```

## `run_id` 与执行单元

Benchmark 当前的原子执行单元是“一份冻结后的 `ExperimentConfig`”。

每个执行单元都会：

- 独立拥有 `run_id`
- 独立拥有实验目录
- 独立返回 `ExperimentResult`

这样 Benchmark 层只需要关心调度、恢复和聚合，而不必深入训练或评估内部实现。

## 当前返回对象

`BenchmarkResult` 当前重点字段包括：

- `benchmark_name`
- `status`
- `runs`
- `summary_path`
- `leaderboard_path`
- `failures_path`
- `manifest_path`
- `report_path`
- `metadata`

其中 `status` 可能为：

- `succeeded`
- `partial_success`
- `failed`

## 当前恢复语义

### `successful_skip`

- 已成功的 run 默认跳过
- 适合增量补跑未完成实验

### `failed_only`

- 仅重试失败 run
- 已成功和未开始的 run 会被跳过

### `unfinished_only`

- 仅继续未完成 run
- 已成功和已失败 run 会被跳过

### `force`

- 强制重跑全部组合

## 当前 Reporter 产物说明

### `summary.csv`

逐 run 展平表，适合查看每次实验的：

- 模型
- 数据集
- seed
- status
- 主指标与摘要指标

### `leaderboard.csv`

聚合排序表，适合查看：

- 同一模型在同一数据集上的平均表现
- 均值、方差与排名

### `failures.csv`

失败排查表，适合定位：

- 哪个组合失败
- 失败阶段
- 错误码
- 错误信息

### `trend.csv`

趋势表，适合分析：

- 多 seed 实验的逐 run 变化
- 主指标随 run 的波动

### `stability.csv`

稳定性表，适合分析：

- 多 seed 聚合的均值
- 标准差
- 变异系数

### `report.html`

面向人工浏览的聚合摘要页。

## 最小示例

```python
from recsys.pipeline.benchmark import BenchmarkConfig, ResumeMode, run_benchmark

bench_cfg = BenchmarkConfig(
    benchmark_name="demo_benchmark",
    models=["itemcf"],
    datasets=["taac2026_data_sample"],
    seeds=[42, 43],
    resume_mode=ResumeMode.SUCCESSFUL_SKIP,
    max_concurrent_runs=1,
    output_root="./outputs",
    experiment_output_dir="./outputs/experiments",
)

result = run_benchmark(bench_cfg)
print(result.status)
print(result.summary_path)
print(result.report_path)
```

## 当前限制

文档必须明确保留下面这些限制：

- `run_benchmark()` 依赖 `run_experiment()`，而后者的训练型路径尚未完成
- benchmark 预设名当前只是被写入运行配置，还没有形成完整的统一配置解析体系
- 当前仓库中的部分 `configs/experiment/*.yaml` 仍引用了未实现模型或未注册数据集，不能直接当作“现成可运行配置”宣传
- `scripts/run_benchmark.py` 仍是占位脚本，不应作为稳定 CLI 入口写入文档

## 当前最重要的结论

RecBench 的 Benchmark 层已经具备矩阵展开、恢复模式、失败隔离、受控并发和结果聚合能力。当前最需要文档诚实表达的，不是“Benchmark 还没实现”，而是“Benchmark 主干已实现，但其可运行范围仍受单实验可运行模型范围限制”。
