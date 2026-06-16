---
title: Benchmarking Guide
description: 单实验、批量 benchmark、结果目录与失败恢复策略
---

# Benchmarking Guide

## 目标

RecBench 的 benchmark 层不应只是“循环跑一堆模型”，而应满足以下目标：

- 区分单实验与批量 benchmark 的职责
- 让模型、数据集、指标组合可复用、可追踪
- 让结果目录稳定且便于比较
- 让失败任务隔离，不阻塞整批实验
- 让中断后能够恢复，而不是整批重跑

当前仓库已经有：

- `src/recsys/pipeline/benchmark.py`
- `src/recsys/pipeline/reporter.py`
- `scripts/run_benchmark.py`
- `configs/experiment/*.yaml`

但这些文件目前仍以骨架为主，因此这份文档重点约束“应该怎么设计和落地”。

## 先区分单实验与批量 benchmark

这是当前最重要的边界。

### 单实验

单实验的目标是：

- 跑通一个确定的配置组合
- 得到一次完整训练或推理结果
- 输出该次实验的模型、指标、日志和 artifact

典型例子：

- 一个模型 + 一个数据集 + 一组固定超参数

### 批量 benchmark

批量 benchmark 的目标是：

- 组织多个单实验组合
- 收集并聚合它们的结果
- 形成 leaderboard、对比表和汇总报告

典型例子：

- 多个模型 × 一个数据集
- 一个模型 × 多个数据集
- 多个模型 × 多个数据集 × 多个种子

### 为什么必须拆开

如果不拆这两个层次，后面很容易出现：

- benchmark runner 知道太多训练细节
- 单实验逻辑无法单独测试
- 报错和恢复粒度只能按整批处理

因此最佳实践是：

- `pipeline.py` 或 `experiment.py` 负责单实验
- `benchmark.py` 负责批量调度与聚合

## Benchmark 的最小职责

一个成熟的 benchmark runner 至少应负责：

- 展开实验矩阵
- 为每个组合生成稳定的 run 标识
- 调用单实验入口
- 捕获成功与失败状态
- 按统一格式写出结果摘要
- 生成聚合报告

它不应直接负责：

- 具体模型训练逻辑
- 数据预处理细节
- 评估指标内部实现

## 推荐的组合展开方式

benchmark 配置更适合描述“实验清单”，而不是复制一整份完整配置。

推荐由 `configs/experiment/*.yaml` 描述：

- 要跑哪些模型
- 要跑哪些数据集
- 要跑哪些 seed
- 使用哪组训练/evaluation/runtime 预设

例如，benchmark 配置更像：

- `models = [itemcf, mf, deepfm]`
- `datasets = [taac2026]`
- `seeds = [2026, 2027, 2028]`

而不是在 benchmark 文件里重复写一整套 optimizer、scheduler、metrics 细节。

## 推荐的执行单元

benchmark 的基本执行单元应该是：

- 一个 fully resolved experiment config

也就是说，在真正执行前，每个组合都应先被解析成一份完整配置快照，包含：

- 数据集选择
- 模型选择
- 训练参数
- 评估参数
- runtime 参数
- 输出路径

这样每个 run 才能独立复现。

## 结果目录应该怎么设计

### 总体原则

结果目录必须满足：

- 人类可读
- 程序可扫描
- 中断后可恢复
- 聚合时不依赖猜测文件名

### 推荐目录结构

建议长期统一到如下结构：

```text
outputs/
|-- experiments/
|   `-- {run_id}/
|       |-- config.yaml
|       |-- status.json
|       |-- metrics.json
|       |-- predictions.parquet
|       |-- curves/
|       |-- checkpoints/
|       `-- logs/
`-- benchmarks/
    `-- {benchmark_name}/
        |-- manifest.json
        |-- summary.csv
        |-- leaderboard.csv
        |-- failures.csv
        |-- report.html
        `-- runs/
            |-- {run_id_1}.link
            |-- {run_id_2}.link
            `-- ...
```

### 目录分层解释

- `experiments/`: 保存每一次独立实验的原子结果
- `benchmarks/`: 保存整批 benchmark 的聚合视图

这个分层很重要，因为：

- 单实验可以独立调试
- benchmark 可以引用多个已有 run
- 后续做 resume 时不需要重新解释历史目录

## `run_id` 应该怎么生成

推荐 `run_id` 由稳定字段组成，而不是只靠时间戳。

建议至少包含：

- benchmark 或 experiment 名称
- dataset 名
- model 名
- seed
- 可选的配置 hash

例如：

- `taac2026__deepfm__seed2026`
- `benchmark_classical__itemcf__taac2025__seed42`

如果担心同名覆盖，可以附加短 hash：

- `taac2026__deepfm__seed2026__a1b2c3`

## 单实验结果里最少要保存什么

为了让 benchmark 可恢复、可比较，单实验目录最少应保存：

- `config.yaml`: 完整解析后的配置快照
- `status.json`: 当前 run 状态
- `metrics.json`: 结构化指标结果
- `logs/`: 训练与运行日志

如果任务允许，建议进一步保存：

- `predictions.parquet`
- `curves/`
- `checkpoints/`
- `artifacts/`

## `status.json` 的推荐字段

失败恢复最关键的是状态文件。

推荐至少记录：

- `run_id`
- `status`
- `started_at`
- `finished_at`
- `dataset`
- `model`
- `seed`
- `error_type`
- `error_message`
- `resume_supported`

其中 `status` 建议统一为：

- `pending`
- `running`
- `succeeded`
- `failed`
- `skipped`

## 批量 benchmark 的聚合结果

benchmark 完成后，最少应输出：

- `manifest.json`
- `summary.csv`
- `leaderboard.csv`
- `failures.csv`

### `manifest.json`

它是整批实验的索引文件，建议记录：

- benchmark 名称
- 参与的 runs
- 配置快照位置
- 聚合时间
- 成功/失败数量

### `summary.csv`

每一行代表一次单实验，建议字段包括：

- `run_id`
- `dataset`
- `model`
- `seed`
- `status`
- `primary_metric`
- 主要指标列

### `leaderboard.csv`

它用于排序展示，通常基于主指标排序，并可附加：

- mean
- std
- rank

### `failures.csv`

它用于失败排查，建议记录：

- `run_id`
- `dataset`
- `model`
- `seed`
- `error_type`
- `error_message`

## 失败隔离策略

这是 benchmark 系统能否长期使用的关键。

### 推荐原则

- 单个 run 失败不能阻塞整批 benchmark
- 失败必须有结构化记录
- 失败信息必须能定位到具体组合

### 当前仓库已经有的方向

`benchmark.py` 的骨架注释已经明确写了：

- failed experiments don't block others

这个方向是正确的，后续实现时应保留。

### 推荐处理方式

每个实验组合在 benchmark 中都应包裹统一的错误边界：

- 捕获异常
- 写入该 run 的 `status.json`
- 将失败摘要写入 `failures.csv`
- 继续下一个组合

不要只在控制台打印报错然后继续，否则后面很难统计和恢复。

## 失败恢复策略

### 最重要的原则

恢复不应靠“人工判断哪些目录看起来没跑完”，而应依赖明确状态。

### 推荐恢复语义

benchmark runner 启动时应支持：

- 跳过已成功的 runs
- 重试失败的 runs
- 继续未完成的 runs
- 强制重跑所有 runs

建议至少支持以下模式：

- `resume=successful_skip`
- `resume=failed_only`
- `resume=unfinished_only`
- `resume=force`

### 如何判断一个 run 可恢复

推荐按以下顺序判断：

1. 是否存在 `status.json`
2. `status` 是否为 `succeeded`
3. 关键产物是否齐全，如 `metrics.json`
4. 配置 hash 是否匹配当前请求

只有同时满足条件，才真正视为“可跳过”。

## 中断恢复与幂等性

benchmark runner 必须尽量幂等。

这意味着：

- 重复执行同一个 benchmark，不应生成一堆不可区分的重复目录
- 已成功的 run 默认可复用
- 失败的 run 默认可追踪

如果没有幂等性，后面结果目录会很快变成垃圾堆。

## 并发策略怎么选

当前 `benchmark.py` 的说明提到：

- Parallel execution

但最佳实践是先保证稳定，再扩并发。

### 推荐阶段性策略

第一阶段：

- 串行执行 benchmark
- 优先把状态、结果目录和失败恢复做稳

第二阶段：

- 引入受控并发
- 并发粒度放在“实验组合级别”

第三阶段：

- 根据设备资源进一步做 GPU-aware 调度

### 为什么不要一开始就高并发

因为推荐实验通常涉及：

- GPU 内存争抢
- 数据加载冲突
- checkpoint 写入冲突
- 日志目录冲突

这些问题在运行时主干未稳定前会放大复杂度。

## 单实验与 benchmark 的接口契约

benchmark runner 最好只依赖单实验统一接口，例如：

- 输入：一份完整 experiment config
- 输出：一份结构化 experiment result

推荐 experiment result 至少包含：

- `run_id`
- `status`
- `summary_metrics`
- `artifact_paths`
- `error`

这样 benchmark 层就不需要了解训练内部细节。

## 汇总与统计的边界

批量 benchmark 不只是“把结果拼表格”，还应区分：

- 原始 run 结果
- 聚合统计结果
- 排行榜视图
- 可视化报告

推荐分层：

- `metrics.json`: 单次 run 原始结果
- `summary.csv`: 全部 run 展平结果
- `leaderboard.csv`: 排序后的聚合结果
- `report.html`: 可读报告

## 当前仓库的落地建议

结合现有骨架，我建议按以下顺序推进：

1. 先稳定 `run_experiment()` 的结构化输出
2. 再实现 `run_benchmark()` 的矩阵展开与错误隔离
3. 再实现 `Reporter` 的 CSV 与 leaderboard 生成
4. 最后再加入并发和高级可视化

## 需要避免的反模式

请尽量避免：

- benchmark 文件重复拷贝整套训练配置
- 没有 `status.json` 就尝试恢复
- 每次运行都重新生成不可追踪目录
- 单个 run 失败后直接中止整批实验
- 结果目录里只留图片，不留结构化原始结果
- benchmark runner 直接耦合模型或 trainer 内部实现

## 一句话总结

对 RecBench 来说，最佳实践不是“尽快把所有组合都跑起来”，而是：

- 用单实验作为原子执行单元
- 用 benchmark 组织实验矩阵与聚合结果
- 用稳定目录结构保存 run 状态与 artifact
- 用显式恢复策略保证长批任务可持续运行
