---
title: Configuration Guide
description: Hydra + dataclass 设计、配置目录组织和覆盖方式最佳实践
---

# Configuration Guide

## 目标

RecBench 的配置系统不应只是“把参数写进 YAML”，而应满足以下目标：

- 支持按组件组合配置，而不是维护一个越来越大的单文件
- 支持命令行覆盖，方便实验对比
- 提供结构化校验，尽量在配置阶段而不是运行时发现错误
- 让模型、数据集、训练器和评估逻辑各自拥有清晰的配置边界
- 能被单实验和批量 benchmark 共用

当前仓库已经有 `configs/` 目录和 `hydra-core` 依赖，但 `src/recsys/utils/config.py` 仍是骨架，因此这里描述的是推荐设计方向与落地规范。

## 当前状态

当前可见配置结构：

- `configs/config.yaml`: 主配置文件
- `configs/dataset/`: 数据集配置目录
- `configs/experiment/`: benchmark 场景配置目录
- `configs/model/`: 模型配置目录，当前仍不完整

当前主配置已经在使用 Hydra 的 `defaults` 列表，但仍有几个明显问题：

- 注释中的运行入口还是旧路径
- 配置大多仍是“单文件集中定义”
- 尚未引入 dataclass 作为结构化 schema
- `model.params` 仍是开放字典，缺少类型约束

因此，下一步最佳实践不是继续堆更多 YAML，而是建立 “YAML 组合 + dataclass schema” 的双层结构。

## 推荐设计

### 1. 总体原则

推荐采用下面的分工：

- YAML: 负责组合、选择和覆盖
- dataclass: 负责类型、默认值和校验边界
- Config Store: 负责把 dataclass 注册为 Hydra 可识别的 schema

换句话说，YAML 决定“这次实验选什么”，dataclass 决定“这些配置项是否合法”。

这种方式的好处是：

- YAML 保持可读性和组合能力
- dataclass 提供 IDE 友好性和类型检查
- CLI override 仍然保留 Hydra 的灵活性
- 错误可以在配置合成阶段尽早暴露

## 推荐目录结构

建议逐步演进到如下结构：

```text
configs/
|-- config.yaml
|-- experiment/
|   |-- single.yaml
|   |-- benchmark_classical.yaml
|   `-- benchmark_all.yaml
|-- dataset/
|   |-- taac2025.yaml
|   `-- taac2026.yaml
|-- model/
|   |-- itemcf.yaml
|   |-- mf.yaml
|   `-- deepfm.yaml
|-- training/
|   |-- default.yaml
|   `-- fast_dev.yaml
|-- evaluation/
|   |-- ctr.yaml
|   `-- ranking.yaml
|-- runtime/
|   |-- local.yaml
|   `-- ci.yaml
`-- hydra/
    `-- default.yaml
```

上面这个结构的关键思想是：

- `experiment/` 负责挑选“组合”
- `dataset/`、`model/`、`training/`、`evaluation/` 负责提供组件配置
- `runtime/` 负责输出路径、日志、设备、并发等运行环境差异

## 推荐 dataclass 分层

建议在 `src/recsys/utils/config.py` 中逐步建立结构化配置对象，例如：

- `ExperimentConfig`
- `DataConfig`
- `ModelConfig`
- `TrainingConfig`
- `EvaluationConfig`
- `RuntimeConfig`
- `RecBenchConfig`

其中：

- `RecBenchConfig` 是顶层聚合对象
- 每个子配置只描述一个稳定边界
- 每个 dataclass 只包含本组件真正拥有的字段

最佳实践是让每个新组件都拥有自己的 dataclass，而不是依赖一个共享的松散参数命名空间。这样模型和数据集更独立、更可复用，也更适合后续扩展。

## YAML 与 dataclass 的边界

推荐按下面的规则划分：

- 放进 dataclass 的内容：
  - 类型明确的字段
  - 需要默认值的字段
  - 需要校验的字段
  - 多组件共享的公共字段
- 放进 YAML 的内容：
  - 组件选择
  - 实验组合
  - 环境差异
  - 少量便于切换的超参数预设

不推荐把所有模型专属参数都长期塞进 `model.params: Dict[str, Any]`。更好的做法是：

- 顶层 `ModelConfig` 只放通用字段，如 `name`、`family`
- 每个模型自己再有一个 companion dataclass 作为具体 schema

## 推荐的主配置思路

主配置文件 `configs/config.yaml` 应该尽量轻，只负责 defaults 组合和全局入口。

建议它长期只承担下面几件事：

1. 定义 defaults 列表
2. 指定默认 experiment、dataset、model、training、evaluation、runtime
3. 放少量真正跨组件的根级设置

一个关键点是 composition order。推荐在主配置与子配置里都显式使用 `_self_`，避免覆盖优先级变得不可预测。

## Structured Config 最佳实践

Hydra 的 Structured Config 不只可以用来直接定义配置对象，也非常适合作为配置文件的 schema。推荐做法是：

- 为顶层配置提供一个 `RecBenchConfig`
- 为每个 config group 提供对应 base schema
- 在 YAML 的 defaults 中显式继承这些 base schema

这样做的效果是：

- YAML 仍然可以独立存在
- 但每个 YAML 都会在合成时被 dataclass schema 校验
- 命令行覆盖时的类型错误能尽早报出

这比纯 YAML 更稳，也比把所有配置都硬编码在 Python 里更灵活。

## 当前项目推荐字段边界

### `experiment`

应该放：

- `name`
- `seed`
- `tags`
- `output_dir`
- `track_with`
- `notes`

不应该放：

- 某个模型的专属结构参数
- DataLoader 细节

### `data`

应该放：

- `name`
- `data_dir`
- `split_ratios`
- `batch_size`
- `num_workers`
- `max_seq_len`
- `min_seq_len`
- `neg_sample_count`

需要注意：

- 数据路径应尽量使用 `${hydra:runtime.cwd}` 作为锚点，避免输出目录变化时相对路径失效
- 数据集特有字段可以放在具体 dataset config 中，而不是全部塞进全局 data 配置

### `model`

应该放：

- `name`
- `family`
- `task_type`

谨慎处理：

- `params`

建议中长期把 `params` 逐步替换成按模型注册的结构化配置，而不是长期保留为无约束字典。

### `training`

应该放：

- `epochs`
- `learning_rate`
- `weight_decay`
- `optimizer`
- `scheduler`
- `warmup_epochs`
- `early_stopping_patience`
- `gradient_clip_val`
- `mixed_precision`
- `accumulate_grad_batches`

### `evaluation`

应该放：

- `metrics`
- `ranking_k`
- `threshold`
- `generate_curves`
- `statistical_test`

最佳实践是让 `evaluation` 配置和任务类型对齐，而不是永远尝试“一次性算所有指标”。

### `runtime`

建议新增：

- `device`
- `num_devices`
- `deterministic`
- `log_level`
- `output_root`
- `resume_from`
- `fast_dev_run`

把这些字段单独拆到 `runtime`，可以减少 `experiment` 与训练逻辑的耦合。

## 命令行覆盖规范

Hydra 的价值之一就是覆盖能力，但覆盖也需要约束。

推荐覆盖场景：

- 切换数据集
- 切换模型
- 调整 batch size
- 调整学习率
- 切换 runtime 模式

示例：

```bash
uv run python scripts/run_single.py model=deepfm dataset=taac2026
uv run python scripts/run_single.py training.batch_size=512 training.learning_rate=3e-4
uv run python scripts/run_benchmark.py experiment=benchmark_classical runtime=ci
```

推荐原则：

- 组合切换优先使用 config group
- 单字段调参使用点号覆盖
- 不要在命令行上传太多 ad hoc 参数，避免实验不可复现

## 输出路径与工作目录

Hydra 的另一个常见坑是工作目录与输出目录。

最佳实践建议：

- 数据目录、预训练权重目录等输入路径，尽量基于 `${hydra:runtime.cwd}`
- 实验输出、日志、checkpoint 目录，统一落在 `outputs/` 下
- 不要依赖当前 shell 的相对路径碰巧正确

对于 RecBench，更推荐的思路是：

- 输入路径使用项目根目录作为锚点
- 输出路径由 runtime 或 experiment 统一控制
- benchmark 与单实验共用同一套 artifact 目录规范

## Benchmark 配置建议

`configs/experiment/*.yaml` 不应重复定义所有底层字段，而应更像“实验清单”。

例如，它更适合描述：

- 本次要跑哪些数据集
- 本次要跑哪些模型
- 聚合输出名称是什么

而不适合重复复制一整份训练和评估配置。否则 benchmark 文件会快速膨胀，而且难以维护。

## 配置校验建议

推荐分三层校验：

1. Hydra 合成时的 schema 校验
2. Python 启动时的语义校验
3. 组件实例化前的业务校验

示例思路：

- 类型不合法：由 Structured Config 捕获
- `split_ratios` 和不为 1：由配置加载器捕获
- 所选模型不支持当前任务：由 registry + pipeline 捕获

这三层不要混在一起，否则错误信息会非常难理解。

## 项目落地建议

对于当前仓库，我建议按下面顺序推进：

1. 在 `utils/config.py` 中定义顶层 dataclass
2. 给 `config.yaml` 建立顶层 schema
3. 给 `dataset/`、`training/`、`evaluation/` 先补 config group
4. 再给核心样板模型补 companion dataclass
5. 最后把 `benchmark` 场景接到统一的配置入口

最先应该稳定的不是所有模型参数，而是：

- 顶层配置树
- dataset/model/training/evaluation 的边界
- CLI 覆盖规范
- 输出目录规则

## 当前仓库需要避免的反模式

请尽量避免：

- 把所有配置长期写在一个大 YAML 里
- 让 `params: dict` 成为永久方案
- 让 benchmark 文件复制粘贴训练细节
- 同时存在多套入口命名与路径约定
- 用代码里的默认值和 YAML 里的默认值互相打架

## 一句话总结

对 RecBench 来说，最佳实践不是“只用 Hydra”或“只用 dataclass”，而是：

- 用 Hydra 做组合与覆盖
- 用 dataclass 做 schema 与类型边界
- 用 config group 管理组件选择
- 用统一的输出与校验规则保证实验可复现
