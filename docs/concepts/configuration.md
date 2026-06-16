---
title: Configuration Guide
description: 当前配置系统实现、字段边界与推荐用法
---

# Configuration Guide

## 配置系统目标

RecBench 的配置系统承担三件事：

- 用 YAML 或配置对象表达“这次实验要跑什么”
- 用 dataclass 固定“这些字段长什么样”
- 用语义校验尽量在运行前发现明显错误

当前仓库中，这套能力已经由 `src/recsys/utils/config.py` 落地，而不是停留在设计草案阶段。

## 当前实现

`src/recsys/utils/config.py` 已经提供：

- 顶层配置树 `RecBenchConfig`
- 六个子配置 dataclass：`ExperimentConfig`、`DataConfig`、`ModelConfig`、`TrainingConfig`、`EvaluationConfig`、`RuntimeConfig`
- `load_config()`：从 YAML 或 Hydra 上下文加载配置
- `validate_config()`：执行基础语义校验
- `resolve_paths()`：归一化输入输出路径
- `get_config_snapshot()`：生成可落盘配置快照
- Hydra `ConfigStore` 注册逻辑

这意味着文档应该基于当前 dataclass 字段来说明配置边界，而不是沿用旧的“planned schema”描述。

## 顶层配置树

当前顶层结构如下：

```text
RecBenchConfig
|-- experiment
|-- data
|-- model
|-- training
|-- evaluation
`-- runtime
```

每个子树只描述一个稳定边界，避免把所有参数堆进同一个开放字典。

## 字段边界

### `experiment`

适合放实验元信息：

- `name`
- `tags`
- `notes`
- `track_with`

这里不应放设备、种子、输出根目录等运行时控制字段。

### `data`

当前 `DataConfig` 已支持：

- `name`
- `data_dir`
- `split_ratios`
- `batch_size`
- `num_workers`
- `max_seq_len`
- `min_seq_len`
- `neg_sample_count`

这一层表达 dataset adapter 与 DataLoader 的公共配置，而不是模型专属特征工程。

### `model`

当前 `ModelConfig` 已支持：

- `name`
- `family`
- `task_type`
- `problem_type`
- `params`

其中 `params` 当前仍是开放字典，用于承接模型专属参数；这是当前实现允许的现实状态，但不是长期最理想形态。

### `training`

当前 `TrainingConfig` 已支持：

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

这部分字段与 `training/` 模块中的 optimizer、scheduler、trainer 工厂直接对应。

### `evaluation`

当前 `EvaluationConfig` 已支持：

- `metrics`
- `ranking_k`
- `threshold`
- `generate_curves`
- `statistical_test`

评估配置应和任务类型对齐；例如 ranking 任务重点使用 `ranking_k`，pointwise 任务重点使用阈值与分类指标集合。

### `runtime`

当前 `RuntimeConfig` 已支持：

- `device`
- `seed`
- `deterministic`
- `log_level`
- `output_root`
- `resume_from`
- `fast_dev_run`
- `num_devices`

注意：设备、种子、输出根目录都属于 `runtime`，不属于 `experiment`。

## 当前校验能力

`validate_config()` 当前已经校验：

- `split_ratios` 总和是否约为 1
- `split_ratios` 是否为正
- `evaluation.metrics` 是否为空
- `ranking_k` 是否为正整数列表
- `learning_rate` 是否大于 0
- `seed` 是否为非负整数

这是一套基础但真实存在的校验能力。文档应如实描述它的范围，同时避免误写成“所有组件名称和参数都会被完整校验”。

## 配置来源

当前配置加载支持两种模式：

### 1. 直接 YAML 加载

适合脚本、测试或显式传配置文件路径的场景。

```python
from recsys.utils.config import load_config

cfg = load_config("configs/config.yaml")
print(cfg.model.name)
print(cfg.runtime.output_root)
```

### 2. Hydra 上下文加载

如果运行环境已经通过 Hydra 生成配置，`load_config()` 会优先尝试从 Hydra 上下文读取。

## 配置快照

当前运行时会使用 `get_config_snapshot()` 生成可序列化快照，用于单实验目录中的 `config.yaml`。

快照会包含：

- 解析后的配置内容
- `_meta.schema_version`
- `_meta.resolved_at`
- `_meta.config_hash`

这保证了单实验结果具备可追溯性与可复现基础。

## 推荐配置示例

下面的示例与当前 dataclass 和训练/评估基础设施一致：

```yaml
experiment:
  name: itemcf_demo
  tags: [baseline, ranking]
  notes: "minimal runnable baseline"
  track_with: null

data:
  name: taac2026
  data_dir: ./data
  split_ratios: [0.8, 0.1, 0.1]
  batch_size: 256
  num_workers: 4
  max_seq_len: 50
  min_seq_len: 2
  neg_sample_count: 4

model:
  name: itemcf
  family: classical
  task_type: ranking
  problem_type: implicit_ranking
  params:
    similarity: cosine
    top_k_neighbors: 50
    recommend_k: 10

training:
  epochs: 10
  learning_rate: 1e-3
  weight_decay: 1e-5
  optimizer: adam
  scheduler: cosine
  warmup_epochs: 0
  early_stopping_patience: 5
  gradient_clip_val: null
  mixed_precision: null
  accumulate_grad_batches: 1

evaluation:
  metrics: [ndcg@10, hit_rate@10, recall@10, mrr]
  ranking_k: [10]
  threshold: 0.5
  generate_curves: false
  statistical_test: null

runtime:
  device: auto
  seed: 42
  deterministic: false
  log_level: INFO
  output_root: ./outputs
  resume_from: null
  fast_dev_run: false
  num_devices: 1
```

## 训练型模型的配置注意事项

对于训练型模型，配置层与代码实现之间还有两个现实边界：

- `training` 模块已经支持 optimizer、scheduler、callbacks 和 trainer 工厂
- 但 `run_experiment()` 里的训练型路径尚未接通

因此，训练型配置可以被文档说明为“已存在并与训练基础设施对齐”，但不能写成“已经能通过 experiment 主流程直接跑通”。

## 当前需要避免的旧写法

下面这些写法已经与当前实现不一致：

- 把 `seed`、`device`、`output_root` 写进 `experiment`
- 使用过期的脚本入口名，例如 `python -m src.run`
- 把 `track_with=mlflow` 写成受支持项
- 假设 CLI 覆盖命令已经是稳定公共接口

## 推荐实践

- YAML 负责选择组件与实验预设
- dataclass 负责类型边界和默认值
- 语义校验尽量前移到加载阶段
- 输出路径统一归到 `runtime.output_root`
- 新增模型时，优先让共享 `ModelConfig` 与 `params` 对齐；条件成熟后再拆成更细的结构化配置

## 一句话总结

RecBench 当前的配置系统已经具备“YAML + dataclass + 基础校验 + 快照落盘”的主干能力。写文档时应以 `src/recsys/utils/config.py` 中的真实字段和加载逻辑为准，而不是继续引用早期的规划版结构。
