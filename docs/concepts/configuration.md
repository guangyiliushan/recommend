---
title: Configuration Guide
description: RecBench 配置系统完整指南 — 配置树架构、加载机制、校验规则、运行时工具链与配置文件组织
---

# Configuration Guide

## 概述

RecBench 的配置系统负责将"这次实验要跑什么"从意图转化为可执行的配置对象。它承担三个核心职责：用 YAML 表达实验组合意图、用 dataclass 固定字段边界和类型、在运行前通过语义校验拦截明显的错误配置。

配置系统以 `src/recsys/utils/config.py` 为主干实现，辅以设备管理、日志追踪、可复现性保障、性能画像和进度追踪五个运行时工具模块。所有模块通过 `src/recsys/utils/__init__.py` 统一对外暴露稳定符号。

## 配置系统架构

当前配置系统采用 Hydra + YAML + dataclass 混合架构：

- **Hydra** 负责 YAML 组合、CLI 覆盖和 ConfigStore 结构化配置注册
- **OmegaConf** 负责 DictConfig 合并与解析
- **dataclass** 负责字段边界定义、类型约束和默认值
- **语义校验函数** 在 dataclass 类型检查之外执行跨字段逻辑校验

配置的生命周期为：YAML 文件 → OmegaConf 合并 → dataclass 实例化 → 路径解析 → 语义校验 → 管道桥接 → 快照落盘。

## 顶层配置树

顶层 `RecBenchConfig` 聚合六个子配置 dataclass，形成一棵清晰的配置树：

| 子树 | 职责 | 边界 |
|:---|:---|:---|
| `experiment` | 实验元信息 | 名称、标签、备注，不包含设备/种子等运行时字段 |
| `data` | 数据集与 DataLoader 配置 | 数据集选择、切分策略、批大小、序列长度等 |
| `model` | 模型选择与参数 | 模型注册名、家族、任务类型、模型专属参数 |
| `training` | 训练超参数 | 轮数、学习率、优化器、调度器、混合精度等 |
| `evaluation` | 评估配置 | 指标列表、ranking K 值、阈值策略、曲线生成 |
| `runtime` | 运行时环境 | 设备、种子、确定性模式、日志级别、输出路径 |

每个子树的字段由对应的 dataclass 严格定义，不接受超出定义的额外字段。这意味着在 YAML 中写入未定义的键会被 OmegaConf 在合并阶段拦截。

### experiment 子树

放置实验的元信息标识。包含四个字段：

- `name`：实验名称，作为运行 ID 的组成部分
- `tags`：标签列表，用于分组和筛选
- `notes`：自由文本备注
- `track_with`：实验追踪后端，当前支持 `"tensorboard"` 和 `"wandb"`，设为 `null` 则不启用

设备、种子、输出根目录等运行时控制字段应放在 `runtime` 子树，不应混入 `experiment`。

### data 子树

定义数据集适配器与 DataLoader 的公共配置。字段与数据集适配器（如 `TAAC2025Dataset`、`Movielens1MDataset`）的构造参数和安全校验规则直接对应：

- `name`：数据集注册键名，如 `"taac2026_second_round"` 或 `"movielens_1m"`
- `data_dir`：数据缓存根目录
- `split_mode`：数据切分方式，支持 `"temporal"`（按时间戳时序切分，默认）和 `"random"`（随机打乱后按比例切分）。该字段通过 `recbench_to_experiment_config()` 桥接传递至数据集适配器
- `split_ratios`：训练集、验证集、测试集的比例三元组
- `batch_size`、`num_workers`：DataLoader 的批次大小与并行工作进程数
- `max_seq_len`、`min_seq_len`：序列最大截断长度与最小过滤长度
- `neg_sample_count`：每个正样本对应的负样本数量
- `min_action_type`：TAAC 2025 的行为过滤阈值，`0` 表示包含所有行为类型（曝光+点击），`1` 表示仅提取点击作为正交互

### model 子树

定义模型的选择与参数传递：

- `name`：模型注册键名，如 `"itemcf"` 或 `"hyformer"`
- `family`：模型家族，如 `"classical"`（经典协同过滤）或 `"unified"`（统一架构）
- `task_type`：任务类型，`"ranking"`（排序）或 `"pointwise"`（点式预估）或 `"multitask"`（多任务）
- `problem_type`：问题子类型，如 `"binary"`（二分类）、`"implicit_ranking"`（隐式排序）
- `params`：模型专属参数的开放字典，由各模型在实例化时自行解析。当前这是实现允许的现实状态——不同模型家族的参数空间差异较大，尚不适合全部纳入强类型字段

`task_type` 和 `problem_type` 会影响评估系统的默认主要指标选择和评估路径分流。

### training 子树

定义训练过程的超参数，字段与 `training/` 模块中的优化器工厂、调度器工厂和 Trainer 工厂直接对应：

- `epochs`、`learning_rate`、`weight_decay`：基础训练控制
- `optimizer`、`scheduler`：优化器和学习率调度器名称，如 `"adam"`、`"cosine"`
- `warmup_epochs`：学习率预热轮数
- `early_stopping_patience`：早停耐心值
- `gradient_clip_val`：梯度裁剪阈值，`null` 表示不裁剪
- `mixed_precision`：混合精度训练模式，`"fp16"`、`"bf16"` 或 `null`
- `accumulate_grad_batches`：梯度累积步数

这些字段仅对可训练模型生效。对于 ItemCF 等非训练模型，训练阶段会被跳过。

### evaluation 子树

定义评估过程的配置，字段与 `src/recsys/evaluation/evaluator.py` 中的 `EvaluationConfig` 对齐：

- `metrics`：评估指标名称列表，支持分类指标（如 `roc_auc`、`log_loss`、`accuracy`）和排序指标（如 `ndcg@10`、`hit_rate@10`、`recall@10`、`mrr`）
- `ranking_k`：排序评估的 Top-K 值列表，如 `[5, 10, 20]`
- `threshold`：分类决策阈值，默认 `0.5`
- `generate_curves`：是否生成 ROC 和 PR 曲线数据
- `statistical_test`：统计检验方法，预留字段

主要指标的默认选择逻辑由评估器的 `get_primary_metric()` 方法依据 `task_type` 和 `problem_type` 自动确定：ranking 任务默认使用 `ndcg@K`，pointwise 二分类默认使用 `pr_auc`，多分类默认使用 `accuracy`。

### runtime 子树

定义跨组件的运行时环境设置：

- `device`：计算设备，支持 `"auto"`（自动探测 CUDA/MPS/CPU）、`"cuda"`、`"cpu"`、`"mps"`
- `seed`：全局随机种子
- `deterministic`：是否启用 PyTorch 确定性算法模式
- `log_level`：日志级别，默认 `"INFO"`
- `output_root`：所有产物输出的根目录
- `resume_from`：从指定检查点恢复训练
- `fast_dev_run`：快速开发模式（仅跑少量 batch 用于调试）
- `num_devices`：使用的设备数量

## 配置加载机制

配置加载通过 `load_config()` 函数实现，支持两种模式。

### YAML 独立加载模式

适用于脚本、测试或显式传配置文件路径的场景。函数从磁盘读取 YAML 文件，通过 OmegaConf 解析为 DictConfig，再转换为强类型的 `RecBenchConfig` 实例，最后依次执行路径解析和语义校验。支持通过 `overrides` 参数传入 CLI 覆盖列表（如 `["model=deepfm", "training.learning_rate=3e-4"]`），覆盖项会通过 OmegaConf 的合并机制叠加到基础配置上。

### Hydra 上下文加载模式

当程序通过 Hydra 入口（如 `scripts/run.py`）运行时，`load_config()` 会优先尝试从 Hydra 的 `HydraConfig` 上下文获取已合成的配置对象，避免重复解析。这一模式利用了 Hydra 的 YAML 组合能力——通过 `defaults` 列表和 CLI 的 `dataset=`、`model=` 等 group 选择器自动加载对应子树 YAML 文件。

### ConfigStore 注册

配置模块在导入时自动执行 `_register_config_store()`，将六个子配置 dataclass 注册为 Hydra Structured Config。注册内容包含基础配置节点和各 group 的 schema 节点：

- 基础节点 `base_recbench` 对应完整的 `RecBenchConfig`
- Group 节点覆盖 `experiment`、`dataset`、`model`、`training`、`evaluation`、`runtime` 六个子树

这意味着 Hydra 可以在 CLI 中自动补全各 group 的字段名，也能在类型不匹配时给出明确的错误提示。注册过程对 Hydra 未安装的环境静默跳过。

## 配置校验

`validate_config()` 在 dataclass 的类型检查之外执行第二层语义校验，当前覆盖以下规则：

- `split_ratios` 三项之和必须约为 `1.0`（容许 `±0.001` 的浮点误差）
- `split_ratios` 各项必须为正数
- `evaluation.metrics` 不能为空列表
- `ranking_k` 若提供则所有值必须为正整数
- `learning_rate` 必须大于 `0`
- `seed` 必须为非负整数

校验失败时抛出 `ConfigError` 异常，携带错误码、阶段标识和修复提示。校验不覆盖模型名称或数据集名称是否已注册——这类运行时校验由实验管线在引导阶段执行。

## 配置快照与可复现性

每次实验运行时，通过 `get_config_snapshot()` 生成配置快照并写入实验目录下的 `config.yaml` 文件。快照包含解析后的完整配置内容以及三条元信息：

- `_meta.schema_version`：快照结构版本号
- `_meta.resolved_at`：配置解析完成时间（ISO 8601 格式）
- `_meta.config_hash`：配置内容的 MD5 短哈希

配置快照保证了单实验结果的可追溯性——任何人在任何时间都可以根据快照文件重建完全相同的配置输入。

## Pipeline 桥接

`recbench_to_experiment_config()` 是配置系统与实验管线之间的桥接函数。它将 Hydra 层的 `RecBenchConfig`（包含六个子配置树）转换为管线层可消费的 `ExperimentConfig`（来自 `src/recsys/pipeline/experiment.py`）。

桥接过程完成了以下转换：

- 将 `RecBenchConfig` 的六个子树展开为 `ExperimentConfig` 的扁平字典字段（`data_config`、`model_config`、`training_config`、`evaluation_config`、`runtime_config`）
- `split_mode`、`split_ratios`、`min_action_type` 等数据字段完整透传
- `model.params` 字典原样传递，由模型在实例化时自行解析
- `evaluation.ranking_k` 在未提供时默认填充为 `[10]`

## 运行时工具链

配置系统集成了五个运行时工具模块，共同支撑实验的可复现性、可观测性和性能诊断。

### 设备管理

`src/recsys/utils/device.py` 提供统一的设备选择与能力探测。`get_device()` 函数按以下优先级确定计算设备：

1. 显式传入的设备字符串（`"cpu"`、`"cuda"`、`"mps"`、`"auto"`）
2. 环境变量 `RECBENCH_DEVICE`
3. 自动探测模式：优先 CUDA（若可用）、其次 Apple Silicon MPS（若可用）、最后回退 CPU

探测结果以 `DeviceInfo` 结构返回，包含设备对象、设备类型、CUDA 可用性、混合精度训练支持（AMP fp16 和 bf16）、GPU 名称、计算能力和总显存容量。对于不支持混合精度的旧款 GPU，`supports_amp` 和 `supports_bf16` 会被标记为 `false`。`get_device_info_summary()` 函数将探测结果格式化为人类可读的摘要字符串。

### 日志与实验追踪

`src/recsys/utils/logging.py` 基于 loguru 构建统一日志系统。`setup_logging()` 函数接收运行 ID、输出目录和追踪后端参数，完成以下初始化：

- 创建 `{output_dir}/experiments/{run_id}/logs/` 目录结构
- 配置终端日志 handler（彩色可读格式，INFO 级别）
- 配置文件日志 handler（完整格式含结构化上下文，DEBUG 级别，100 MB 轮转，30 天保留）
- 配置文件错误日志 handler（WARNING 级别）
- 通过 handler 桥接将标准 logging 库的消息转发到 loguru
- 可选初始化 TensorBoard 或 WandB 追踪器

日志系统设计为幂等操作——重复调用 `setup_logging()` 不会重复添加 handler。`LoggingContext` 结构记录已初始化的日志路径和追踪器信息。

`log_experiment_summary()` 用于在实验结束时记录格式化的实验摘要，包括数据集名称、模型名称、随机种子和所有评估指标值。

### 可复现性保障

`src/recsys/utils/reproducibility.py` 提供两层可复现性控制。`set_seed()` 统一设置 Python random、NumPy、PyTorch（含 CUDA）的全局随机种子，返回 `SeedInfo` 记录各组件实际生效状态。`deterministic_mode()` 启用 PyTorch 确定性算法模式，同时禁用 cuDNN benchmark 和 TF32 加速。该函数支持 `warn_only` 参数——当算子不支持确定性算法时，可选择仅警告而非抛出异常。

`get_reproducibility_summary()` 汇总当前的可复现配置状态（种子值、确定性模式、CUDA/cuDNN 版本信息），用于写入实验状态文件和日志。

### 性能画像

`src/recsys/utils/profiling.py` 提供模型性能画像能力。`profile_model()` 是主入口，接收模型、输入样本、设备和画像配置，依次执行四项测量：

- **参数量统计**：通过 `count_parameters()` 统计总参数量、可训练参数量和按一级模块汇总的参数量分布
- **推理延迟**：通过 `measure_inference_latency()` 在预热后测量多次推理的延迟（均值、标准差、最小/最大值）和吞吐量（样本/秒）
- **GPU 显存**：通过 `get_memory_usage()` 获取当前已分配、已保留和峰值 GPU 显存（仅 CUDA 设备）
- **FLOPs 估算**：通过 thop 或 fvcore 估算模型计算量，两项依赖均为可选，缺失时优雅降级

所有测量在 eval 模式下进行，对模型不产生副作用。画像结果以 `ProfilingResult` 结构返回，包含结构化数据和警告列表。

### 进度追踪

`src/recsys/utils/progress.py` 提供分层级的进度追踪，通过环境变量控制行为：

- `RECSYS_PROGRESS=0`（默认）：静默模式，不输出进度条
- `RECSYS_PROGRESS=1`：显示 tqdm 进度条
- `RECSYS_PROGRESS=2`：DEBUG 日志模式，输出阶段级的开始/完成日志
- `RECSYS_BENCHMARK_MODE=1`：Benchmark 并发模式，强制静默以避免多进程进度条交错

`progress_phase()` 是一个上下文管理器，在进度模式开启时创建 tqdm 进度条，在静默或 benchmark 模式时返回空操作对象。`phase_timer()` 是轻量级阶段计时器，仅记录耗时到传入的字典，不产生任何视觉输出。

## 配置文件组织

`configs/` 目录按功能层级组织，支持 Hydra 的 group 选择器组合机制：

```
configs/
├── config.yaml                        # 主配置文件，含 defaults 列表
├── dataset/                           # 数据集子树配置
│   ├── movielens_1m.yaml
│   ├── taac2025.yaml
│   └── taac2026.yaml
├── experiment/                        # Benchmark 矩阵配置
│   ├── benchmark_all.yaml
│   ├── benchmark_classical.yaml
│   ├── benchmark_deep_ctr.yaml
│   ├── benchmark_feature_cross.yaml   # planned
│   ├── benchmark_pcvr.yaml            # planned
│   ├── benchmark_sequence.yaml        # planned
│   └── benchmark_unified_gen.yaml     # planned
└── model/                             # 模型子树配置
    └── classical/
        └── itemcf.yaml
```

### 主配置文件

`configs/config.yaml` 是 Hydra 的默认入口配置，通过 `defaults` 列表声明了所有可被子树覆写的 group。它包含完整的六个子树示例值，同时禁用了 Hydra 自身的日志输出以避免与业务日志混淆。

### 数据集配置

`configs/dataset/` 目录下每个 YAML 文件定义一个数据集的完整默认参数。文件被 Hydra 通过 `dataset=<name>` CLI 选项加载并覆写主配置的 `data` 子树。当前包含：

- `movielens_1m.yaml`：经典序列推荐基准，`max_seq_len=50`，`min_seq_len=2`
- `taac2025.yaml`：全模态生成式推荐，支持 1M 和 10M 两个变体，默认 `min_seq_len=5`
- `taac2026.yaml`：轻量级广告 CTR/CVR 预估，支持两种 split_mode

### Benchmark 配置

`configs/experiment/` 目录下每个 YAML 文件定义一个 Benchmark 的模型×数据集矩阵。文件通过 `--config` 参数传递给 `scripts/run_benchmark.py`。当前三个配置处于活跃状态（`benchmark_all.yaml`、`benchmark_classical.yaml`、`benchmark_deep_ctr.yaml`），四个配置处于 planned 状态（内容已被注释，待对应模型实现后启用）。

### 模型配置

`configs/model/` 目录按模型家族分子目录组织。每个模型一个 YAML 文件，定义模型注册名、家族、任务类型和专属参数。当前仅 `classical/itemcf.yaml` 已落定，定义了 ItemCF 的相似度策略、邻居数、推荐列表长度和归一化开关四项参数。

## 配置实践建议

- YAML 负责选择组件与表达实验预设，dataclass 负责字段边界和类型约束，不要在 YAML 中写入未定义的字段
- 语义校验尽量前移到加载阶段，减少运行到一半才发现配置错误的代价
- 所有产物输出路径统一归属到 `runtime.output_root`，不要在各子树中分散定义输出路径
- `seed`、`device`、`output_root` 等运行时控制字段统一放在 `runtime` 子树，不要混入 `experiment` 子树
- 新增模型时优先利用 `model.params` 开放字典传递专属参数；条件成熟时可拆分为更细粒度的结构化配置
- 切分模式通过 `data.split_mode` 统一控制，CLI 入口（如 `scripts/run_single.py --split-mode`）负责透传
- Benchmark 配置文件描述的是要跑哪些模型和数据集，不应复制整套训练细节

## 常见误用与避免

以下写法与当前实现不一致，应避免：

- 将 `seed`、`device`、`output_root` 放入 `experiment` 子树——这些属于 `runtime`
- 使用已废弃的脚本入口名（如 `python -m src.run`）——正确入口为 `scripts/run.py` 或 `scripts/run_single.py`
- 将 `track_with=mlflow` 写成受支持项——当前仅支持 `tensorboard` 和 `wandb`
- 在 YAML 中引用未注册的数据集或模型键名——注册校验在运行时而非配置加载阶段执行
- 引用 `dssm` 作为训练型模型示例——当前训练型模型为 `hyformer`
- 假设所有 CLI 覆盖命令已是稳定的公共接口——Hydra 覆盖语法虽已可用，但字段路径应参考 `RecBenchConfig` 的 dataclass 定义

## 未来展望

以下能力已在设计规划中，但尚未落地：

- **模型配置的强类型化**：当前 `model.params` 为开放字典，未来可为成熟模型家族（如 deep_ctr、sequence）定义结构化的 `params` dataclass，在配置加载阶段即可校验参数合法性
- **Hydra 组合的完整模型配置覆盖**：`configs/model/` 下目前仅有 ItemCF 一个 YAML 文件，HyFormer 等其他模型的配置尚未以 YAML 形式组织。待模型注册激活后补充对应配置文件
- **statistical_test 字段的落地**：`evaluation.statistical_test` 当前为预留字段，Reporter v3 中的显著性检验能力暂未实现
- **混合精度训练的完整集成**：`training.mixed_precision` 字段已定义且可透传，但训练管线中的混合精度支持需进一步验证和文档化
- **resume_from 恢复训练的端到端验证**：`runtime.resume_from` 字段已定义，但基于检查点的完整恢复流程尚未经过端到端测试
- **profiling 的 Pipeline 集成**：性能画像模块已独立可用，但尚未集成到实验管线的默认执行流程中（当前需手动调用）
