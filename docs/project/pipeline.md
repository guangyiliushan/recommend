---
title: Pipeline Guide
description: 单实验主干的完整流程、能力路由、训练底层与评估集成
---

# Pipeline Guide

## 概述

RecBench 的 Pipeline 层负责将配置、数据、模型、评估和产物串成一条可复现的单实验执行主干。它不承担批量调度和聚合报告的职责——这些由 Benchmark 层负责。当前实现位于 `src/recsys/pipeline/experiment.py`，是项目最核心的执行编排器。

## 单实验主入口

Pipeline 层的公共入口是 `run_experiment()` 函数，接收 `ExperimentConfig` 对象（来自 `src/recsys/pipeline/experiment.py`，与 Hydra 层的 `RecBenchConfig` 不是同一个类），返回结构化的 `ExperimentResult`。

`ExperimentConfig` 包含以下核心字段：实验名称、数据集注册键名、模型注册键名、随机种子、输出目录，以及五个子配置字典（数据配置、模型配置、训练配置、评估配置、运行时配置）。

## 完整执行流程

`run_experiment()` 按八个阶段顺序执行，每个阶段对应 `ExperimentPhase` 枚举的一个值：

1. **配置冻结（CONFIG）**：校验并冻结 `ExperimentConfig`，计算配置的 SHA-256 哈希，生成格式为 `{实验名}__{数据集}__{模型}__seed{种子}__{短哈希}` 的运行 ID
2. **引导初始化（BOOTSTRAP）**：触发模型与数据集注册的副作用导入，调用 `setup_runtime()` 设置全局随机种子、PyTorch 确定性模式和日志级别
3. **数据加载（DATA）**：通过 `DATASET_REGISTRY` 按注册键实例化数据集，调用 `load()` 完成下载和数据切分
4. **模型实例化（MODEL）**：通过 `MODEL_REGISTRY` 创建模型实例，自动从数据集采集用户数、物品数、特征列名、ID 空间类型等元信息，通过 `schema_metadata` 字典传入模型构造函数
5. **训练执行（TRAINING）**：仅可训练模型进入此阶段
6. **预测推理（PREDICTION）**：根据模型能力选择执行路径
7. **评估计算（EVALUATION）**：将 `PredictionBundle` 送入评估器
8. **产物落盘（ARTIFACT）**：写入 `status.json`、`metrics.json`、`config.yaml`、`predictions.parquet` 和曲线文件

每个阶段完成后记录耗时，失败时阶段名会出现在结构化错误的 `phase` 字段中。

## 能力路由与双路径执行

Pipeline 通过 `route_execution()` 函数根据模型的 `Capability.TRAINABLE` 能力标记选择执行路径，而不是假设所有模型都需要梯度训练。

### 非训练路径

适用于 ItemCF 等不需要梯度优化的传统方法。执行流程为：

- 从数据集的 train 切分中提取用户-物品交互对列表，优先使用各 Split 提供的 `iter_user_item_pairs_fast()` 快速提取接口
- 调用 `model.fit()` 完成模型构建（如相似度矩阵计算）
- 从 train 切分提取用户到物品集合的映射作为用户历史，优先使用 `extract_user_item_mapping_fast()` 快速提取接口
- 从 test 切分提取用户到物品集合的映射作为测试真值（ground truth）
- 调用 `model.predict()` 获取预测得分
- 将所有的得分、标签和分组信息汇总为 `PredictionBundle`

快速提取接口的存在使得交互数据收集的复杂度从 O(总样本数) 降至 O(用户数)，在处理 MovieLens-1M 的 71 万个样本位置时效果显著。

### 训练路径

适用于继承 `NeuralRecommender` 并实现了 `forward()` 和 `compute_loss()` 方法的神经网络模型（如 HyFormer）。当前训练路径已通过 HyFormer 验证可正常接通。执行流程为：

- 通过 `BaseDataset.get_dataloader()` 构建 train/val/test 三个 DataLoader
- 调用 `create_trainer()` 将 `NeuralRecommender` 包装为 `LightningRecommender`（PyTorch Lightning 适配层），同时编译 `pl.Trainer`
- 执行 `trainer.fit(lit_model, train_loader, val_loader)` 进行训练
- 执行 `trainer.predict(lit_model, test_loader)` 进行推理
- 通过 `_assemble_bundle_from_predictions()` 将各 batch 的预测结果汇总为 `PredictionBundle`

`LightningRecommender` 是一个薄适配器，不实现模型语义。它将 DataLoader 产出的 batch 字典转换为 `Batch` 视图（同时完成设备迁移），再调用 `model.forward()` 和 `model.compute_loss()`。预测阶段返回标准化的中间预测块（包含分数、标签和分组信息），由 Pipeline 统一汇总。

### Schema Metadata 传递

训练路径在创建模型时，Pipeline 自动从数据集实例采集以下元信息并通过 `schema_metadata` 字典传入模型：

- `num_users` 和 `num_items`：Dense Remap 后的唯一用户数和物品数
- `feature_cols`：特征列名列表
- `padding_idx`：填充槽位索引（通常为 0）
- `user_id_space` 和 `item_id_space`：ID 空间类型（`dense_1_based` 或 `raw`）
- `max_user_id` 和 `max_item_id`：原始 ID 的最大值

## 训练基础设施

Pipeline 的训练路径依赖六个训练子模块，位于 `src/recsys/training/` 目录下。

### 训练适配器（trainer.py）

`LightningRecommender` 将 `NeuralRecommender` 适配为 PyTorch Lightning 的 `LightningModule`。它在训练步中调用 `model.compute_loss()` 并记录多个损失分量（如 `train/loss`、`train/focal_loss` 等），在验证步中记录 `val/loss`。预测步返回包含分数、标签、任务输出和分组 ID 的标准化字典，不在此处写入文件。

`TrainerFactory` 负责将训练配置和运行时配置编译为 `pl.Trainer` 实例。它内部依次调用分布式策略解析器、日志器构建器、回调组装器和优化器/调度器配置构建器，最后组合出完整的 Trainer 对象。

`create_trainer()` 是一次性创建 Trainer 和包装模型的便捷函数，供 Pipeline 直接调用，减少装配复杂度。

### 优化器（optimizers.py）

`build_optimizer()` 根据 `OptimizerConfig` 构建优化器实例，支持四种稳定基础优化器：Adam、AdamW、SGD 和 Adagrad。可选扩展 Lion 优化器（若 PyTorch 版本支持，否则回退为 AdamW）。

`build_param_groups()` 支持参数组策略——通过配置可指定不施加权重衰减的参数名模式（如 bias 和 LayerNorm）以及 embedding 参数的独立学习率缩放因子。若未提供参数组配置则返回单一默认组。

工厂函数 `get_optimizer()` 提供按名称的简单查找接口。

### 调度器（schedulers.py）

`build_scheduler()` 根据 `SchedulerConfig` 构建学习率调度器，支持七种调度策略：`cosine`（余弦退火）、`cosine_warmup`（带线性预热的余弦退火）、`step`（固定步长衰减）、`multi_step`（多里程碑衰减）、`plateau`（基于监控指标的 ReduceLROnPlateau）、`onecycle`（OneCycleLR）和 `polynomial`（多项式衰减）。

所有调度器通过统一的 `SchedulerOutput` 结构返回，包含调度器实例、调度类型（`step`/`epoch`/`plateau`）和 Lightning 兼容的 `interval` 与 `frequency` 字段。`build_warmup_scheduler()` 提供独立的预热构建函数，可通过 `SequentialLR` 与主调度器组合。

### 回调（callbacks.py）

`build_callbacks()` 根据训练配置组装回调列表。内置回调包括：

- `ModelCheckpoint`：按监控指标保存最佳模型，支持保留最近和 Top-K 个检查点
- `EarlyStopping`：基于监控指标的早停，耐心值可配置
- `LearningRateMonitor`：逐步记录学习率变化
- `RichProgressBar`：终端富文本进度条

此外提供三个自定义回调：

- `GradientNormMonitor`：按指定频率（默认 50 步）记录梯度的 L2 范数
- `MemoryMonitor`：按指定频率（默认每个 epoch）记录 GPU 显存的已分配量和保留量
- `RunSummaryCallback`：训练结束时输出摘要日志，内容包括总 epoch 数、总步数、模型参数量和最佳检查点路径

### 损失函数（losses.py）

损失函数模块基于 `LOSS_REGISTRY` 构建统一的损失函数注册与工厂机制。支持的基础损失包括：

- `BCELossWrapper` 和 `BCEWithLogitsLossWrapper`：二分类交叉熵（后者含数值稳定性优化）
- `CrossEntropyWrapper`：多分类交叉熵（含标签平滑）
- `BPRLoss`：贝叶斯个性化排序损失，用于隐式反馈的三元组排序
- `InfoNCELoss`：对比学习损失，含温度参数
- `TOP1Loss`：Session 推荐的 TOP1 损失
- `FocalLoss`：缓解类别不平衡的焦点损失，含可配置的 alpha 和 gamma 参数
- `MultiTaskLoss`：多任务加权损失，支持固定权重和不确定性加权两种模式
- `AdaptiveHuberLoss`：鲁棒回归的自适应 Huber 损失

函数式版本 `sigmoid_focal_loss()` 直接接受 logits，支持 `none`/`mean`/`sum` 三种归约方式，与 Trainer 的分离式损失计算路径兼容。工厂函数 `get_loss()` 按名称从注册表获取损失实例。

### 分布式策略（distributed.py）

`resolve_strategy()` 将运行时配置（设备类型、设备数量）映射为 PyTorch Lightning 的 strategy 参数。映射规则为：CPU 设备使用单机 `auto` 策略；单 GPU 使用 `auto` 策略；多 GPU 自动升级为 `ddp` 策略；混合精度（fp16/bf16）透传给 `precision` 参数。FSDP 和 DeepSpeed 当前返回明确的未就绪错误，计划在后续阶段逐步接入。

`check_distributed_available()` 执行环境兼容性检查：DDP 需要至少 2 个 GPU；FSDP 和 DeepSpeed 当前均返回未启用状态并给出替代建议。

`get_strategy_kwargs()` 将解析后的策略配置转换为可直接传入 `pl.Trainer` 构造函数的参数字典。

## 统一预测产物

无论模型来自哪条路径，Pipeline 最终统一产出 `PredictionBundle` 对象，这是评估层与模型解耦的关键契约。常见字段包括：

- `task_type`：任务类型（pointwise / ranking / multitask）
- `problem_type`：问题子类型（binary / multiclass / implicit_ranking）
- `y_true`：真实标签
- `y_score`：预测分数
- `y_pred`：离散预测值（可选）
- `group_ids`：排序任务的分组标识
- `candidate_ids`：候选物品 ID 列表
- `task_outputs`：多任务各任务的分数输出
- `task_labels`：多任务各任务的标签
- `metadata`：附加元信息

## 评估集成

Pipeline 在拿到 `PredictionBundle` 后调用 `evaluate()` 主入口函数。评估器根据 bundle 的 `task_type` 字段自动分流：pointwise 任务走二分类或多分类指标计算路径，ranking 任务走排序指标计算路径，multitask 任务逐任务头分别评估后汇总。

Pipeline 自身不实现任何指标逻辑——它只负责将预测产物交给评估层。评估器返回的 `EvaluationResult` 包含汇总指标、分任务指标、分组指标和曲线数据，Pipeline 将其中的可序列化部分写入 `metrics.json`。

## 产物输出

单次实验产出的完整文件清单：

| 产物文件                | 内容                                        | 写入阶段 |
| :---------------------- | :------------------------------------------ | :------- |
| `status.json`           | 运行 ID、状态、起止时间、主指标值、错误信息 | ARTIFACT |
| `config.yaml`           | 冻结后的完整配置快照                        | CONFIG   |
| `metrics.json`          | 汇总指标、分任务指标、分组指标、曲线数据    | ARTIFACT |
| `predictions.parquet`   | 列式预测结果，按任务类型组织列结构          | ARTIFACT |
| `curves/roc_curve.json` | ROC 曲线数据点                              | ARTIFACT |
| `curves/pr_curve.json`  | PR 曲线数据点                               | ARTIFACT |
| `checkpoints/`          | PyTorch Lightning 检查点文件（仅训练路径）  | TRAINING |
| `logs/run.log`          | 运行日志（含 DEBUG 级别完整信息）           | 贯穿     |
| `logs/stderr.log`       | 错误日志（WARNING 及以上）                  | 贯穿     |

`write_predictions_parquet()` 根据任务类型组织列结构：pointwise 任务每样本一行（含 `y_score`、`y_true`、`y_pred` 等），ranking 任务每分组一行（含 `group_id` 和按候选项展开的分数列），multitask 任务每任务每样本一行（含 `task_name`、`y_score`、`y_true`、`y_mask` 等）。若 bundle 数据无法展开为表格则跳过写入。

`write_curve_artifacts()` 将评估结果中的 ROC 和 PR 曲线数据以 JSON 格式写入 `curves/` 子目录。

## 单实验目录结构

```
outputs/runs/{run_id}/
├── config.yaml              # 配置快照
├── status.json              # 运行状态
├── metrics.json             # 评估结果
├── predictions.parquet      # 预测结果
├── curves/
│   ├── roc_curve.json
│   └── pr_curve.json
├── checkpoints/             # 模型检查点（训练路径）
│   └── best-epoch=*.ckpt
└── logs/
    ├── run.log
    └── stderr.log
```

## 结构化错误处理

Pipeline 采用结构化错误模型 `ExperimentError`，包含错误码（如 `DATA_LOAD_ERROR`、`MODEL_INIT_ERROR`）、失败阶段（来自 `ExperimentPhase` 枚举）、错误消息和可选修复提示。失败时不会抛出不透明异常——错误信息被封装进 `ExperimentResult.error` 字段，Benchmark 层可以稳定消费而无需了解内部细节。

## Pipeline 层的边界

Pipeline 层与训练层的职责划分：

- Pipeline 决定何时需要 Trainer，训练层决定如何训练神经模型
- 训练层不负责实验目录结构和产物契约
- Pipeline 不负责优化器、调度器和回调的具体实现细节

Pipeline 层与 Benchmark 层的职责划分：

- `run_experiment()` 负责单次实验的一个配置组合
- `run_benchmark()` 负责多个配置组合的矩阵展开、调度和恢复
- `Reporter.generate()` 负责聚合已完成的实验结果

这三个边界在代码中已拆分为独立的模块（`experiment.py`、`benchmark.py`、`reporter.py`），文档中也保持对应的清晰划分。

## 当前限制

- 训练型模型当前可运行的实例为 `hyformer`，其余模型文件的注册装饰器仍处于注释状态
- 非训练路径（ItemCF）已验证与 MovieLens-1M、TAAC 2025、TAAC 2026 等数据集的兼容性
- Benchmark 预设与 Experiment 预设的完整 Hydra 解析能力尚未全部打通
- Pipeline 层依赖 Benchmark 层在展开矩阵时提供完整配置，自身不承担配置展开职责

## 未来展望

- **FSDP 和 DeepSpeed 分布式策略的接入**：`distributed.py` 中已预留检查和错误提示，计划在训练主干稳定后分阶段接入
- **Pipeline 层的性能画像集成**：性能画像模块（`profiling.py`）已独立可用，但尚未集成到 `run_experiment()` 的默认执行流程中
- **多模态 batch 契约的 Pipeline 级统一**：当前多模态嵌入数据主要在评估阶段使用，作为训练输入的标准化 Pipeline 路径尚未定义

## 参考

- [Pipeline 源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/pipeline/experiment.py)
- [训练框架源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/training/trainer.py)
- [配置系统文档](../concepts/configuration.md)
- [Benchmark 文档](benchmarking.md)
