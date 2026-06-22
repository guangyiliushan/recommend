---
title: Experiments
description: 实验管线完整指南 — CLI 入口、Benchmark 矩阵、评估体系、聚合报告与当前实现状态
---

# Experiments

## 概述

`docs/experiments/` 目录用于沉淀实验层的说明文档，面向需要复现实验、运行 Benchmark 或理解评估流程的开发者。它涵盖可用 CLI 入口、活跃的 Benchmark 套件、评估指标体系和结果聚合产物。

本文档的职责是描述当前仓库中真实可运行的实验能力与流程，不承担 API 契约、数据格式或架构原则的说明（这些内容属于 `docs/project/` 和 `docs/concepts/` 的范畴）。

## CLI 入口总览

RecBench 提供多条 CLI 入口，覆盖从单个实验到批量 Benchmark 再到消融分析的完整场景。所有入口均通过 uv 运行。

| 脚本                                 | 用途                                               | 典型场景                           |
| :----------------------------------- | :------------------------------------------------- | :--------------------------------- |
| `scripts/run.py`                     | Hydra 主入口，通过 YAML 组合配置运行单次实验       | 日常开发、参数组合探索             |
| `scripts/run_single.py`              | 命令行直参模式运行单次实验                         | 快速验证、无 Hydra 依赖的 CI 环境  |
| `scripts/run_benchmark.py`           | 批量 Benchmark，支持矩阵展开与配置文件             | 多模型×多数据集×多 seed 的全面对比 |
| `scripts/run_ablation.py`            | 消融实验矩阵执行器                                 | 单超参数变体对比分析               |
| `scripts/download_data.py`           | 通过 HuggingFace 下载数据集到本地缓存              | 首次使用前的数据准备               |
| `scripts/benchmark_data_pipeline.py` | 数据管线性能基准测试                               | 对比存储格式、压缩算法与计算后端   |
| `scripts/generate_report.py`         | 从 Benchmark 的 summary.csv 生成 Markdown 对比报告 | 实验完成后生成可读报告             |

### run.py — Hydra 主入口

基于 Hydra 配置框架的单次实验入口，从 `configs/config.yaml` 加载默认配置，支持通过 CLI 覆盖任意叶子字段，并通过 `dataset=` 和 `model=` 前缀组合不同数据集的 YAML 子树和模型配置。流程为：加载 YAML 树 → 校验配置 → 桥接为管线层 `ExperimentConfig` → 执行 `run_experiment()`。

### run_single.py — 命令行直参模式

支持两种运行模式：

- **argparse 模式**（默认）：直接通过命令行参数指定模型注册名、数据集注册名、随机种子、训练超参数和评估指标。额外提供 `--min-action-type` 参数控制 TAAC 2025 的正交互过滤阈值，以及 `--split-mode` 参数选择时序或随机切分方式。
- **Hydra 模式**（`--hydra`）：从 `configs/config.yaml` 加载基准配置并注入各实验参数，支持 `--hydra-overrides` 按点号路径覆盖任意字段。

### run_benchmark.py — 批量 Benchmark

支持两种运行模式：

- **配置文件模式**：通过 `--config` 指定 YAML 配置文件（如 `configs/experiment/benchmark_classical.yaml`），自动展开模型×数据集×种子矩阵。
- **命令行模式**：通过 `--models` 和 `--datasets` 直接指定参与矩阵的组件列表。

**核心特性**：

- **恢复模式**（`--resume-mode`）：支持四种策略 — `successful_skip`（跳过已完成，默认）、`failed_only`（仅重试失败）、`unfinished_only`（继续未完成）和 `force`（强制全量重跑）。
- **受控并发**（`--max-concurrent`）：控制并行运行的最大实验数，默认串行执行。
- **Hydra 集成**（`--hydra`）：从 YAML 加载每轮实验的默认参数，注入 data、model、training、evaluation 和 runtime 配置子树。

### run_ablation.py — 消融实验

为单一超参数构建多组变体配置，逐组执行实验并输出 CSV 对比表。每个模型预置了默认消融参数候选值（如 ItemCF 支持 `similarity`、`top_k_neighbors`、`recommend_k`、`normalize` 四组参数空间），也可通过 `--vary` 自定义参数和候选值列表。

### download_data.py — 数据集下载

维护注册名到 HuggingFace 仓库 ID 的映射表，支持 `taac2025_1M`、`taac2025_10M`、`taac2026_data_sample` 和 `taac2026_second_round` 四个数据集的命令行下载。通过 `--cache-dir` 指定本地缓存目录。MovieLens-1M 在数据集适配器内部自行处理下载，不通过此脚本。

### benchmark_data_pipeline.py — 数据管线性能基准

生成合成推荐风格数据集（可配置行数、用户数、物品数和特征列数），然后对存储格式（CSV/Parquet/Feather/ORC）、压缩算法（Snappy/Zstd/LZ4/Gzip）和计算后端（pandas/pyarrow/polars/dask/vaex/modin）的全组合矩阵进行读写性能测量。测量项包括读取耗时、写入耗时、输出文件大小、压缩比、缓存命中耗时和峰值内存使用。输出产物为 `summary.csv`、`formats.csv`、`backends.csv`、`report.md` 和 `report.json`。

### generate_report.py — 报告生成

从 Benchmark 产出的 `summary.csv` 加载实验数据，按模型分组计算耗时和主要指标的均值、标准差、最小值和最大值，生成包含速度对比、准确性对比和内存使用的 Markdown 性能对比报告。支持通过 `--output` 指定输出目录。

## 实验管线架构

### 单实验执行流程

`run_experiment()`（位于 `src/recsys/pipeline/experiment.py`）是 RecBench 的原子执行单元，编排从配置到产物的完整流程，分为八个阶段：

1. **配置解析（CONFIG）**：冻结 `ExperimentConfig`，计算配置哈希，生成稳定的运行 ID
2. **引导初始化（BOOTSTRAP）**：触发模型与数据集注册副作用，设置随机种子和设备
3. **数据加载（DATA）**：通过 `DATASET_REGISTRY` 实例化数据集，调用 `load()` 执行下载与切分
4. **模型实例化（MODEL）**：通过 `MODEL_REGISTRY` 创建模型，传入模型配置和数据集元信息（用户数、物品数）
5. **训练执行（TRAINING）**：仅可训练模型进入此阶段，通过 PyTorch Lightning Trainer 执行训练
6. **预测推理（PREDICTION）**：根据模型能力自动选择训练路径（DataLoader + Lightning）或非训练路径（直接调用 `predict()`）
7. **评估计算（EVALUATION）**：将 `PredictionBundle` 送入评估器，根据任务类型（ranking/pointwise/multitask）自动选择评估分支
8. **产物落盘（ARTIFACT）**：写入 `status.json`、`metrics.json`、`config.yaml`、`predictions.parquet` 和曲线文件

### 双路径执行机制

实验管线通过 `route_execution()` 函数根据模型能力自动选择执行路径：

- **可训练模型路径**（`Capability.TRAINABLE`）：通过 `BaseDataset.get_dataloader()` 构建 DataLoader → 包装为 `LightningRecommender` → `Trainer.fit()` 训练 → `Trainer.predict()` 推理 → 汇总 `PredictionBundle`。适用于 HyFormer 等神经网络模型。
- **非训练模型路径**：直接从数据集切分中提取用户-物品交互对（优先通过 `iter_user_item_pairs_fast()` 快速路径），调用 `model.predict()` 获取预测得分，构造 `PredictionBundle`。适用于 ItemCF 等经典方法。

非训练路径中，优先利用各 Split 实现提供的快速提取接口（`iter_user_item_pairs_fast()` 和 `extract_user_item_mapping_fast()`），以 O(用户数) 而非 O(总样本数) 复杂度收集交互数据。

### 实验结果结构

每次实验产出一个 `ExperimentResult`，包含以下核心字段：

- **运行 ID**：格式为 `{实验名}__{数据集}__{模型}__seed{种子}__{短哈希}`，保证稳定性和可恢复性
- **执行状态**：`succeeded` / `failed` / `skipped`
- **汇总指标**：主指标及其他评估指标的名称-数值映射
- **产物路径**：`status.json`、`metrics.json`、`config.yaml`、`predictions.parquet` 等文件的磁盘路径
- **结构化错误**：失败时包含错误码、失败阶段、错误信息和可选提示
- **运行时元信息**（v2）：模型家族、任务类型、训练样本数、模型参数量、峰值内存等

## Benchmark 配置矩阵

`configs/experiment/` 目录下的 YAML 文件定义了 Benchmark 的模型×数据集矩阵。当前状态分为活跃配置（可直接运行）和预留模板（待模型实现后启用）两类。

### 活跃的 Benchmark 配置

| 配置文件                   | 模型矩阵         | 数据集矩阵                                                                           | 说明                                                        |
| :------------------------- | :--------------- | :----------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| `benchmark_classical.yaml` | itemcf           | movielens_1m, taac2025_1M                                                            | 经典协同过滤基线，两个序列推荐数据集                        |
| `benchmark_deep_ctr.yaml`  | hyformer         | taac2026_data_sample, taac2026_second_round                                          | CTR/CVR 点式预估，含 domain 序列的 richer 变体              |
| `benchmark_all.yaml`       | itemcf, hyformer | movielens_1m, taac2025_1M, taac2025_10M, taac2026_data_sample, taac2026_second_round | 全矩阵覆盖，将可训练和非训练模型同时包裹在一个 Benchmark 中 |

当前仓库中仅两个模型已完成注册并可用：`itemcf`（非训练、ranking 任务）和 `hyformer`（训练型、pointwise 任务）。其余模型文件虽已存在于源码树中，但注册装饰器被注释，尚未激活。

### 预留模板（待实现）

以下配置文件内容已被注释，保留为待模型实现后的启用模板：

| 配置文件                       | 计划模型                                        | 计划数据集                    | 状态    |
| :----------------------------- | :---------------------------------------------- | :---------------------------- | :------ |
| `benchmark_feature_cross.yaml` | DCN, DCNv2, DHEN, RankMixer                     | criteo_kaggle                 | planned |
| `benchmark_sequence.yaml`      | SASRec, BERT4Rec, MIMN, SIM, ETA, SDIM          | movielens_1m, taobao_behavior | planned |
| `benchmark_unified_gen.yaml`   | HSTU, OneTrans, HyFormer, GenRec, COBRA         | criteo_kaggle                 | planned |
| `benchmark_pcvr.yaml`          | ESMM, ESM2, DESMM, HM3, DCMT, ChorusCVR, RankUp | taobao_behavior               | planned |

## 结果聚合与报告

Benchmark 执行完毕后，由 Reporter（位于 `src/recsys/pipeline/reporter.py`）消费所有 `ExperimentResult` 并生成结构化聚合产物。Reporter 不与模型、数据集或训练器直接接触，所有输入均来自实验结果的序列化数据，可离线重建。

### 聚合产物清单

| 产物文件          | 内容说明                                                                    | 版本 |
| :---------------- | :-------------------------------------------------------------------------- | :--- |
| `summary.csv`     | 每轮实验一行，包含运行 ID、数据集、模型、种子、状态、主指标值和所有评估指标 | v1   |
| `leaderboard.csv` | 按模型×数据集×主指标聚合，仅包含成功的运行，含均值、标准差和排名            | v1   |
| `failures.csv`    | 仅包含失败的运行，含失败阶段、错误码和错误信息                              | v1   |
| `manifest.json`   | Benchmark 元信息索引，记录 Benchmark 名称、生成时间和所有运行的基本信息     | v1   |
| `report.html`     | 交互式 HTML 摘要页，内联 JavaScript 排序，无需外部依赖                      | v1   |
| `trend.csv`       | 同一模型×数据集组合下不同种子的指标变化趋势                                 | v2   |
| `stability.csv`   | 跨种子的指标稳定性分析，含均值、标准差和变异系数                            | v2   |

### 趋势与稳定性分析（v2）

v2 版本的 Reporter 新增了两个分析维度：

- **趋势分析**：对每个模型×数据集组合，按种子展开所有运行的指标值、耗时和数据规模，用于观察不同随机初始化下的性能波动。
- **稳定性分析**：对跨种子的指标进行均值、标准差和变异系数（CV = 标准差/均值绝对值）的计算，量化模型在随机性下的表现稳定性。

### 单实验产物

每次独立实验在输出目录下生成以下文件：

- `status.json`：运行状态、起止时间、主指标值和错误信息
- `metrics.json`：完整的评估结果，含汇总指标、分任务指标、分组指标和曲线数据
- `config.yaml`：冻结后的完整配置快照
- `predictions.parquet`：预测结果列式文件，按任务类型（pointwise/ranking/multitask）组织列结构
- `curves/`：ROC 曲线和 PR 曲线的 JSON 数据文件

### 生成性能对比报告

`scripts/generate_report.py` 独立于 Benchmark 管线运行，从任意 `summary.csv` 生成 Markdown 格式的速度对比、准确性对比和内存使用表格，可指定输入和输出目录。

## 评估体系

评估层（位于 `src/recsys/evaluation/`）采用四层分离架构，各层职责明确、互不交叉。

### 指标计算层

点式评估指标（`metrics.py`）实现了以下分类指标的三层计算体系：

- **原子统计层**：混淆矩阵与基础计数（TP、TN、FP、FN）
- **派生指标层**：准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1 分数、特异度（Specificity）、阴性预测值（NPV）、假阳性率（FPR）、假阴性率（FNR）、平衡准确率
- **阈值无关层**：ROC-AUC、PR-AUC、平均精确率（Average Precision）、对数损失（Log Loss）、Brier 分数、ROC 曲线点和 PR 曲线点

所有指标使用规范英文键名（如 `accuracy`、`roc_auc`、`pr_auc`），同时提供完整的中英文别名映射表，用户可在配置中使用简写形式。

排序评估指标（`ranking.py`）实现了逐组计算再跨组聚合的排序指标：

- **NDCG@K**：归一化折损累计增益
- **MRR**：平均倒数排名
- **HitRate@K**：命中率
- **Recall@K**：召回率
- **Precision@K**：精确率
- **MAP**：平均精确率均值

所有排序指标严格遵循"先逐组计算、再跨组聚合"原则，对空组、无正样本组和候选不足 K 个的组有明确的处理策略。

### 评估编排层

评估器（`evaluator.py`）作为评估主入口，接收 `PredictionBundle` 和 `EvaluationConfig`，根据 `task_type` 自动分流：

- `pointwise`：调用分类指标计算（二分类或多分类）
- `ranking`：调用排序指标计算
- `multitask`：逐任务头分别评估后汇总

主要指标（primary metric）的默认选择逻辑：ranking 任务取 `ndcg@K`（K 取第一个配置值），pointwise 和 multitask 任务根据 problem_type 选择 `pr_auc`（二分类）、`accuracy`（多分类）或 `roc_auc`（其他）。

评估配置支持阈值策略选择（固定阈值、从 bundle 提取、逐任务阈值）、曲线生成控制和样本权重启用等参数。

### 可视化层

可视化模块（`visualization.py`）只消费已计算好的结构化结果和曲线点，输出图表或原始曲线文件，不负责重新计算数值指标。支持的输出包括 ROC 曲线、PR 曲线、阈值扫描结果、NDCG@K 折线图和指标排行榜图。设计原则为原始曲线点优先于图片文件，支持多模型叠加但要求相同任务类型、数据集和切分。

## 当前实验层的现实边界

### 已确认的模型与数据集组合

截至当前，仓库中已完成注册且可正常运行的组合为：

- **itemcf**（非训练、ranking）：movielens_1m、taac2025_1M、taac2025_10M、taac2026_data_sample、taac2026_second_round
- **hyformer**（训练型、pointwise）：taac2026_data_sample、taac2026_second_round

itemcf 对所有已注册数据集均兼容，hyformer 主要适配 TAAC 2026 的点式预估格式。`benchmark_all.yaml` 已配置了完整的交叉矩阵，runner 会在执行时自动跳过不兼容的组合（如 hyformer 在序列格式数据集上）。

### train_hyformer.py 的定位

`scripts/train_hyformer.py` 是一个独立的模型结构快速调试脚本，使用随机生成的内存数据（TensorDataset）直接训练 HyFormer 模型，不通过数据集适配器、Pipeline 或评估系统。它支持双优化器（Adagrad 管理稀疏参数、AdamW 管理稠密参数）、Focal Loss 和完整的命令行参数覆盖。该脚本的设计目的是快速验证模型结构的前向传播和反向传播正确性，生产环境的训练必须使用 `scripts/run_single.py` 或 `scripts/run.py`。

### 已知限制

- 当前仅 itemcf 和 hyformer 两个模型已注册激活，其余约 50 个模型文件（DeepFM、SASRec、DCN、ESMM 等）的注册装饰器均处于注释状态，待实现后再激活。
- 四个 Benchmark 配置文件（feature_cross、sequence、unified_gen、pcvr）为预留模板，其内容已被注释，引用数据集（criteo_kaggle、taobao_behavior）也未注册。
- 多模态嵌入在训练 batch 契约中尚未统一为端到端可训练特征通道，当前主要用于评估阶段的候选约束。

## 实验页面组织建议

### 当前适合优先成文的实验主题

基于现有代码支撑度，以下实验主题最值得独立成文：

1. **ItemCF 基线实验**：非训练 ranking 路径的完整复现指南，涵盖相似度策略选择、邻居数调优和评估指标解读
2. **HyFormer 训练实验**：点式预估模型的训练与评估完整流程，涵盖双优化器配置、Focal Loss 选择和 split_mode 选择
3. **全矩阵 Benchmark**：使用 `benchmark_all.yaml` 运行 itemcf + hyformer 跨多数据集的全量对比
4. **消融分析**：通过 `run_ablation.py` 对 ItemCF 的相似度函数和邻居数进行参数敏感性分析

### 推荐页面模板

每篇实验页建议包含以下章节：

1. 页面目标与适用范围
2. 当前仓库支持范围（模型、数据集、评估指标）
3. 前置准备（数据下载、依赖安装）
4. 运行命令与参数说明
5. 产出物与结果解读
6. 常见问题排查

### 文件命名规范

使用小写字母、连字符分隔，名称直接反映实验主题。例如 `baseline-itemcf.md`、`baseline-hyformer.md`、`benchmark-all-multiseed.md`。

## 未来展望

以下能力已在设计规划或源码占位中预留：

- **剩余约 50 个模型的注册激活与端到端验证**：涵盖 DeepFM、SASRec、DCN、ESMM、HSTU 等多类模型家族，激活后需与对应 Benchmark 配置文件联动启用。
- **criteo_kaggle 和 taobao_behavior 等公开基准数据集适配器**：这四个数据集在 `data/datasets/__init__.py` 的模块文档中已列出，且 Benchmark 模板中已有引用，但适配器尚未实现。
- **LaTeX 表格导出与统计显著性检验**：Reporter v3 规划中的能力，用于学术论文级别的结果输出。
- **多模态端到端训练管线**：将 TAAC 2025 的多模态嵌入作为模型输入的标准化 batch 格式和训练流程。
