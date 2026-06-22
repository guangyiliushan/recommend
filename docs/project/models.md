---
title: Model Integration Guide
description: 模型契约、注册机制、能力路由与当前实现范围的完整指南
---

# Model Integration Guide

## 概述

RecBench 的模型层围绕统一的注册机制、最小公共契约和能力显式声明构建。它的核心设计目标不是提供一个"万能基类"，而是定义一组薄接口让模型遵守，同时通过注册表实现模型的自动发现和按需获取。

本文档描述当前已落地的模型契约、注册机制、能力声明体系以及真实可运行的模型范围。目录预留但未实现的模型统一归入"未来展望"章节。

## 核心契约对象

模型层围绕六个核心契约对象构建：

| 对象                | 文件                        | 职责                                                                        |
| :------------------ | :-------------------------- | :-------------------------------------------------------------------------- |
| `BaseRecommender`   | `core/base_model.py`        | 所有模型的抽象基类，定义最小公有接口                                        |
| `NeuralRecommender` | `core/base_model.py`        | 可训练神经模型的基类，继承 BaseRecommender 并增加 forward/compute_loss 要求 |
| `Batch`             | `core/base_model.py`        | 受控的标准 batch 视图，将数据集字典转换为有类型保障的字段访问层             |
| `ModelOutput`       | `core/base_model.py`        | 模型的内部统一输出结构，包含 scores、probs、task_outputs 等字段             |
| `PredictionBundle`  | `core/prediction_bundle.py` | 模型与评估层之间的唯一数据契约，所有 predict() 输出必须遵守                 |
| `Capability`        | `core/base_model.py`        | 能力枚举，Pipeline 据此选择执行路径                                         |

### BaseRecommender

`BaseRecommender` 是所有模型的最薄公共父类，定义 `fit()` 和 `predict()` 两个核心接口，并通过 `supports()` 方法声明能力（如 `Capability.TRAINABLE`）。它不强制所有模型实现梯度训练——非训练模型（如 ItemCF）仍通过基类的通用接口接入 Pipeline。

### NeuralRecommender

`NeuralRecommender` 继承自 `BaseRecommender`，为需要梯度训练的神经模型提供统一的基类。子类必须实现 `forward(batch)` 和 `compute_loss(batch, output)` 两个方法。`count_parameters()` 提供参数量统计。

### Batch — 标准批数据视图

`Batch` 是一个受控的标准 batch 视图，由 DataLoader 产出的原始字典转换而来，提供按语义分层的字段访问能力。字段分为七个层次：

- **标识字段**：`sample_id`、`user_id`、`item_id`、`group_id`、`session_id`
- **监督字段**：`label`、`labels`、`target_item_id`
- **候选字段**：`candidate_item_ids`、`candidate_mask`
- **历史字段**：`history_item_ids`、`history_mask`
- **特征字段**：`user_feats`、`item_feats`、`context_feats`
- **多任务字段**：`task_labels`、`task_masks`
- **多模态字段**：`text_emb`、`image_emb`、`video_emb`

各层字段为可选（`None` 表示缺失），模型可通过 `batch.has("field_name")` 检查字段是否存在。

### PredictionBundle — 统一预测产物

`PredictionBundle` 是模型与评估层之间的唯一数据契约。无论模型来自训练路径还是非训练路径，其 `predict()` 输出必须包装为此结构。核心字段包括：

- `task_type`：任务类型（pointwise / ranking / multitask）
- `problem_type`：问题子类型（binary / multiclass / implicit_ranking 等六种）
- `y_true` 和 `y_score`：真实标签与预测分数列表
- `group_ids`：排序任务的分组标识（ranking 任务必须提供）
- `task_outputs` 和 `task_labels`：多任务场景各任务头的输出与标签字典
- `metadata`：附加元信息字典（数据集名、模型名、样本数等）

`PredictionBundle` 内置 `validate()` 方法，在校验阶段执行任务类型合法性检查（pointwise/ranking/multitask）、问题类型合法性检查、长度一致性校验（y_true 与 y_score）和 ranking 任务分组信息完整性检查。不合规的 bundle 会在评估前被拦截。

## 模型注册与发现

### 注册表机制

RecBench 使用通用 `Registry` 类（`core/registry.py`）管理四种全局注册表：`MODEL_REGISTRY`（模型）、`DATASET_REGISTRY`（数据集）、`METRIC_REGISTRY`（指标）和 `LOSS_REGISTRY`（损失函数）。

`Registry` 支持以装饰器方式注册类并附加任意元信息（键值对）。注册后可通过名称获取类、获取元信息、列出所有注册项、按元信息键值对筛选，以及通过包路径自动发现导入所有子模块以触发注册副作用。

### 模型发现入口

`models/model_registry.py` 提供以下公共 API：

- `auto_discover_models()`：递归导入 `recsys.models` 包下所有子模块，触发各模型的 `@MODEL_REGISTRY.register()` 装饰器执行。返回加载的模块总数。
- `get_model(name)`：按注册键名获取模型类，未注册时抛出 KeyError 并列出所有可用模型。
- `get_model_metadata(name)`：获取模型的注册元信息字典。
- `list_models()`：返回所有已注册模型名称的排序列表。
- `list_models_by_family(family)`：按模型家族筛选。
- `list_models_by_task_type(task_type)`：按任务类型筛选。
- `list_trainable_models()` 和 `list_non_trainable_models()`：按是否支持梯度训练筛选。

### 模型家族与任务类型

当前定义的七个模型家族（`MODEL_FAMILIES`）和三种任务类型（`TASK_TYPES`）：

| 家族键名        | 涵盖领域               | 示例算法                                        |
| :-------------- | :--------------------- | :---------------------------------------------- |
| `classical`     | 传统协同过滤与矩阵分解 | ItemCF, MF, FM, GRU4Rec                         |
| `deep_ctr`      | 深度点击率预估         | DeepFM, DIN, DIEN, DLRM, Wide&Deep, YouTube DNN |
| `sequence`      | 序列推荐               | SASRec, BERT4Rec, SIM, MIMN, ETA, SDIM          |
| `feature_cross` | 特征交叉模型           | DCN, DCNv2, DHEN, RankMixer                     |
| `pcvr`          | 多任务转化率预估       | ESMM, ESM2, DESMM, HM3, DCMT, ChorusCVR, RankUp |
| `unified`       | 统一架构               | HSTU, OneTrans, HyFormer, InterFormer, Wukong   |
| `generative`    | 生成式推荐             | RecGPT, TIGER, IDGenRec, COBRA, GenRec, Molar   |

三种任务类型为 `pointwise`（CTR/CVR 分类）、`ranking`（排序学习）和 `multitask`（多任务学习）。

## 当前真实实现范围

截至当前，源码目录中约 55 个模型文件已经存在，但仅两个模型已完成注册并可通过 Pipeline 运行：

### itemcf — 经典协同过滤基线

ItemCF 是 Sarwar 等人（2001）提出的基于物品的协同过滤算法，属于 `classical` 家族。

**关键特征**：
- 任务类型：`ranking`
- 问题类型：`implicit_ranking`
- 支持训练：否（非训练模型，不依赖梯度优化）
- 通过 `fit()` 构建物品-物品相似度矩阵，通过 `predict()` 为每个用户生成推荐排序

**可配置参数**：
- `similarity`：相似度计算策略，支持 `"cosine"`（余弦相似度）和 `"iuf"`（逆用户频率加权）
- `top_k_neighbors`：相似度矩阵的 Top-K 截断（每个物品只保留最相似的 K 个邻居），默认 50
- `recommend_k`：推荐列表长度，默认 10
- `normalize`：是否对相似度矩阵做最大-最小归一化，默认开启

**与 Pipeline 的集成**：ItemCF 通过非训练路径接入。Pipeline 从数据集切分中提取用户-物品交互对，调用 `fit()` 构建模型，再从 test 切分提取真实标签，调用 `predict()` 生成推荐列表，产出 ranking 类型的 `PredictionBundle`。

**数据集兼容性**：ItemCF 对所有已注册数据集均兼容（movielens_1m、taac2025_1M、taac2025_10M、taac2026_data_sample、taac2026_second_round）。

**配置文件**：`configs/model/classical/itemcf.yaml`

### hyformer — 双塔神经网络

HyFormer 是统一架构家族中的双塔神经网络模型，已通过训练型路径完整验证。

**关键特征**：
- 任务类型：`pointwise`
- 问题类型：`binary`
- 支持训练：是（继承 `NeuralRecommender`，实现 `forward()` 和 `compute_loss()`）
- 双塔结构：用户塔（user embedding + MLP）和物品塔（item embedding + MLP）经余弦相似度计算得分
- 训练时使用 BCE 损失函数（支持 Focal Loss 变体）
- 内置 ID 边界校验，防止因未 Dense Remap 导致的 embedding 越界

**可配置参数**：
- `d_model`：隐藏层维度，默认 64
- `emb_dim`：Embedding 维度，默认 64
- `num_heads`：注意力头数，默认 4
- `num_blocks`：Transformer 块数，默认 2
- `dropout`：Dropout 率，默认 0.1

**稀疏/密集参数分离**：HyFormer 支持将 Embedding 参数（稀疏）与 MLP 参数（密集）分离，分别使用不同的优化器和学习率。Embedding 参数通常使用 Adagrad（学习率 0.05），MLP 参数使用 AdamW（学习率 0.001）。

**Schema Metadata 契约**：Pipeline 在创建 HyFormer 时自动从数据集采集并通过 `schema_metadata` 字典传入以下元信息：
- `num_users` 和 `num_items`：Dense Remap 后的唯一用户/物品数
- `user_id_space` 和 `item_id_space`：ID 空间类型（`dense_1_based` 表示 1-based 连续）
- `padding_idx`：填充槽位索引（固定为 0）
- `max_user_id` 和 `max_item_id`：原始最大 ID 值

**数据集兼容性**：HyFormer 主要适配 TAAC 2026 的点式预估格式（`taac2026_data_sample` 和 `taac2026_second_round`），这些数据集内置 Dense ID Remap，确保 embedding 初始化安全。

**调试脚本**：`scripts/train_hyformer.py` 使用随机内存数据直接训练，不经过数据集适配器和 Pipeline，用于快速验证模型结构的前向/反向传播正确性。

## 状态说明：文件存在 vs 模型已注册

源码目录中七个模型家族下有约 55 个 Python 文件，但需要严格区分两种状态：

- **已注册激活**（2 个）：`itemcf`（classical 家族）和 `hyformer`（unified 家族）。它们的 `@MODEL_REGISTRY.register()` 装饰器已生效，可通过 `get_model()` 获取并实例化。
- **文件已存在但未注册**（约 53 个）：包括 `deepfm.py`、`sasrec.py`、`dcn.py`、`esmm.py`、`hstu.py` 等。这些文件的注册装饰器处于注释状态，目录和文件已创建但无法通过 Pipeline 调用。文件存在不等于模型可用。

文档中引用模型时必须确认其注册状态，不可将目录存在等同于功能完成。

## 模型接入要求

### 通用要求（适用于所有模型）

1. 通过 `@MODEL_REGISTRY.register()` 装饰器完成注册，声明元信息
2. 明确声明继承关系（`BaseRecommender` 或 `NeuralRecommender`）
3. 明确 `task_type` 和 `problem_type`，确保与评估器兼容
4. 实现 `fit()` 和 `predict()` 方法
5. 模型输出可被包装为标准 `PredictionBundle`
6. 在 `configs/model/{family}/{name}.yaml` 下添加对应的 YAML 配置文件
7. 至少有一条聚焦测试覆盖注册、形状或运行时兼容性

### 注册元信息要求

每个模型注册时必须提供的元信息字段（这些字段会被 `get_model_metadata()` 和 `list_models_by_*()` 等 API 消费）：

- `family`：模型家族名称（必须为七个已定义家族之一）
- `modality`：数据模态（如 `sequential`、`tabular`）
- `tasks`：支持的任务类型列表（如 `ranking`、`ctr`）
- `supports_training`：是否支持训练（布尔值）
- `required_features`：所需的输入特征列表
- `default_metrics`：默认评估指标列表

### 训练型模型额外要求

1. 继承 `NeuralRecommender`，实现 `forward(batch: Batch) -> ModelOutput` 和 `compute_loss(batch, output) -> dict`
2. 确认模型结构兼容 `LightningRecommender` 的适配（模型通过 `forward` 接收 Batch 输入，`compute_loss` 返回字典形式的损失）
3. 若使用 Embedding 层，确保数据集已完成 Dense ID Remap（`0` 为填充槽位），或模型侧已做好 ID 边界校验保护

### 非训练型模型额外要求

1. 直接实现 `fit(interactions)` 和 `predict(user_history, ground_truth)` 方法
2. 模型预测结果必须包含分组信息（ranking 场景的 `group_ids`）

## 接入前检查清单

新增模型前确认以下条件均已满足：

1. 数据层已提供模型所需的全部特征
2. 模型输出可以转换为 `PredictionBundle` 的标准字段
3. 默认指标与声明的 `task_type` 一致（ranking 使用排序指标，pointwise 使用分类指标）
4. 配置参数有合理默认值
5. 训练型模型已确认与 Trainer 主干的兼容性（optimizer、scheduler、loss 均可通过配置指定）
6. 若模型使用 Embedding 查找，已确认数据集完成 Dense ID Remap 或模型侧已做边界保护
7. 已在 `configs/model/{family}/{name}.yaml` 添加对应的 YAML 配置文件

## 当前推荐的接入顺序

结合仓库现状，推荐的模型扩展顺序为：

1. 继续以 `itemcf` 为基线稳定 ranking 评估闭环
2. 以 `hyformer` 为样板验证训练型模型的完整流程
3. 逐步激活 `deep_ctr` 家族（DeepFM、DIN 等）的点式预估模型
4. 激活 `sequence` 家族的序列推荐模型（SASRec、BERT4Rec）
5. 逐步扩展多任务（pcvr）和生成式（generative）家族

## 未来展望

以下模型和配置已在源码目录中预留但尚未完成注册激活：

- **classical 家族剩余模型**：矩阵分解（MF）、因子分解机（FM）、GRU4Rec 等三个文件已存在但注册装饰器未激活
- **deep_ctr 家族**：DeepFM、DIN、DIEN、DLRM、Wide&Deep、YouTube DNN 六个文件已存在
- **sequence 家族**：SASRec、BERT4Rec、MIMN、SIM、ETA、SDIM 六个文件已存在
- **feature_cross 家族**：DCN、DCNv2、DHEN、RankMixer 四个文件已存在
- **pcvr 家族**：ESMM、ESM2、DESMM、HM3、DCMT、ChorusCVR、RankUp 等约 15 个文件已存在
- **unified 家族剩余模型**：HSTU、OneTrans、InterFormer、HOMER、MTMixAtt、Longer、Wukong 七个文件已存在
- **generative 家族**：COBRA、GenRec、IDGenRec、RecGPT、Molar 等约 14 个文件已存在

此外，以下配置和文档待补充：

- `configs/model/` 下除 `classical/itemcf.yaml` 外，其余模型家族的 YAML 配置文件尚未创建
- 各模型家族的专属实验页（如 `docs/experiments/baseline-hyformer.md`）待撰写

## 参考

- [核心契约源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/core/base_model.py)
- [预测产物契约](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/core/prediction_bundle.py)
- [模型注册表源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/models/model_registry.py)
- [Pipeline 文档](pipeline.md)
- [配置系统文档](../concepts/configuration.md)
