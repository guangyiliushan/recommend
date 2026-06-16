---
title: Dataset Guide
description: dataset adapter、split、batch schema、负采样与多模态输入边界
---

# Dataset Guide

## 目标

RecBench 的数据层不应只是“把原始表读进 DataLoader”，而应承担以下职责：

- 把不同来源的数据集统一为可注册、可发现的 adapter
- 把原始数据转换为稳定的 split 语义
- 把样本组织成模型可以消费的 batch schema
- 明确哪些逻辑属于 dataset adapter，哪些属于训练或采样层
- 为 tabular、sequence、多模态和未来的生成式推荐提供可扩展边界

当前仓库已经有 `BaseDataset`、`dataset_registry.py` 和两个真实适配样例 `taac2025.py`、`taac2026.py`，因此这份文档既说明当前实现，也说明推荐演进方向。

## 当前代码结构

当前数据层的核心文件包括：

- `src/recsys/core/base_dataset.py`
- `src/recsys/data/dataset_registry.py`
- `src/recsys/data/datasets/taac2025.py`
- `src/recsys/data/datasets/taac2026.py`

从职责上看：

- `BaseDataset` 提供统一的生命周期和 DataLoader 帮助方法
- `dataset_registry.py` 负责数据集注册
- `taac2025.py` 代表序列型与多模态候选输入
- `taac2026.py` 代表 tabular + sequence 的广告样本输入

## Dataset Adapter 应该负责什么

### 推荐职责

每个 dataset adapter 应负责：

- 原始数据下载或加载
- 原始列到语义字段的映射
- split 构造
- 样本级输出格式定义
- 数据集元信息暴露

### 不推荐承担的职责

dataset adapter 不应承担：

- 模型专属特征拼接逻辑
- 训练时 callback 或 logger 行为
- 与具体 trainer 强耦合的状态管理
- 复杂的实验调度逻辑

一句话说，dataset adapter 负责把“数据集”变成“可消费样本”，但不负责把样本变成某个模型的专用输入流水线。

## 当前 `BaseDataset` 的契约

当前 `BaseDataset` 的核心约定是：

- 子类实现 `_load_raw()`
- 子类实现 `_prepare_splits(raw)`
- 外部通过 `load()` 触发完整流程
- 外部通过 `get_split()` 或 `get_dataloader()` 获取 split

这套契约的优点是简单、统一、容易落地。

但当前实现也有一个值得长期优化的点：

- `BaseDataset` 同时继承了 `torch.utils.data.Dataset`
- 但实际 `TAAC2025Dataset` 和 `TAAC2026Dataset` 都不直接把自己当作训练样本集使用，而是把真正的样本逻辑放在内部 split dataset 中

这说明顶层 `BaseDataset` 更接近“dataset adapter 容器”，而不是“最终训练样本对象”。

## 最佳实践建议

中长期更推荐把 `BaseDataset` 视为 adapter 或 data bundle：

- `BaseDataset`: 负责元信息、加载、切分、DataLoader 工厂
- `Split Dataset`: 负责 `__len__` / `__getitem__`
- `collate_fn`: 负责 batch 级拼接与 padding

这样分层更清晰，也更适合后续支持多种任务形态。

## Split 语义怎么定义

### 统一 split 命名

推荐项目内统一只使用以下 split 名称：

- `train`
- `val`
- `test`
- `full`

当前 `BaseDataset.get_split()` 已经围绕这四种 split 组织，这是正确方向。

### split 的职责边界

split 应代表“实验语义切分”，而不是“任意子集切片”。

也就是说：

- `train` 用于训练
- `val` 用于调参与早停
- `test` 用于最终报告
- `full` 仅用于特定离线分析、索引构建或无监督预处理

不要让 `train/val/test` 同时承担“线上回放窗口”“特征生成窗口”“模型输入窗口”等多重语义，否则后续很难维护。

### 当前实现情况

在当前仓库中：

- `TAAC2025Dataset` 会把用户序列样本打散后按比例分成 train/val/test
- `TAAC2026Dataset` 会把样本行 shuffle 后按比例分成 train/val/test

这是一种可接受的 MVP 做法，但对推荐系统来说仍有风险：

- 它更接近随机划分，而不是严格时间切分
- 容易引入时序泄漏
- 对 ranking、retrieval、sequence 任务不一定公平

## 推荐的 split 策略

建议后续按任务类型明确 split 策略：

### Tabular CTR/CVR

优先考虑：

- 时间切分
- 用户级或曝光时间级切分

避免：

- 完全随机打散后切分

### Sequence Recommendation

优先考虑：

- 以用户行为序列为单位构造训练窗口
- 保留最后一步或最后若干步用于验证与测试

避免：

- 先把所有序列样本完全打散再切分，导致未来信息进入训练

### Retrieval / Candidate Ranking

优先考虑：

- 先定义候选池
- 再按用户或时间切分交互

避免：

- 在切分后临时生成不一致的候选集合

## Batch Schema 应该怎么设计

### 当前现状

当前两个适配器实际上已经展示了两种 batch 风格：

`TAAC2025` 的序列样本大致输出：

- `item_ids`
- `labels`

`TAAC2026` 的 tabular 样本大致输出：

- `user_id`
- `item_id`
- `label`
- `user_feats`
- `item_feats`
- `domain_seqs`

这说明当前仓库已经默认 batch 是“字典形式的张量集合”，这是合理的。

### 推荐统一原则

推荐所有 dataset split 的 `__getitem__()` 都返回 `Dict[str, Tensor]`，但字段命名要遵守统一约定。

建议分成以下层级：

- 标识字段：
  - `user_id`
  - `item_id`
  - `session_id`
- 监督字段：
  - `label`
  - `labels`
  - `target_item_id`
- 稠密或离散特征：
  - `user_feats`
  - `item_feats`
  - `context_feats`
- 序列字段：
  - `history_item_ids`
  - `history_mask`
  - `domain_seqs`
- 多模态字段：
  - `text_emb`
  - `image_emb`
  - `video_emb`
  - `mm_emb`
- 采样字段：
  - `negative_item_ids`
  - `candidate_item_ids`

## 推荐的命名规范

为了避免模型层出现大量 if/else，建议遵守以下命名规则：

- 单标签 pointwise 任务优先用 `label`
- 序列逐位置监督可用 `labels`
- 用户历史优先用 `history_*`
- 候选集合优先用 `candidate_*`
- 负样本优先用 `negative_*`
- mask 字段统一使用 `_mask` 后缀

## 是否要引入 Batch dataclass

当前项目暂时使用字典形式足够灵活，这对早期阶段是合理的。

但从最佳实践看，随着模型增多，更推荐引入一个标准 batch schema 层，例如：

- 顶层仍然返回字典
- 进入模型前转换为标准 batch 对象
- 或在 `collate_fn` 中构造受控的数据结构

这样做的好处是：

- 模型接口更稳定
- 字段缺失时错误更清晰
- 文档、测试和运行时更容易对齐

## 负采样的边界

### 当前现状

当前 `BaseDataset` 已经暴露了 `neg_sample_count` 参数，`TAAC2025` 的内部 split 也保存了这个值，但目前还没有一个统一、显式的负采样边界被真正落地。

这说明仓库已经意识到负采样的重要性，但还没形成统一机制。

### 最佳实践建议

负采样不应散落在各个模型里，也不应完全隐式地塞进 dataset adapter 内部。

更推荐下面的职责划分：

- dataset adapter：定义采样所需的基础语义，如正样本、候选池、用户历史
- sampler 或 preprocessing 层：负责实际负样本生成策略
- batch schema：显式携带负样本字段
- model：只消费负样本，不决定负采样算法

### 为什么不能把负采样都写死在 adapter 里

因为不同任务对负采样的需求差异非常大：

- pointwise CTR 可能不需要额外负采样
- retrieval 经常需要 in-batch negatives 或 candidate negatives
- sequence ranking 可能需要时间一致的 hard negatives
- 多模态模型可能还需要模态对齐采样

如果把采样策略写死在 dataset adapter 里，后续模型一多就会非常混乱。

## 推荐的负采样接口

对当前项目，更好的长期方向是：

- dataset adapter 提供候选池或可采样空间
- `negative_sampling.py` 负责统一策略实现
- 输出 batch 时显式包含：
  - `negative_item_ids`
  - `candidate_item_ids`
  - 必要时包含负样本 mask 或权重

## 多模态输入边界

### 当前现状

`TAAC2025Dataset` 的元信息已经明确包含：

- `user_feat`
- `item_feat`
- `mm_emb`

并且其数据源说明里还出现了：

- text/image/video embedding

但当前实际 batch 输出仍主要是：

- `item_ids`
- `labels`

这说明多模态语义在“元信息层”已经存在，在“训练 batch 层”还没有彻底落地。

### 推荐边界

多模态输入不要让 dataset adapter 直接替模型做融合。

更合理的职责划分是：

- dataset adapter：读取并对齐模态字段，保证索引一致
- batch schema：显式提供各模态张量或 embedding
- model：决定如何使用这些模态

换句话说，adapter 负责“给到”，model 负责“怎么用”。

### 多模态字段设计建议

推荐尽量避免一个含义过宽的单字段 `mm_emb` 长期存在。

更好的长期设计是显式拆开：

- `text_emb`
- `image_emb`
- `video_emb`
- `text_mask`
- `image_mask`
- `video_mask`

如果早期为了兼容性保留 `mm_emb`，也建议在文档中明确其具体拼接方式和维度约定。

## Tabular 与 Sequence 的边界

当前仓库已经能看出两种明显的数据形态：

- `TAAC2026`: 偏 tabular + domain sequence
- `TAAC2025`: 偏 sequence + multimodal retrieval context

因此不建议为所有任务设计一个唯一 batch 结构。

更好的做法是：

- 保留统一命名规范
- 允许不同任务拥有不同字段子集
- 让 evaluator 和 model contract 明确声明自己依赖哪些字段

这样既能统一接口，又不会把所有任务硬压成一个低质量超集。

## Collate 与 Padding 的建议

当前 `TAAC2025` 在 `__getitem__()` 内部做了序列截断和 padding，这对 MVP 有帮助，但长期更推荐把复杂 padding 逻辑放进 `collate_fn`：

- 样本级 `__getitem__()` 返回原始长度信息
- batch 级 `collate_fn` 统一做 padding、mask 和排序

原因是：

- 更节省单样本处理成本
- 更适合变长序列
- mask 生成更集中
- 更利于未来支持 packed sequence、attention mask、梯度检查点等策略

## 数据集元信息应该暴露什么

每个数据集 adapter 最少应暴露：

- `dataset_name`
- `dataset_url`
- `feature_cols`
- `label_col`
- `num_users`
- `num_items`

建议进一步补充：

- `task_types`
- `modalities`
- `split_strategy`
- `supports_negative_sampling`
- `batch_schema_version`

这些元信息对 benchmark、日志和调试都很有帮助。

## Dataset Registry 最佳实践

当前 `dataset_registry.py` 采用显式导入触发注册，这在早期项目中是稳定可控的做法。

但长期需要注意：

- 新增数据集后必须同步注册
- benchmark 配置中引用的 dataset 名称必须和 registry 保持一致
- 数据集注册元信息要足够支撑筛选与校验

对当前项目来说，dataset registry 最好最终支持：

- 按任务筛选
- 按模态筛选
- 按 family 或 benchmark 套件筛选

## 当前仓库的重点改进建议

结合现有实现，我会优先推进下面几件事：

1. 把 `BaseDataset` 明确为 adapter 容器，而不是最终样本集
2. 引入统一 batch 命名规范
3. 把复杂 padding 与组 batch 逻辑迁移到 `collate_fn`
4. 让负采样从隐式参数变成显式批字段
5. 把多模态输入从元信息层推进到明确 batch schema
6. 为 split 策略补充时间一致性和泄漏约束

## 一句话总结

对 RecBench 来说，最佳实践不是让每个数据集随意返回一堆张量，而是：

- 用 dataset adapter 统一加载与切分
- 用 split 统一实验语义
- 用 batch schema 统一模型输入契约
- 用独立采样层管理负采样
- 用清晰字段边界承接多模态输入
