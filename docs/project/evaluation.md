---
title: Evaluation Guide
description: pointwise、ranking、multitask 指标边界与 Evaluator 输入输出契约
---

# Evaluation Guide

## 目标

RecBench 的评估层不应只是“算几个指标”，而应承担以下职责：

- 根据任务类型选择正确的指标族
- 把模型输出转换为统一的评估输入
- 明确 pointwise、ranking、multitask 三类任务的边界
- 生成可复用、可比较、可追溯的评估产物
- 为后续 benchmark 汇总、可视化和统计检验提供稳定契约

当前仓库的 `metrics.py`、`evaluator.py`、`ranking.py` 和 `visualization.py` 仍主要是骨架，因此这份文档既说明推荐设计，也约束后续实现方向。

## 当前代码结构

当前评估层的主要文件包括：

- `src/recsys/evaluation/metrics.py`
- `src/recsys/evaluation/evaluator.py`
- `src/recsys/evaluation/ranking.py`
- `src/recsys/evaluation/visualization.py`

从现有说明可以看出，项目希望支持：

- Classification 指标
- Ranking 指标
- Business 指标
- ROC / PR 曲线
- 统计检验

方向是正确的，但目前最大风险是“所有指标都想一次性算”，导致任务边界混乱。

## Evaluator 应该负责什么

### 推荐职责

Evaluator 应负责：

- 收集模型预测结果
- 将预测结果标准化
- 按任务类型调用正确的指标集合
- 生成结构化评估结果
- 按需触发曲线数据与可视化数据生成

### 不推荐承担的职责

Evaluator 不应承担：

- 模型训练逻辑
- 负采样策略决定
- 数据切分策略决定
- 模型专属后处理分支的大量堆叠
- benchmark 调度本身

一句话说，Evaluator 的职责是“消费预测产物并产出评估产物”，而不是“决定训练、采样或实验编排”。

## 为什么要先分任务边界

推荐系统里“预测”并不只有一种形式。

至少有三类典型任务：

- pointwise：对单条样本输出一个概率、分数或类别
- ranking：对一组候选排序，重点关心 Top-K 表现
- multitask：对同一个样本同时输出多个任务头，例如 CTR + CVR

如果不先把这三类边界拆开，后面很容易出现这些问题：

- 用 pointwise 指标错误评估 ranking 模型
- 用全局 AUC 替代用户级排序质量
- 把多任务头随意平均，结果没有业务意义

因此，RecBench 的评估层首先应按任务类型组织，而不是按函数文件名组织。

## Pointwise 指标边界

### 什么是 pointwise

pointwise 任务的核心特征是：

- 每个样本独立预测
- 输出通常是一个标量概率、分数或类别
- 常见于 CTR、CVR、二分类、多分类等任务

### 典型输入

pointwise 评估通常至少需要：

- `y_true`
- `y_score`

有些指标还需要：

- `y_pred`

其中：

- `y_true` 是真实标签
- `y_score` 是模型输出的概率或连续分数
- `y_pred` 是阈值化后的离散预测

### 推荐指标

pointwise 任务优先考虑：

- `roc_auc`
- `pr_auc`
- `log_loss`
- `brier_score`
- `accuracy`
- `precision`
- `recall`
- `f1`

### 指标使用边界

推荐原则：

- 有类别不平衡时，优先关注 `pr_auc`、`log_loss`、`roc_auc`
- `accuracy` 只能作为补充，不应成为主指标
- 需要阈值的指标必须明确阈值来源，不能隐式默认

### 当前项目中的典型适用场景

例如：

- `TAAC2026` 这类 tabular CTR/CVR 样本更适合 pointwise 评估基线
- `DeepFM`、`WideDeep`、`DLRM` 这类模型天然更容易先接入 pointwise 契约

## Ranking 指标边界

### 什么是 ranking

ranking 任务的核心特征是：

- 一个用户、请求或上下文对应多个候选项
- 模型目标不是单点分类正确，而是把正确项排在前面
- 评估重点在 Top-K 质量，而不是单样本独立准确率

### 典型输入

ranking 评估至少需要：

- query 级或 user 级分组信息
- 每组内的候选项分数
- 每组内的相关性标签或目标项位置

推荐显式包含：

- `group_ids`
- `candidate_ids`
- `y_true`
- `y_score`

必要时还可包含：

- `k_list`
- `weights`

### 推荐指标

ranking 任务优先考虑：

- `ndcg@k`
- `mrr`
- `hit_rate@k`
- `recall@k`
- `map`
- `precision@k`

### 指标使用边界

ranking 评估时最常见的错误是：

- 直接把所有候选样本摊平成一维，然后算全局 AUC
- 忽略 query/user 分组
- 不固定候选集定义

这会让指标失去实际意义。

因此 ranking evaluator 必须明确：

- 分组单位是什么
- 每组候选集如何构造
- Top-K 的 K 列表是什么

### 用户级与全局级的区别

对推荐任务来说，更推荐：

- 先在每个用户或请求级别计算排序结果
- 再在样本组层面聚合

而不是直接做“全局排序后只算一次”。

## Multitask 指标边界

### 什么是 multitask

multitask 任务的核心特征是：

- 同一个样本同时预测多个目标
- 各任务头可能语义不同、分布不同、样本可用性不同

典型例子包括：

- CTR + CVR
- 点击、转化、停留时长等联合建模
- 曝光、点击、购买漏斗多任务

### 典型输入

multitask 评估应显式区分每个任务头的数据，而不是把它们混在一起。

推荐结构包括：

- `task_outputs`
- `task_labels`
- `task_masks`

也可以理解为：

- 每个任务头都有自己的 `y_true`
- 每个任务头都有自己的 `y_score`
- 每个任务头都有自己的可用样本 mask

### 推荐指标

multitask 通常不是一种新指标，而是“多个任务指标集合”。

例如：

- CTR 头：`roc_auc`、`log_loss`
- CVR 头：`roc_auc`、`pr_auc`
- 联合报告：按任务分别输出，不建议直接粗暴平均

### 指标使用边界

多任务最容易犯的错误是：

- 不区分样本可用范围
- 直接把多个任务头的指标简单平均
- 把某个辅助头的指标和主任务指标放在同一优先级

最佳实践是：

- 分任务分别输出
- 明确主任务与辅助任务
- 只有在业务上有清晰解释时才定义总分

## Evaluator 输入契约

### 不推荐的做法

不推荐让 Evaluator 直接接收：

- 任意模型返回的随意字典
- 只靠字段名猜测任务类型
- 没有分组信息的 ranking 输出

这种方式会让模型越多，评估层越混乱。

### 推荐的标准输入

更好的做法是统一一个预测产物契约，可以叫：

- `PredictionBundle`
- 或 `EvaluationBatch`

它不一定必须一开始就做成 dataclass，但字段语义必须稳定。

推荐顶层字段包括：

- `task_type`
- `y_true`
- `y_score`
- `y_pred`
- `group_ids`
- `candidate_ids`
- `task_outputs`
- `task_labels`
- `task_masks`
- `metadata`

## 按任务拆分的输入规范

### Pointwise 输入契约

建议最少包含：

- `task_type = "pointwise"`
- `y_true`
- `y_score`

可选：

- `y_pred`
- `sample_weight`
- `threshold`

### Ranking 输入契约

建议最少包含：

- `task_type = "ranking"`
- `group_ids`
- `candidate_ids`
- `y_true`
- `y_score`

可选：

- `k_list`
- `sample_weight`

### Multitask 输入契约

建议最少包含：

- `task_type = "multitask"`
- `task_outputs`
- `task_labels`

可选：

- `task_masks`
- `task_weights`
- `primary_task`

## Evaluator 输出契约

Evaluator 的输出也不应只是一个扁平 `Dict[str, float]`。

更推荐分层输出，例如包含：

- `summary_metrics`
- `task_metrics`
- `curve_artifacts`
- `group_metrics`
- `metadata`

## 推荐输出结构

### `summary_metrics`

用于 benchmark 排行与主结果展示，例如：

- `primary_metric`
- `roc_auc`
- `ndcg@10`

### `task_metrics`

用于多任务和分任务报告，例如：

- `ctr.roc_auc`
- `cvr.pr_auc`

### `curve_artifacts`

用于保存可视化或原始曲线数据，例如：

- ROC 曲线点
- PR 曲线点
- Top-K 曲线

### `group_metrics`

用于更细粒度诊断，例如：

- 按用户分组
- 按场景分组
- 按样本桶分组

### `metadata`

建议包含：

- `task_type`
- `dataset_name`
- `model_name`
- `num_examples`
- `num_groups`
- `threshold`
- `k_list`

## Evaluator 与 Model Contract 的关系

Evaluator 应依赖模型输出契约，而不是具体模型类。

更准确地说：

- model contract 负责定义模型会产出什么
- evaluator contract 负责定义评估层需要消费什么

两者之间应通过稳定的中间结构对接。

这样做的好处是：

- 新增模型时不需要改 evaluator 核心逻辑
- 评估逻辑可独立测试
- benchmark 更容易复用

## Evaluator 与 Dataset Schema 的关系

Evaluator 不应直接依赖 dataset adapter 的内部实现，但会依赖数据层暴露出的评估必需字段。

例如：

- ranking 需要数据层能提供 query/user 分组语义
- multitask 需要数据层或模型层知道哪些任务标签有效
- pointwise 需要明确 label 与 score 的对应关系

因此数据层、模型层、评估层三者之间需要形成一条稳定链路：

- dataset schema 定义可用监督信息
- model output schema 定义预测结果
- evaluator contract 定义消费方式

## 曲线与可视化边界

当前 `visualization.py` 已经预留了：

- ROC
- PR
- 雷达图
- Leaderboard
- Heatmap
- 训练曲线

最佳实践是把“数值指标计算”和“可视化生成”分开：

- `metrics.py` / `ranking.py` 负责数值计算
- `evaluator.py` 负责组装结果
- `visualization.py` 负责把结构化结果渲染为图

不要让指标函数直接负责画图，否则很难测试和复用。

## 统计检验边界

仓库里已经提到：

- McNemar test
- Wilcoxon signed-rank test
- paired t-test

这些检验不应该默认对所有任务一概使用。

推荐原则：

- pointwise 分类比较可考虑 McNemar
- 排序模型比较更适合基于 query/user 级指标做配对检验
- multitask 需要按任务分别检验

不要把统计检验结果和主指标混成一个单一分数。

## 当前仓库的实现建议

结合现有骨架，我建议按下面顺序推进：

1. 先定义统一的 `PredictionBundle`
2. 实现 pointwise evaluator 最小闭环
3. 实现 ranking evaluator，并强制要求 `group_ids`
4. 实现 multitask evaluator，支持任务头级输出
5. 再接 ROC / PR / leaderboard 等可视化逻辑

## 与当前数据层的衔接建议

结合已有 `datasets.md`，推荐这样连接：

- `TAAC2026` 风格数据优先接入 pointwise evaluator
- `TAAC2025` 风格数据优先接入 sequence / ranking evaluator
- 多任务 PCVR 类模型后续进入 multitask evaluator

这样可以先用最小代价打通三条清晰路径，而不是一开始就追求“一个 evaluator 包打天下”。

## 需要避免的反模式

请尽量避免：

- 用一个超大函数处理所有任务
- ranking 任务没有分组信息
- 多任务头直接平均成一个总数
- pointwise 指标与 ranking 指标混在同一优先级
- 评估结果只返回扁平字典，丢失上下文信息
- 图表生成与数值计算完全耦合

## 一句话总结

对 RecBench 来说，最佳实践不是“把所有指标都实现一遍”，而是：

- 先按 pointwise / ranking / multitask 拆清任务边界
- 再定义统一的 Evaluator 输入输出契约
- 用结构化结果承接数值指标、曲线产物和 benchmark 汇总
