---
title: Evaluation Guide
description: 评估层完整指南 — 点式分类指标、分组排序指标、路由编排与可视化导出
---

# Evaluation Guide

## 概述

RecBench 的评估层负责消费统一的预测产物 `PredictionBundle`，输出结构化指标、分任务结果与曲线数据。它不直接接触模型或 DataLoader——所有评估输入必须已经规范化为 `PredictionBundle` 的标准字段。

评估层采用四层分离架构，各层职责明确、互不交叉：

| 层级         | 文件               | 职责                                                 |
| :----------- | :----------------- | :--------------------------------------------------- |
| 编排路由     | `evaluator.py`     | 校验输入、按 task_type 分流、组装结构化输出          |
| 点式分类指标 | `metrics.py`       | 三层指标计算（原子统计 → 派生指标 → 阈值无关）       |
| 分组排序指标 | `ranking.py`       | 逐组计算再跨组聚合的排序指标                         |
| 可视化导出   | `visualization.py` | 消费已计算的结构化结果，输出曲线 JSON 文件和可选图表 |

## 评估主入口

评估器的公共入口是 `evaluate()` 函数，接收 `PredictionBundle` 和可选的 `EvaluationConfig` 两个参数。它根据 bundle 的 `task_type` 字段自动分流到三个评估路径：

- `pointwise`：调用 `evaluate_pointwise()`，走分类指标计算
- `ranking`：调用 `evaluate_ranking()`，走排序指标计算
- `multitask`：调用 `evaluate_multitask()`，逐任务头分别评估后汇总

## 输入契约

`PredictionBundle` 是评估层与模型/Pipeline 之间的统一接口。所有预测结果必须先规范化为这个结构，评估器才能正确消费。常见输入字段包括：

- `task_type`：任务类型（pointwise / ranking / multitask）
- `problem_type`：问题子类型（binary / multiclass / implicit_ranking）
- `y_true`：真实标签数组
- `y_score`：预测分数数组
- `y_pred`：离散预测值数组（可选）
- `group_ids`：排序任务的分组标识列表，每个元素对应一个用户或请求
- `candidate_ids`：每个分组的候选物品 ID 列表
- `task_outputs`：多任务场景下各任务头的分数输出字典
- `task_labels`：多任务场景下各任务头的标签字典
- `metadata`：附加元信息字典

缺少分组信息的 ranking 结果不符合当前评估器契约。evaluator 在校验阶段会检查关键字段的完整性，不满足要求时在结果中返回警告而非静默跳过。

## 返回契约

`evaluate()` 返回的 `EvaluationResult` 包含以下关键字段：

- `summary_metrics`：主汇总指标字典，Benchmark 层直接消费此字段进行聚合
- `task_metrics`：分任务/分头指标字典，多任务场景每个任务头独立一组
- `group_metrics`：分段诊断结果，用于更细粒度的分析
- `curve_artifacts`：曲线数据字典（内存中或路径引用），包含 ROC/PR 曲线点等
- `metadata`：评估运行时元信息
- `warnings`：警告列表（非致命问题）
- `errors`：错误列表（致命问题）

Pipeline、Benchmark 和 Reporter 都可以直接消费 `EvaluationResult`，不需要重新拼装指标。

## Pointwise 评估

### 适用范围

Pointwise 任务适用于二分类（CTR 点击率预估）、多分类和多标签场景，每个样本独立打分。

### 当前已实现指标

`metrics.py` 实现了三层计算体系：

**原子统计层**：混淆矩阵与基础计数，包括真阳（TP）、真阴（TN）、假阳（FP）、假阴（FN）四个基础值，所有派生指标均基于这四个值计算。

**派生指标层**：基于混淆矩阵计算的标准分类指标：

- 准确率（Accuracy）
- 精确率（Precision / PPV）
- 召回率（Recall / Sensitivity / TPR）
- F1 分数
- 特异度（Specificity / TNR）
- 阴性预测值（NPV）
- 假阳性率（FPR）
- 假阴性率（FNR）
- 平衡准确率（Balanced Accuracy）

**阈值无关层**：不依赖固定决策阈值的指标：

- ROC-AUC：接收者操作特征曲线下面积
- PR-AUC：精确率-召回率曲线下面积
- 平均精确率（Average Precision）
- 对数损失（Log Loss）
- Brier 分数
- ROC 曲线数据点
- PR 曲线数据点

### 指标别名系统

所有指标使用规范英文键名（如 `accuracy`、`roc_auc`、`pr_auc`、`log_loss`），同时提供完整的中英文别名映射表。用户在配置中可以使用简写形式（如 `ACC`、`F1-score`、`ROC AUC` 等），评估器会自动归一化为规范名称。

规范指标列表包括：accuracy、precision、recall、f1、specificity、npv、fpr、fnr、balanced_accuracy、roc_auc、pr_auc、average_precision、log_loss、brier_score。

### 曲线数据

Pointwise 评估支持导出 ROC 曲线点、PR 曲线点和阈值扫描数据。这些曲线数据以结构化的 JSON 格式输出，原始数据点优先于图片文件。

### 配置要点

评估配置 `EvaluationConfig` 中与 pointwise 相关的关键参数为：`metrics`（指定要计算的指标列表）、`threshold`（分类决策阈值，默认 0.5）、`threshold_strategy`（阈值策略，支持 fixed / bundle / tuned_on_val / per_task 四种）和 `generate_curves`（是否生成曲线数据）。

主要指标的默认选择逻辑：pointwise 二分类任务默认使用 `pr_auc` 作为主指标，多分类默认使用 `accuracy`，可通过 `primary_metric` 参数覆盖。

## Ranking 评估

### 适用范围

Ranking 任务适用于用户级推荐列表、请求级候选排序和 Top-K 质量评估等场景。

### 当前已实现指标

`ranking.py` 实现了六种排序指标：

- **NDCG@K**：归一化折损累计增益，衡量排序列表的质量（排名越靠前权重越高）
- **MRR**：平均倒数排名，关注第一个相关项出现的位置
- **HitRate@K**：命中率，候选列表中是否包含至少一个相关项
- **Recall@K**：召回率，相关项中有多少出现在推荐列表中
- **Precision@K**：精确率，推荐列表中有多少是相关项
- **MAP**：平均精确率均值，综合衡量排序精度

### 核心计算原则

Ranking 评估严格遵循"先逐组计算、再跨组聚合"的原则。具体流程为：

1. 按 `group_ids` 将数据分割为独立的用户组
2. 对每个组，使用该组的候选列表和真实标签计算各指标值
3. 跨所有组聚合得到最终指标（均值、标准差等）

这意味着当前 ranking 评估不是把所有候选项摊平成一维后做全局计算，而是严格依赖分组语义。这一设计与推荐系统的真实评估场景一致——每个用户独立评估，再跨用户汇总。

### 对特殊情况处理

Ranking 评估对边界情况有明确的处理策略：空组或候选列表为空的组会产生警告并被跳过；无正样本的组在 NDCG、HitRate 等指标上返回 0；候选数量不足 K 个时以实际候选数作为有效 K 值。

### 配置要点

评估配置中与 ranking 相关的关键参数为：`metrics`（指定 ranking 指标名称）、`ranking_k`（指定 Top-K 值列表，如 `[5, 10, 20]`）。指标名称支持别名归一化（如 `NDCG` 映射为 `ndcg_at_k`，`hitrate` 映射为 `hit_rate_at_k`）。

主要指标的默认选择逻辑：ranking 任务默认使用 `ndcg@K` 作为主指标（K 取 `ranking_k` 列表的第一个值）。

### 输入要求

Ranking 任务至少需要提供 `task_type="ranking"`、`group_ids`（分组标识）、`y_true`（真实标签）和 `y_score`（预测分数）。缺少分组信息的 ranking 结果不符合当前评估器契约，会在校验阶段被拦截。

## Multitask 评估

### 适用范围

Multitask 任务适用于同一样本同时预测多个任务头（如 CTR + CVR）、主任务加辅助任务的联合输出等场景。

### 当前能力

评估器已支持按任务头分别评估，而不是把多个任务头粗暴平均成单个总分。每个任务头独立走对应的评估路径（pointwise 或 ranking），最终汇总在 `task_metrics` 字段中。至少需要提供 `task_type="multitask"`、`task_outputs`（各任务输出字典）和 `task_labels`（各任务标签字典）。

## 可视化与导出

`visualization.py` 实现了两类能力：原始结构化数据导出和可选图表绘制。

### 结构化数据导出

- ROC 曲线数据导出为 JSON 文件
- PR 曲线数据导出为 JSON 文件
- 阈值扫描结果导出为 JSON 文件
- Ranking 指标按 K 值展开导出为 JSON 文件
- 完整评估指标导出为 CSV 文件

### 可选图表绘制

- ROC 曲线图
- PR 曲线图
- 指标按 K 值变化折线图
- 模型指标排行榜图
- 分类结果可视化
- Ranking 结果可视化

图表功能依赖 matplotlib，为可选能力（不可用时优雅降级）。设计原则为原始曲线 JSON/CSV 是第一公民，图表只是附加的展示层产物。可视化模块支持多模型叠加，但要求相同任务类型、数据集和切分。

## Pipeline 与评估层的关系

当前正确的分工是：

- Pipeline 负责从模型和数据集中产出 `PredictionBundle`
- 评估层负责将 bundle 转换为结构化 `EvaluationResult`
- 可视化层负责从 `EvaluationResult` 生成曲线和图表产物

评估层不负责从模型或 DataLoader 直接收集预测，也不实现训练逻辑。这一分工使得评估器可以被 Pipeline、Benchmark 和独立的分析脚本共用。

`EvaluationConfig` 中提供了阈值策略选择，支持从 bundle 中提取阈值的 `bundle` 策略、逐任务的 `per_task` 策略和通过验证集调优的 `tuned_on_val` 策略（后两者为预留字段）。

## 当前限制

- 评估器严格依赖 `PredictionBundle` 的标准化字段，不兼容非标准格式的预测输入
- `statistical_test` 字段在 `EvaluationConfig` 中已预留但当前未实现——Reporter v3 规划中的统计显著性检验尚未落地
- `tuned_on_val` 和 `per_task` 阈值策略为预留值，当前默认使用固定阈值 `fixed` 策略
- 多分类评估路径已定义但尚未经过完整的端到端训练型模型验证
- 可视化图表功能依赖 matplotlib 可选依赖，缺失时所有 `plot_*` 函数优雅降级为无操作

## 未来展望

- **统计显著性检验**：Reporter v3 规划的成对统计检验（如 paired t-test、Wilcoxon），用于多 seed 实验中模型性能差异的显著性判断
- **阈值调优策略落地**：`tuned_on_val` 策略的完整实现需要 Pipeline 层在训练过程中自动搜索最佳阈值并注入 EvaluationConfig
- **感知排序指标**：ERR（Expected Reciprocal Rank）和 AUC-ROC 的 ranking 变体等扩展指标
- **多模态评估**：针对多模态生成式推荐场景的特定评估协议和指标

## 参考

- [评估层源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/evaluation/evaluator.py)
- [Pipeline 文档](pipeline.md)
- [Benchmark 文档](benchmarking.md)
- [产物持久化契约](artifacts.md)
