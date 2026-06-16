---
title: Evaluation Guide
description: 当前 evaluator、分类指标、排序指标与可视化能力
---

# Evaluation Guide

## 目标

RecBench 的评估层负责消费统一的预测产物，并输出结构化指标、分任务结果与曲线数据。

当前仓库中的评估层已经由以下文件实现：

- `metrics.py`
- `ranking.py`
- `visualization.py`
- `evaluator.py`

因此本页应围绕“当前已经实现了什么、怎样使用、还有什么边界”来写，而不是把评估层继续描述成骨架。

## 当前主入口

当前 evaluator 主入口是：

```python
evaluate(bundle: PredictionBundle, config: Optional[EvaluationConfig] = None)
```

它会根据 `PredictionBundle.task_type` 自动路由到：

- pointwise
- ranking
- multitask

对应的辅助入口还包括：

- `evaluate_pointwise()`
- `evaluate_ranking()`
- `evaluate_multitask()`

## 输入契约

当前评估层统一消费 `PredictionBundle`。

这意味着模型或 pipeline 不应把“各自定义的随意字典”直接交给 evaluator，而应先把预测结果规范化成统一字段。

常见输入字段包括：

- `task_type`
- `problem_type`
- `y_true`
- `y_score`
- `y_pred`
- `group_ids`
- `candidate_ids`
- `task_outputs`
- `task_labels`
- `metadata`

## 返回契约

当前 evaluator 返回 `EvaluationResult`，主要字段包括：

- `summary_metrics`
- `task_metrics`
- `group_metrics`
- `curve_artifacts`
- `metadata`
- `warnings`
- `errors`

这让 pipeline、benchmark 和 reporter 都可以直接消费评估结果，而不需要重新拼装指标。

## Pointwise 评估

### 当前适用范围

pointwise 任务适用于：

- 二分类
- 多分类
- 单样本独立打分场景

### 当前已实现指标

`metrics.py` 当前已实现并可直接用于文档说明的能力包括：

- 指标别名规范化
- confusion matrix 及派生指标
- `accuracy`
- `precision`
- `recall`
- `f1`
- `balanced_accuracy`
- `specificity`
- `npv`
- `fpr`
- `fnr`
- `roc_auc`
- `pr_auc`
- `average_precision`
- `log_loss`
- `brier_score`

### 当前曲线数据

pointwise 评估还支持导出：

- ROC 曲线点
- PR 曲线点
- threshold sweep

## Ranking 评估

### 当前适用范围

ranking 任务适用于：

- 用户级推荐列表
- 请求级候选排序
- Top-K 质量评估

### 当前已实现指标

`ranking.py` 当前已实现：

- `ndcg@k`
- `mrr`
- `hit_rate@k`
- `recall@k`
- `precision@k`
- `map`

### 当前计算方式

当前 ranking evaluator 的关键特征是：

- 先按 `group_ids` 分组
- 逐组计算指标
- 再跨组聚合

这意味着文档必须明确：当前 ranking 评估不是把所有候选摊平成一维后做全局计算，而是严格依赖分组语义。

### 当前输入要求

ranking 任务至少应提供：

- `task_type = "ranking"`
- `group_ids`
- `candidate_ids`
- `y_true`
- `y_score`

缺少分组信息的 ranking 结果，不符合当前 evaluator 契约。

## Multitask 评估

### 当前适用范围

multitask 任务适用于：

- 同一样本同时预测多个任务头
- CTR/CVR 等联合建模
- 主任务 + 辅助任务的联合输出

### 当前能力

当前 evaluator 已支持按任务头分别评估，而不是把多个任务头粗暴平均成单个总分。

这也是文档中应该重点强调的现实能力。

### 当前输入要求

至少应提供：

- `task_type = "multitask"`
- `task_outputs`
- `task_labels`

可选扩展字段可由 `PredictionBundle.metadata` 承接。

## 可视化与导出

`visualization.py` 当前已经实现两类能力：

### 1. 原始结构化导出

- `export_roc_curve`
- `export_pr_curve`
- `export_threshold_sweep`
- `export_ranking_metrics_at_k`
- `export_metrics_csv`

### 2. 可选绘图

- `plot_roc_curve`
- `plot_pr_curve`
- `plot_metrics_at_k`
- `plot_metric_leaderboard`
- `visualize_classification_results`
- `visualize_ranking_results`

当前文档应把“曲线 JSON/CSV 是第一公民、绘图是可选附加能力”写清楚，而不是只强调图片输出。

## 配置示例

### pointwise 示例

```python
from recsys.evaluation import EvaluationConfig, evaluate

cfg = EvaluationConfig(
    metrics=["roc_auc", "log_loss", "pr_auc"],
    threshold=0.5,
    generate_curves=True,
)

result = evaluate(bundle, cfg)
print(result.summary_metrics)
print(result.curve_artifacts)
```

### ranking 示例

```python
from recsys.evaluation import EvaluationConfig, evaluate

cfg = EvaluationConfig(
    metrics=["ndcg@10", "hit_rate@10", "recall@10", "mrr"],
    ranking_k=[5, 10],
    generate_curves=True,
)

result = evaluate(bundle, cfg)
print(result.summary_metrics)
```

## 与 pipeline 的关系

当前正确的分工是：

- pipeline 负责拿到 `PredictionBundle`
- evaluator 负责计算结构化指标
- visualization 负责生成曲线与图表产物

文档不应再把 evaluator 写成“将来计划由 pipeline 内部实现的一部分”。

## 当前限制

当前必须保留的边界包括：

- evaluator 依赖 `PredictionBundle`，不负责从模型或 dataloader 直接收集预测
- 统计检验字段虽然在配置里预留，但当前文档不应把各种统计检验写成已完整实现能力
- 当前 `evaluation/__init__.py` 与部分内部导入路径存在技术债，但这不影响功能层面“评估能力已实现”的事实

## 当前最重要的结论

RecBench 的评估层当前已经具备 pointwise、ranking、multitask 三条主评估路径，以及曲线数据和基础可视化导出能力。文档中最重要的不是继续讨论“应如何设计 evaluator”，而是准确说明当前 evaluator 已经如何工作、输入输出长什么样、还有哪些边界尚未继续扩展。
