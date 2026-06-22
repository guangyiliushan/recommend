# 数据集 EDA 报告

> **数据集**：movielens_1m (796,244 rows)
> **生成时间**：2026-06-22T10:24:24.903933+00:00
> **数据来源**：registry:movielens_1m
> **原始行数**：796,244（加载时预采样至 500,000 行）
> **图表目录**：[../../../assets/figures/eda/movielens_1m](../../../assets/figures/eda/movielens_1m)

## 1. 数据集概况

- **行数**：500,000
- **列数**：2
- **内存占用**：7.6 MB
- **含标签列**：否
- **含时间戳**：否
- **反馈类型**：implicit
- **序列类型**：none
- **模态**：single

- **列分组**：core(2), domain_seq(0), item_feat(0), user_feat(0)

## 2. 列布局概览

**列分组分布**

<div class="echarts" data-src="../../../assets/figures/eda/movielens_1m/column_layout.echarts.json" style="width:100%;min-height:400px;"></div>


## 3. 行为类型分布

## 4. 特征缺失率

**各特征缺失率**

<div class="echarts" data-src="../../../assets/figures/eda/movielens_1m/null_rates.echarts.json" style="width:100%;min-height:400px;"></div>


- **整体缺失率**：0.00%
- **高缺失率 Top-5**：
  - `user_id`：0.0%
  - `item_id`：0.0%

## 5. 稀疏特征基数

**特征基数**

<div class="echarts" data-src="../../../assets/figures/eda/movielens_1m/cardinality.echarts.json" style="width:100%;min-height:400px;"></div>


**基数区间分布**

<div class="echarts" data-src="../../../assets/figures/eda/movielens_1m/cardinality_bins.echarts.json" style="width:100%;min-height:400px;"></div>


- **基数区间**：1K-10K: 2

## 6. 特征覆盖率

**特征覆盖率热力图**

<div class="echarts" data-src="../../../assets/figures/eda/movielens_1m/coverage_heatmap.echarts.json" style="width:100%;min-height:400px;"></div>


## 7. 序列长度分布

> !!! warning "分析跳过"
> 章节「序列分析」当前数据集不支持此分析 (reason: No columns found matching pattern 'domain_'.)。


## 8. 用户 & 物品分析

**用户活跃度**

<div class="echarts" data-src="../../../assets/figures/eda/movielens_1m/user_activity.echarts.json" style="width:100%;min-height:400px;"></div>


- **用户数**：6,022
- **人均交互**：83.0
- **中位数(P50)**：48
- **P95**：281
- **P99**：454

- **物品数**：3,043
- **平均曝光**：164.3

**用户活跃度分布**

<div class="echarts" data-src="../../../assets/figures/eda/movielens_1m/user_activity_histogram.echarts.json" style="width:100%;min-height:400px;"></div>


**物品流行度分布**

<div class="echarts" data-src="../../../assets/figures/eda/movielens_1m/item_popularity_histogram.echarts.json" style="width:100%;min-height:400px;"></div>


## 9. 特征有效性（单特征 AUC）

> !!! warning "分析跳过"
> 章节「特征有效性」当前数据集不支持此分析 (reason: Label column 'label_type' not found in DataFrame.)。


## 10. 缺失值模式

## 11. 稠密特征分布

## 12. 序列行为模式

## 13. 稀疏度与冷启动分析

**交互密度**

<div class="echarts" data-src="../../../assets/figures/eda/movielens_1m/sparsity_gauge.echarts.json" style="width:100%;min-height:400px;"></div>


- **矩阵稀疏度**：97.2715%
- **交互密度**：2.73%
- **物品**：3,043 个，平均曝光 164.3 次
- **物品流行度 Gini 系数**：0.5629（0=均匀，1=极度集中）
- **冷启动用户占比**：7.2%（交互数 < P5 阈值）
- **冷启动物品占比**：5.4%（交互数 < P5 阈值）
- **用户交互集中度**：
  - top1pct：6.9%
  - top5pct：23.9%
  - top10pct：38.0%
- **长尾覆盖**：12.1%（后 50% 物品贡献的交互占比）
**物品流行度洛伦兹曲线**

<div class="echarts" data-src="../../../assets/figures/eda/movielens_1m/item_lorenz_curve.echarts.json" style="width:100%;min-height:400px;"></div>



## 14. 时序行为分析

> !!! warning "分析跳过"
> 章节「时序分析」当前数据集不支持此分析 (reason: No timestamp column found (looked for: timestamp, time, created_at).)。


## 15. 评分分析

> !!! warning "分析跳过"
> 章节「评分分析」当前数据集不支持此分析 (reason: No rating column found (looked for: rating, Rating, score, stars).)。


---

*本报告由 `recsys.data.eda` 模块自动生成。可通过 `uv run recsys-dataset-eda --help` 查看 CLI 选项。*
