# 数据集 EDA 报告

> **数据集**：taac2025_1M (1,001,845 rows)
> **生成时间**：2026-06-22T11:09:12.893330+00:00
> **数据来源**：registry:taac2025_1M
> **原始行数**：1,001,845（加载时预采样至 500,000 行）
> **图表目录**：[../../../assets/figures/eda/taac2025_1m](../../../assets/figures/eda/taac2025_1m)

## 1. 数据集概况

- **行数**：500,000
- **列数**：2
- **内存占用**：61.0 MB
- **含标签列**：否
- **含时间戳**：否
- **反馈类型**：unknown
- **序列类型**：nested_seq
- **模态**：single

- **列分组**：core(1), domain_seq(0), item_feat(0), other(1), user_feat(0)

## 2. 列布局概览

**列分组分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/column_layout.echarts.json" style="width:100%;min-height:400px;"></div>


## 3. 行为类型分布

## 4. 特征缺失率

**各特征缺失率**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/null_rates.echarts.json" style="width:100%;min-height:400px;"></div>


- **整体缺失率**：0.00%
- **高缺失率 Top-5**：
  - `user_id`：0.0%
  - `seq`：0.0%

## 5. 稀疏特征基数

**特征基数**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/cardinality.echarts.json" style="width:100%;min-height:400px;"></div>


**基数区间分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/cardinality_bins.echarts.json" style="width:100%;min-height:400px;"></div>


- **基数区间**：100K+: 2

## 6. 特征覆盖率

**特征覆盖率热力图**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/coverage_heatmap.echarts.json" style="width:100%;min-height:400px;"></div>


## 7. 序列长度分布

**域序列长度**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/sequence_lengths.echarts.json" style="width:100%;min-height:400px;"></div>


**序列长度汇总**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/seq_length_summary.echarts.json" style="width:100%;min-height:400px;"></div>


- **seq**：均值 90，P95=100，空序列率 0.0%

## 8. 用户 & 物品分析

**用户活跃度**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/user_activity.echarts.json" style="width:100%;min-height:400px;"></div>


- **用户数**：500,000
- **人均交互**：1.0
- **中位数(P50)**：1
- **P95**：1
- **P99**：1

**用户活跃度分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/user_activity_histogram.echarts.json" style="width:100%;min-height:400px;"></div>


## 9. 特征有效性（单特征 AUC）

> !!! warning "分析跳过"
> 章节「特征有效性」当前数据集不支持此分析 (reason: Label column 'label_type' not found in DataFrame.)。


## 10. 缺失值模式

## 11. 稠密特征分布

## 12. 序列行为模式

**序列内物品重复率**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/seq_repeat_rate.echarts.json" style="width:100%;min-height:400px;"></div>


- **seq** 重复率：0.23%

## 13. 稀疏度与冷启动分析

> !!! warning "分析跳过"
> 章节「稀疏度分析」当前数据集不支持此分析 (reason: Required columns not found: item_id.)。


## 14. 时序行为分析

> !!! warning "分析跳过"
> 章节「时序分析」当前数据集不支持此分析 (reason: No timestamp column found (looked for: timestamp, time, created_at).)。


## 15. 评分分析

> !!! warning "分析跳过"
> 章节「评分分析」当前数据集不支持此分析 (reason: No rating column found (looked for: rating, Rating, score, stars).)。


---

*本报告由 `recsys.data.eda` 模块自动生成。可通过 `uv run recsys-dataset-eda --help` 查看 CLI 选项。*
