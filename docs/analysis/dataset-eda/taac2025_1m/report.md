# 数据集 EDA 报告

> **数据集**：taac2025_1M (1,001,845 rows)
> **生成时间**：2026-06-17T14:40:00.568498+00:00
> **数据来源**：registry:taac2025_1M
> **原始行数**：1,001,845（加载时预采样至 500,000 行）
> **图表目录**：[../../../assets/figures/eda/taac2025_1m](../../../assets/figures/eda/taac2025_1m)

## 1. 数据集概况

- **行数**：500,000
- **列数**：2
- **内存占用**：61.0 MB
- **含标签列**：否
- **含时间戳**：否

- **列分组**：core(1), domain_seq(0), item_feat(0), other(1), user_feat(0)

## 2. 列布局概览

**列分组分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/column_layout.echarts.json"></div>


## 3. 行为类型分布

## 4. 特征缺失率

**各特征缺失率**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/null_rates.echarts.json"></div>


- **整体缺失率**：0.00%
- **高缺失率 Top-5**：
  - `user_id`：0.0%
  - `seq`：0.0%

## 5. 稀疏特征基数

**特征基数**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/cardinality.echarts.json"></div>


**基数区间分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/cardinality_bins.echarts.json"></div>


- **基数区间**：100K+: 2

## 6. 特征覆盖率

**特征覆盖率热力图**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/coverage_heatmap.echarts.json"></div>


## 7. 序列长度分布

> !!! warning "分析跳过"
> 章节「序列分析」当前数据集不支持此分析 (reason: No columns found matching pattern 'domain_'.)。


## 8. 用户 & 物品分析

**用户活跃度**

<div class="echarts" data-src="../../../assets/figures/eda/taac2025_1m/user_activity.echarts.json"></div>


- **用户数**：500,000
- **人均交互**：1.0
- **中位数(P50)**：1
- **P95**：1
- **P99**：1

## 9. 特征有效性（单特征 AUC）

> !!! warning "分析跳过"
> 章节「特征有效性」当前数据集不支持此分析 (reason: Label column 'label_type' not found in DataFrame.)。


## 10. 缺失值模式

## 11. 稠密特征分布

## 12. 序列行为模式

---

*本报告由 `recsys.data.eda` 模块自动生成。可通过 `uv run recsys-dataset-eda --help` 查看 CLI 选项。*
