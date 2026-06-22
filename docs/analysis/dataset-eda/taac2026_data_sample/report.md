# 数据集 EDA 报告

> **数据集**：taac2026_data_sample (1,000 rows)
> **生成时间**：2026-06-22T10:56:41.277550+00:00
> **数据来源**：registry:taac2026_data_sample
> **图表目录**：[../../../assets/figures/eda/taac2026_data_sample](../../../assets/figures/eda/taac2026_data_sample)

## 1. 数据集概况

- **行数**：1,000
- **列数**：120
- **内存占用**：7.7 MB
- **含标签列**：是
- **含时间戳**：是
- **反馈类型**：implicit_binary
- **序列类型**：domain_seq
- **模态**：multimodal

- **列分组**：core(5), domain_seq(45), item_feat(14), user_feat(56)

### 多模态嵌入分析

以下特征列缺失率极高(>80%)且基数较大(>10)，可能为**跨模态嵌入查找 ID**——仅在特定模态下有值，其他模态下为缺失：

- `item_int_feats_83`：缺失率 83.2%，基数 22
- `item_int_feats_84`：缺失率 83.2%，基数 66
- `item_int_feats_85`：缺失率 83.2%，基数 103

> **建议**：这些列不应做全局缺失值填充。应根据其所属模态分别处理，或作为多模态融合的 gating 信号。

## 2. 列布局概览

**列分组分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/column_layout.echarts.json" style="width:100%;min-height:400px;"></div>


## 3. 行为类型分布

**Label 类型分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/label_distribution.echarts.json" style="width:100%;min-height:400px;"></div>


- `label_type=1`：87.6%
- `label_type=2`：12.4%

## 4. 特征缺失率

**各特征缺失率**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/null_rates.echarts.json" style="width:100%;min-height:400px;"></div>


- **整体缺失率**：14.99%
- **高缺失率 Top-5**：
  - `user_int_feats_101`：91.0%
  - `user_int_feats_102`：87.7%
  - `user_int_feats_103`：86.2%
  - `user_int_feats_109`：85.4%
  - `user_int_feats_100`：84.5%

## 5. 稀疏特征基数

**特征基数**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/cardinality.echarts.json" style="width:100%;min-height:400px;"></div>


**基数区间分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/cardinality_bins.echarts.json" style="width:100%;min-height:400px;"></div>


- **基数区间**：1-10: 30, 11-100: 12, 101-1K: 78

## 6. 特征覆盖率

**特征覆盖率热力图**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/coverage_heatmap.echarts.json" style="width:100%;min-height:400px;"></div>


## 7. 序列长度分布

**域序列长度**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/sequence_lengths.echarts.json" style="width:100%;min-height:400px;"></div>


**序列长度汇总**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/seq_length_summary.echarts.json" style="width:100%;min-height:400px;"></div>


- **domain_a_seq_38**：均值 701，P95=1673，空序列率 0.5%
- **domain_a_seq_39**：均值 701，P95=1673，空序列率 0.5%
- **domain_a_seq_40**：均值 701，P95=1673，空序列率 0.5%
- **domain_a_seq_41**：均值 701，P95=1673，空序列率 0.5%
- **domain_a_seq_42**：均值 701，P95=1673，空序列率 0.5%
- **domain_a_seq_43**：均值 701，P95=1673，空序列率 0.5%
- **domain_a_seq_44**：均值 701，P95=1673，空序列率 0.5%
- **domain_a_seq_45**：均值 701，P95=1673，空序列率 0.5%
- **domain_a_seq_46**：均值 701，P95=1673，空序列率 0.5%
- **domain_b_seq_67**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_68**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_69**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_70**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_71**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_72**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_73**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_74**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_75**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_76**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_77**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_78**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_79**：均值 571，P95=1563，空序列率 1.2%
- **domain_b_seq_88**：均值 571，P95=1563，空序列率 1.2%
- **domain_c_seq_27**：均值 449，P95=1214，空序列率 0.2%
- **domain_c_seq_28**：均值 449，P95=1214，空序列率 0.2%
- **domain_c_seq_29**：均值 449，P95=1214，空序列率 0.2%
- **domain_c_seq_30**：均值 449，P95=1214，空序列率 0.2%
- **domain_c_seq_31**：均值 449，P95=1214，空序列率 0.2%
- **domain_c_seq_32**：均值 449，P95=1214，空序列率 0.2%
- **domain_c_seq_33**：均值 449，P95=1214，空序列率 0.2%
- **domain_c_seq_34**：均值 449，P95=1214，空序列率 0.2%
- **domain_c_seq_35**：均值 449，P95=1214，空序列率 0.2%
- **domain_c_seq_36**：均值 449，P95=1214，空序列率 0.2%
- **domain_c_seq_37**：均值 449，P95=1214，空序列率 0.2%
- **domain_c_seq_47**：均值 449，P95=1214，空序列率 0.2%
- **domain_d_seq_17**：均值 1100，P95=2451，空序列率 8.0%
- **domain_d_seq_18**：均值 1100，P95=2451，空序列率 8.0%
- **domain_d_seq_19**：均值 1100，P95=2451，空序列率 8.0%
- **domain_d_seq_20**：均值 1100，P95=2451，空序列率 8.0%
- **domain_d_seq_21**：均值 1100，P95=2451，空序列率 8.0%
- **domain_d_seq_22**：均值 1100，P95=2451，空序列率 8.0%
- **domain_d_seq_23**：均值 1100，P95=2451，空序列率 8.0%
- **domain_d_seq_24**：均值 1100，P95=2451，空序列率 8.0%
- **domain_d_seq_25**：均值 1100，P95=2451，空序列率 8.0%
- **domain_d_seq_26**：均值 1100，P95=2451，空序列率 8.0%

## 8. 用户 & 物品分析

**用户活跃度**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/user_activity.echarts.json" style="width:100%;min-height:400px;"></div>


- **用户数**：1,000
- **人均交互**：1.0
- **中位数(P50)**：1
- **P95**：1
- **P99**：1

- **物品数**：837
- **平均曝光**：1.2

**跨域用户重叠**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/cross_domain_overlap.echarts.json" style="width:100%;min-height:400px;"></div>


**用户活跃度分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/user_activity_histogram.echarts.json" style="width:100%;min-height:400px;"></div>


**物品流行度分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/item_popularity_histogram.echarts.json" style="width:100%;min-height:400px;"></div>


## 9. 特征有效性（单特征 AUC）

**单特征 AUC**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/feature_auc.echarts.json" style="width:100%;min-height:400px;"></div>


- **AUC Top-5 特征**：
  - `item_int_feats_13`：AUC=0.6111
  - `item_int_feats_9`：AUC=0.5561
  - `item_int_feats_16`：AUC=0.5406
  - `user_int_feats_94`：AUC=0.5376
  - `item_int_feats_6`：AUC=0.5335

- **跳过的特征**：67 个

## 10. 缺失值模式

**共缺失特征对**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/co_missing.echarts.json" style="width:100%;min-height:400px;"></div>


**按标签分组缺失率**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/null_rate_by_label.echarts.json" style="width:100%;min-height:400px;"></div>


### 正负样本缺失率对比

**跨标签缺失率差异**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/label_null_diff.echarts.json" style="width:100%;min-height:400px;"></div>


以下特征在正负样本间的缺失率差距最大（|Δ| > 5%），可能存在**标签条件缺失机制 (MNAR)**：

- `user_int_feats_104`：label=1(38.2%) ↔ label=2(29.8%)，|Δ|=8.4%
- `user_int_feats_96`：label=1(66.9%) ↔ label=2(74.2%)，|Δ|=7.3%
- `user_int_feats_80`：label=1(20.9%) ↔ label=2(13.7%)，|Δ|=7.2%
- `user_int_feats_105`：label=1(31.7%) ↔ label=2(25.0%)，|Δ|=6.7%
- `user_int_feats_92`：label=1(48.6%) ↔ label=2(54.8%)，|Δ|=6.2%

> **建议**：差异较大的特征需警惕**标签泄露**（训练时可见 test 分布特征）或**选择偏差**（特定标签下特征才被记录）。建模时应避免直接使用含 MNAR 的特征做全局填充。

## 11. 稠密特征分布

**稠密特征分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/dense_distributions.echarts.json" style="width:100%;min-height:400px;"></div>


## 12. 序列行为模式

**序列内物品重复率**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/seq_repeat_rate.echarts.json" style="width:100%;min-height:400px;"></div>


- **domain_a_seq_38** 重复率：95.50%
- **domain_a_seq_39** 重复率：2.71%
- **domain_a_seq_40** 重复率：97.71%
- **domain_a_seq_41** 重复率：98.05%
- **domain_a_seq_42** 重复率：91.70%
- **domain_a_seq_43** 重复率：91.13%
- **domain_a_seq_44** 重复率：92.19%
- **domain_a_seq_45** 重复率：96.99%
- **domain_a_seq_46** 重复率：96.21%
- **domain_b_seq_67** 重复率：3.76%
- **domain_b_seq_68** 重复率：94.51%
- **domain_b_seq_69** 重复率：57.78%
- **domain_b_seq_70** 重复率：85.93%
- **domain_b_seq_71** 重复率：85.02%
- **domain_b_seq_72** 重复率：85.56%
- **domain_b_seq_73** 重复率：93.48%
- **domain_b_seq_74** 重复率：94.15%
- **domain_b_seq_75** 重复率：93.44%
- **domain_b_seq_76** 重复率：93.41%
- **domain_b_seq_77** 重复率：88.97%
- **domain_b_seq_78** 重复率：81.77%
- **domain_b_seq_79** 重复率：85.06%
- **domain_b_seq_88** 重复率：84.22%
- **domain_c_seq_27** 重复率：1.19%
- **domain_c_seq_28** 重复率：97.21%
- **domain_c_seq_29** 重复率：35.76%
- **domain_c_seq_30** 重复率：82.31%
- **domain_c_seq_31** 重复率：85.04%
- **domain_c_seq_32** 重复率：98.37%
- **domain_c_seq_33** 重复率：99.19%
- **domain_c_seq_34** 重复率：86.56%
- **domain_c_seq_35** 重复率：74.13%
- **domain_c_seq_36** 重复率：61.85%
- **domain_c_seq_37** 重复率：70.28%
- **domain_c_seq_47** 重复率：27.09%
- **domain_d_seq_17** 重复率：98.87%
- **domain_d_seq_18** 重复率：88.37%
- **domain_d_seq_19** 重复率：78.73%
- **domain_d_seq_20** 重复率：80.30%
- **domain_d_seq_21** 重复率：92.26%
- **domain_d_seq_22** 重复率：94.37%
- **domain_d_seq_23** 重复率：37.47%
- **domain_d_seq_24** 重复率：96.99%
- **domain_d_seq_25** 重复率：97.33%
- **domain_d_seq_26** 重复率：41.14%

## 13. 稀疏度与冷启动分析

**交互密度**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/sparsity_gauge.echarts.json" style="width:100%;min-height:400px;"></div>


- **矩阵稀疏度**：99.8805%
- **交互密度**：0.12%
- **物品**：837 个，平均曝光 1.2 次
- **物品流行度 Gini 系数**：0.1530（0=均匀，1=极度集中）
- **冷启动用户占比**：100.0%（交互数 < P5 阈值）
- **冷启动物品占比**：90.1%（交互数 < P5 阈值）
- **用户交互集中度**：
  - top1pct：1.0%
  - top5pct：5.0%
  - top10pct：10.0%
- **长尾覆盖**：41.8%（后 50% 物品贡献的交互占比）
**物品流行度洛伦兹曲线**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/item_lorenz_curve.echarts.json" style="width:100%;min-height:400px;"></div>



## 14. 时序行为分析

- **时间跨度**：2026-03-05 15:36:40 ～ 2026-03-05 15:49:41（0 天）
- **日均交互**：1000.0
- **峰值日交互**：1,000
**月度交互量趋势**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_data_sample/monthly_volume.echarts.json" style="width:100%;min-height:400px;"></div>



## 15. 评分分析

> !!! warning "分析跳过"
> 章节「评分分析」当前数据集不支持此分析 (reason: No rating column found (looked for: rating, Rating, score, stars).)。


---

*本报告由 `recsys.data.eda` 模块自动生成。可通过 `uv run recsys-dataset-eda --help` 查看 CLI 选项。*
