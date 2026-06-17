# 数据集 EDA 报告

> **数据集**：taac2026_second_round (1,000 rows)
> **生成时间**：2026-06-17T14:27:11.642751+00:00
> **数据来源**：registry:taac2026_second_round
> **图表目录**：[../../../assets/figures/eda/taac2026_second_round](../../../assets/figures/eda/taac2026_second_round)

## 1. 数据集概况

- **行数**：1,000
- **列数**：142
- **内存占用**：9.7 MB
- **含标签列**：是
- **含时间戳**：是

- **列分组**：core(5), domain_seq(45), item_feat(21), user_feat(71)

### 多模态嵌入分析

以下特征列缺失率极高(>80%)且基数较大(>10)，可能为**跨模态嵌入查找 ID**——仅在特定模态下有值，其他模态下为缺失：

- `item_int_feats_83`：缺失率 88.1%，基数 24
- `item_int_feats_84`：缺失率 88.1%，基数 56
- `item_int_feats_85`：缺失率 88.1%，基数 84

> **建议**：这些列不应做全局缺失值填充。应根据其所属模态分别处理，或作为多模态融合的 gating 信号。

## 2. 列布局概览

**列分组分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/column_layout.echarts.json"></div>


## 3. 行为类型分布

**Label 类型分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/label_distribution.echarts.json"></div>


- `label_type=1`：93.1%
- `label_type=2`：6.9%

## 4. 特征缺失率

**各特征缺失率**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/null_rates.echarts.json"></div>


- **整体缺失率**：13.40%
- **高缺失率 Top-5**：
  - `user_int_feats_101`：92.0%
  - `user_int_feats_102`：90.6%
  - `user_int_feats_103`：89.0%
  - `item_int_feats_83`：88.1%
  - `item_int_feats_84`：88.1%

## 5. 稀疏特征基数

**特征基数**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/cardinality.echarts.json"></div>


**基数区间分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/cardinality_bins.echarts.json"></div>


- **基数区间**：1-10: 31, 11-100: 14, 101-1K: 97

## 6. 特征覆盖率

**特征覆盖率热力图**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/coverage_heatmap.echarts.json"></div>


## 7. 序列长度分布

**域序列长度**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/sequence_lengths.echarts.json"></div>


**序列长度汇总**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/seq_length_summary.echarts.json"></div>


- **domain_a_seq_38**：均值 557，P95=1678，空序列率 0.0%
- **domain_a_seq_39**：均值 557，P95=1678，空序列率 0.0%
- **domain_a_seq_40**：均值 557，P95=1678，空序列率 0.0%
- **domain_a_seq_41**：均值 557，P95=1678，空序列率 0.0%
- **domain_a_seq_42**：均值 557，P95=1678，空序列率 0.0%
- **domain_a_seq_43**：均值 557，P95=1678，空序列率 0.0%
- **domain_a_seq_44**：均值 557，P95=1678，空序列率 0.0%
- **domain_a_seq_45**：均值 557，P95=1678，空序列率 0.0%
- **domain_a_seq_46**：均值 557，P95=1678，空序列率 0.0%
- **domain_b_seq_67**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_68**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_69**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_70**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_71**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_72**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_73**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_74**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_75**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_76**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_77**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_78**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_79**：均值 428，P95=1507，空序列率 0.8%
- **domain_b_seq_88**：均值 428，P95=1507，空序列率 0.8%
- **domain_c_seq_27**：均值 373，P95=1063，空序列率 0.0%
- **domain_c_seq_28**：均值 373，P95=1063，空序列率 0.0%
- **domain_c_seq_29**：均值 373，P95=1063，空序列率 0.0%
- **domain_c_seq_30**：均值 373，P95=1063，空序列率 0.0%
- **domain_c_seq_31**：均值 373，P95=1063，空序列率 0.0%
- **domain_c_seq_32**：均值 373，P95=1063，空序列率 0.0%
- **domain_c_seq_33**：均值 373，P95=1063，空序列率 0.0%
- **domain_c_seq_34**：均值 373，P95=1063，空序列率 0.0%
- **domain_c_seq_35**：均值 373，P95=1063，空序列率 0.0%
- **domain_c_seq_36**：均值 373，P95=1063，空序列率 0.0%
- **domain_c_seq_37**：均值 373，P95=1063，空序列率 0.0%
- **domain_c_seq_47**：均值 373，P95=1063，空序列率 0.0%
- **domain_d_seq_17**：均值 1180，P95=2713，空序列率 2.8%
- **domain_d_seq_18**：均值 1180，P95=2713，空序列率 2.8%
- **domain_d_seq_19**：均值 1180，P95=2713，空序列率 2.8%
- **domain_d_seq_20**：均值 1180，P95=2713，空序列率 2.8%
- **domain_d_seq_21**：均值 1180，P95=2713，空序列率 2.8%
- **domain_d_seq_22**：均值 1180，P95=2713，空序列率 2.8%
- **domain_d_seq_23**：均值 1180，P95=2713，空序列率 2.8%
- **domain_d_seq_24**：均值 1180，P95=2713，空序列率 2.8%
- **domain_d_seq_25**：均值 1180，P95=2713，空序列率 2.8%
- **domain_d_seq_26**：均值 1180，P95=2713，空序列率 2.8%

## 8. 用户 & 物品分析

**用户活跃度**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/user_activity.echarts.json"></div>


- **用户数**：1,000
- **人均交互**：1.0
- **中位数(P50)**：1
- **P95**：1
- **P99**：1

- **物品数**：968
- **平均曝光**：1.0

**跨域用户重叠**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/cross_domain_overlap.echarts.json"></div>


## 9. 特征有效性（单特征 AUC）

**单特征 AUC**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/feature_auc.echarts.json"></div>


- **AUC Top-5 特征**：
  - `item_int_feats_114`：AUC=0.5817
  - `user_int_feats_57`：AUC=0.5495
  - `user_int_feats_109`：AUC=0.5375
  - `item_int_feats_16`：AUC=0.5373
  - `user_int_feats_58`：AUC=0.5360

- **跳过的特征**：85 个

## 10. 缺失值模式

**共缺失特征对**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/co_missing.echarts.json"></div>


**按标签分组缺失率**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/null_rate_by_label.echarts.json"></div>


### 正负样本缺失率对比

**跨标签缺失率差异**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/label_null_diff.echarts.json"></div>


以下特征在正负样本间的缺失率差距最大（|Δ| > 5%），可能存在**标签条件缺失机制 (MNAR)**：

- `user_int_feats_108`：label=1(55.0%) ↔ label=2(39.1%)，|Δ|=15.9%
- `user_int_feats_86`：label=1(71.6%) ↔ label=2(56.5%)，|Δ|=15.1%
- `user_int_feats_94`：label=1(55.5%) ↔ label=2(42.0%)，|Δ|=13.5%
- `user_int_feats_97`：label=1(33.9%) ↔ label=2(21.7%)，|Δ|=12.2%
- `user_int_feats_104`：label=1(41.8%) ↔ label=2(30.4%)，|Δ|=11.3%
- `user_int_feats_96`：label=1(70.8%) ↔ label=2(60.9%)，|Δ|=9.9%
- `item_int_feats_11`：label=1(55.4%) ↔ label=2(46.4%)，|Δ|=9.0%
- `user_int_feats_107`：label=1(37.2%) ↔ label=2(29.0%)，|Δ|=8.2%
- `user_int_feats_99`：label=1(83.5%) ↔ label=2(75.4%)，|Δ|=8.1%
- `user_int_feats_109`：label=1(87.5%) ↔ label=2(79.7%)，|Δ|=7.8%

> **建议**：差异较大的特征需警惕**标签泄露**（训练时可见 test 分布特征）或**选择偏差**（特定标签下特征才被记录）。建模时应避免直接使用含 MNAR 的特征做全局填充。

## 11. 稠密特征分布

**稠密特征分布**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/dense_distributions.echarts.json"></div>


## 12. 序列行为模式

**序列内物品重复率**

<div class="echarts" data-src="../../../assets/figures/eda/taac2026_second_round/seq_repeat_rate.echarts.json"></div>


- **domain_a_seq_38** 重复率：96.23%
- **domain_a_seq_39** 重复率：2.95%
- **domain_a_seq_40** 重复率：97.61%
- **domain_a_seq_41** 重复率：97.98%
- **domain_a_seq_42** 重复率：90.70%
- **domain_a_seq_43** 重复率：90.64%
- **domain_a_seq_44** 重复率：92.18%
- **domain_a_seq_45** 重复率：97.45%
- **domain_a_seq_46** 重复率：94.38%
- **domain_b_seq_67** 重复率：3.37%
- **domain_b_seq_68** 重复率：91.03%
- **domain_b_seq_69** 重复率：46.83%
- **domain_b_seq_70** 重复率：80.68%
- **domain_b_seq_71** 重复率：80.19%
- **domain_b_seq_72** 重复率：81.73%
- **domain_b_seq_73** 重复率：91.61%
- **domain_b_seq_74** 重复率：93.87%
- **domain_b_seq_75** 重复率：89.98%
- **domain_b_seq_76** 重复率：92.11%
- **domain_b_seq_77** 重复率：85.21%
- **domain_b_seq_78** 重复率：75.97%
- **domain_b_seq_79** 重复率：80.64%
- **domain_b_seq_88** 重复率：81.26%
- **domain_c_seq_27** 重复率：0.88%
- **domain_c_seq_28** 重复率：96.50%
- **domain_c_seq_29** 重复率：34.39%
- **domain_c_seq_30** 重复率：79.79%
- **domain_c_seq_31** 重复率：85.08%
- **domain_c_seq_32** 重复率：97.83%
- **domain_c_seq_33** 重复率：98.20%
- **domain_c_seq_34** 重复率：83.55%
- **domain_c_seq_35** 重复率：71.55%
- **domain_c_seq_36** 重复率：61.57%
- **domain_c_seq_37** 重复率：68.70%
- **domain_c_seq_47** 重复率：26.37%
- **domain_d_seq_17** 重复率：98.82%
- **domain_d_seq_18** 重复率：87.68%
- **domain_d_seq_19** 重复率：78.32%
- **domain_d_seq_20** 重复率：80.61%
- **domain_d_seq_21** 重复率：92.90%
- **domain_d_seq_22** 重复率：94.72%
- **domain_d_seq_23** 重复率：38.56%
- **domain_d_seq_24** 重复率：96.45%
- **domain_d_seq_25** 重复率：95.88%
- **domain_d_seq_26** 重复率：39.69%

---

*本报告由 `recsys.data.eda` 模块自动生成。可通过 `uv run recsys-dataset-eda --help` 查看 CLI 选项。*
