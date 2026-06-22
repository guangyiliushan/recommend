"""Markdown report generator — assembles EDA statistics + chart references into a report.

Output format is compatible with MkDocs + ECharts rendering used in TAAC project docs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from recsys.data.eda.sampler import SampleMetadata
from recsys.data.eda.stats.distribution import DistributionResult
from recsys.data.eda.stats.effectiveness import EffectivenessResult
from recsys.data.eda.stats.missing import MissingResult
from recsys.data.eda.stats.overview import OverviewResult
from recsys.data.eda.stats.sequence import SequenceResult
from recsys.data.eda.stats.user_item import UserItemResult

logger = logging.getLogger(__name__)


def _echarts_div(chart_name: str, title: str, assets_dir: str = "assets/figures/eda") -> str:
    """Generate an ECharts div block for MkDocs.

    Parameters
    ----------
    chart_name : str
        Base name of the chart (without .echarts.json extension).
    title : str
        Section title for the chart.
    assets_dir : str
        Relative path to the assets directory from the report location.

    Returns
    -------
    str
        Markdown + HTML block.
    """
    return (
        f"**{title}**\n\n"
        f'<div class="echarts" data-src="{assets_dir}/{chart_name}.echarts.json"'
        f' style="width:100%;min-height:400px;"></div>\n\n'
    )


def _format_skipped(section_title: str, reason: Optional[str]) -> str:
    """Format a 'skipped' section note."""
    msg = ""
    if reason:
        msg = f" (reason: {reason})"
    return f'> !!! warning "分析跳过"\n> 章节「{section_title}」当前数据集不支持此分析{msg}。\n\n'


def _format_sample_note(metadata: SampleMetadata) -> str:
    """Format sampling metadata as a note block."""
    if metadata.sample_strategy == "none":
        return ""

    lines = [
        "",
        '!!! info "采样说明"',
        "    本报告基于采样数据分析生成：",
        f"    - 原始行数：{metadata.total_rows:,}",
        f"    - 采样比例：{metadata.sample_ratio:.1%}",
        f"    - 采样策略：{metadata.sample_strategy}",
        f"    - 分层采样行数：{metadata.strat_rows:,}",
        f"    - 保尾采样行数：{metadata.tail_rows:,}",
        f"    - 去重后行数：{metadata.union_rows:,}",
        f"    - 随机种子：{metadata.seed}",
    ]
    if metadata.total_users is not None:
        lines.append(f"    - 原始用户数：{metadata.total_users:,}")
    if metadata.total_items is not None:
        lines.append(f"    - 原始物品数：{metadata.total_items:,}")
    lines.append("")
    return "\n".join(lines)


def generate_markdown_report(
    overview: OverviewResult,
    missing: MissingResult,
    distribution: DistributionResult,
    sequence: SequenceResult,
    effectiveness: EffectivenessResult,
    user_item: UserItemResult,
    metadata: SampleMetadata,
    chart_files: Dict[str, Path],
    output_path: Path,
    ctx: Any = None,
) -> Path:
    """Generate a complete dataset EDA Markdown report.

    Parameters
    ----------
    overview : OverviewResult
    missing : MissingResult
    distribution : DistributionResult
    sequence : SequenceResult
    effectiveness : EffectivenessResult
    user_item : UserItemResult
    metadata : SampleMetadata
    chart_files : Dict[str, Path]
        Mapping from chart name to file path. Used to verify chart availability.
    output_path : Path
        Destination path for the Markdown report.
    ctx : Any, optional
        RunContext for enriched metadata and dynamic asset paths.

    Returns
    -------
    Path
        Path to the generated report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    has_chart = lambda name: name in chart_files  # noqa: E731

    # Resolve assets directory relative to report location
    assets_dir = ctx.assets_dir_rel if ctx else "assets/figures/eda"

    # ---- Title ----
    lines.append("# 数据集 EDA 报告\n")

    # ---- Metadata block (if ctx available) ----
    if ctx:
        lines.append(f"> **数据集**：{ctx.dataset_label}")
        lines.append(f"> **生成时间**：{ctx.generated_at}")
        lines.append(f"> **数据来源**：{ctx.source_type}:{ctx.source_ref}")
        if ctx.load_sampled and ctx.load_original_rows > 0:
            lines.append(
                f"> **原始行数**：{ctx.load_original_rows:,}"
                f"（加载时预采样至 {ctx.sample_metadata.union_rows:,} 行）"
            )
        lines.append(f"> **图表目录**：[{assets_dir}]({assets_dir})\n")
    else:
        lines.append("> 本报告由 `recsys-dataset-eda` 自动生成。\n")

    # ---- Sampling note (if applicable) ----
    sample_note = _format_sample_note(metadata)
    if sample_note:
        lines.append(sample_note)

    # ---- 1. Overview ----
    lines.append("## 1. 数据集概况\n")
    if overview.skipped:
        lines.append(_format_skipped("数据集概况", overview.skip_reason))
    else:
        lines.append(f"- **行数**：{overview.total_rows:,}")
        lines.append(f"- **列数**：{overview.total_columns}")
        lines.append(f"- **内存占用**：{overview.memory_usage_mb:.1f} MB")
        lines.append(f"- **含标签列**：{'是' if overview.has_label else '否'}")
        lines.append(f"- **含时间戳**：{'是' if overview.has_timestamp else '否'}\n")

        groups = overview.column_groups
        col_summary = ", ".join(
            f"{g}({len(groups[g])})" for g in sorted(groups.keys())
        )
        lines.append(f"- **列分组**：{col_summary}\n")

    # ---- 1.5 Multi-modal embedding lookups ----
    if overview.suspected_multimodal_embeddings:
        lines.append("### 多模态嵌入分析\n")
        lines.append(
            "以下特征列缺失率极高(>80%)且基数较大(>10)，"
            "可能为**跨模态嵌入查找 ID**——仅在特定模态下有值，"
            "其他模态下为缺失：\n"
        )
        for col in overview.suspected_multimodal_embeddings:
            null_rate = missing.column_missing_rates.get(col, 0.0) if not missing.skipped else 0.0
            card = distribution.feature_cardinality.get(col, 0) if not distribution.skipped else 0
            lines.append(f"- `{col}`：缺失率 {null_rate:.1%}，基数 {card}")
        lines.append("")
        lines.append(
            "> **建议**：这些列不应做全局缺失值填充。应根据其所属模态分别处理，"
            "或作为多模态融合的 gating 信号。\n"
        )

    # ---- 2. Column Layout ----
    lines.append("## 2. 列布局概览\n")
    if has_chart("column_layout"):
        lines.append(_echarts_div("column_layout", "列分组分布", assets_dir))
    else:
        lines.append("_列布局图表未生成。_\n")

    # ---- 3. Label Distribution ----
    lines.append("## 3. 行为类型分布\n")
    if has_chart("label_distribution"):
        lines.append(_echarts_div("label_distribution", "Label 类型分布", assets_dir))
    if not distribution.skipped and distribution.label_distribution:
        for lbl, prop in sorted(distribution.label_distribution.items()):
            lines.append(f"- `label_type={lbl}`：{prop:.1%}")
        lines.append("")

    # ---- 4. Missing Rates ----
    lines.append("## 4. 特征缺失率\n")
    if has_chart("null_rates"):
        lines.append(_echarts_div("null_rates", "各特征缺失率", assets_dir))
    if not missing.skipped:
        lines.append(f"- **整体缺失率**：{missing.overall_missing_rate:.2%}")
        top_missing = sorted(
            missing.column_missing_rates.items(), key=lambda x: x[1], reverse=True
        )[:5]
        if top_missing:
            lines.append("- **高缺失率 Top-5**：")
            for col, rate in top_missing:
                lines.append(f"  - `{col}`：{rate:.1%}")
        lines.append("")

    # ---- 5. Cardinality ----
    lines.append("## 5. 稀疏特征基数\n")
    if has_chart("cardinality"):
        lines.append(_echarts_div("cardinality", "特征基数", assets_dir))
    if has_chart("cardinality_bins"):
        lines.append(_echarts_div("cardinality_bins", "基数区间分布", assets_dir))
    if not distribution.skipped:
        bins = distribution.cardinality_bins
        bin_summary = ", ".join(f"{k}: {v}" for k, v in bins.items() if v > 0)
        lines.append(f"- **基数区间**：{bin_summary}\n")

    # ---- 6. Coverage Heatmap ----
    lines.append("## 6. 特征覆盖率\n")
    if has_chart("coverage_heatmap"):
        lines.append(_echarts_div("coverage_heatmap", "特征覆盖率热力图", assets_dir))

    # ---- 7. Sequence Lengths ----
    lines.append("## 7. 序列长度分布\n")
    if sequence.skipped:
        lines.append(_format_skipped("序列分析", sequence.skip_reason))
    else:
        if has_chart("sequence_lengths"):
            lines.append(_echarts_div("sequence_lengths", "域序列长度", assets_dir))
        if has_chart("seq_length_summary"):
            lines.append(_echarts_div("seq_length_summary", "序列长度汇总", assets_dir))
        # Text summary
        for domain, stats in sequence.domain_lengths.items():
            lines.append(
                f"- **{domain}**：均值 {stats['mean']:.0f}，"
                f"P95={stats['p95']:.0f}，空序列率 {stats['empty_rate']:.1%}"
            )
        lines.append("")

    # ---- 8. User / Item Analysis ----
    lines.append("## 8. 用户 & 物品分析\n")
    if user_item.skipped:
        lines.append(_format_skipped("用户/物品分析", user_item.skip_reason))
    else:
        if has_chart("user_activity"):
            lines.append(_echarts_div("user_activity", "用户活跃度", assets_dir))
        if user_item.user_activity:
            ua = user_item.user_activity
            lines.append(f"- **用户数**：{ua.get('total_users', 0):,.0f}")
            lines.append(f"- **人均交互**：{ua.get('mean', 0):.1f}")
            lines.append(f"- **中位数(P50)**：{ua.get('p50', 0):.0f}")
            lines.append(f"- **P95**：{ua.get('p95', 0):.0f}")
            lines.append(f"- **P99**：{ua.get('p99', 0):.0f}\n")
        if user_item.item_popularity:
            ip = user_item.item_popularity
            lines.append(f"- **物品数**：{ip.get('total_items', 0):,.0f}")
            lines.append(f"- **平均曝光**：{ip.get('mean', 0):.1f}\n")
        if has_chart("cross_domain_overlap") and user_item.cross_domain_overlap:
            lines.append(_echarts_div("cross_domain_overlap", "跨域用户重叠", assets_dir))

    # ---- 9. Feature Effectiveness ----
    lines.append("## 9. 特征有效性（单特征 AUC）\n")
    if effectiveness.skipped:
        lines.append(_format_skipped("特征有效性", effectiveness.skip_reason))
    else:
        if has_chart("feature_auc"):
            lines.append(_echarts_div("feature_auc", "单特征 AUC", assets_dir))
        top_auc = sorted(
            effectiveness.feature_auc.items(), key=lambda x: x[1], reverse=True
        )[:5]
        if top_auc:
            lines.append("- **AUC Top-5 特征**：")
            for col, auc in top_auc:
                lines.append(f"  - `{col}`：AUC={auc:.4f}")
            lines.append("")
        if effectiveness.skipped_features:
            lines.append(f"- **跳过的特征**：{len(effectiveness.skipped_features)} 个")
            lines.append("")

    # ---- 10. Co-missing Patterns ----
    lines.append("## 10. 缺失值模式\n")
    if not missing.skipped and missing.co_missing_pairs and has_chart("co_missing"):
        lines.append(_echarts_div("co_missing", "共缺失特征对", assets_dir))
    if not missing.skipped and missing.null_rate_by_label is not None and has_chart("null_rate_by_label"):
        lines.append(_echarts_div("null_rate_by_label", "按标签分组缺失率", assets_dir))

    # ---- 10.5 Label-conditional null rate diff ----
    if not missing.skipped and missing.label_null_diff:
        lines.append("### 正负样本缺失率对比\n")
        if has_chart("label_null_diff"):
            lines.append(_echarts_div("label_null_diff", "跨标签缺失率差异", assets_dir))
        lines.append(
            "以下特征在正负样本间的缺失率差距最大（|Δ| > 5%），"
            "可能存在**标签条件缺失机制 (MNAR)**：\n"
        )
        for col, la, lb, diff in missing.label_null_diff[:10]:
            rate_a = (
                missing.null_rate_by_label.get(la, {}).get(col, 0.0)
                if missing.null_rate_by_label
                else 0.0
            )
            rate_b = (
                missing.null_rate_by_label.get(lb, {}).get(col, 0.0)
                if missing.null_rate_by_label
                else 0.0
            )
            lines.append(
                f"- `{col}`：label={la}({rate_a:.1%}) ↔ label={lb}({rate_b:.1%})，"
                f"|Δ|={diff:.1%}"
            )
        lines.append("")
        lines.append(
            "> **建议**：差异较大的特征需警惕**标签泄露**（训练时可见 test 分布特征）"
            "或**选择偏差**（特定标签下特征才被记录）。"
            "建模时应避免直接使用含 MNAR 的特征做全局填充。\n"
        )

    # ---- 11. Dense Distributions ----
    lines.append("## 11. 稠密特征分布\n")
    if has_chart("dense_distributions"):
        lines.append(_echarts_div("dense_distributions", "稠密特征分布", assets_dir))

    # ---- 12. Sequence Repeat Rate ----
    lines.append("## 12. 序列行为模式\n")
    if not sequence.skipped and has_chart("seq_repeat_rate"):
        lines.append(_echarts_div("seq_repeat_rate", "序列内物品重复率", assets_dir))
    if not sequence.skipped:
        for domain, rate in sequence.seq_repeat_rates.items():
            lines.append(f"- **{domain}** 重复率：{rate:.2%}")
        lines.append("")

    # ---- Footer ----
    lines.append("---\n")
    lines.append(
        "*本报告由 `recsys.data.eda` 模块自动生成。"
        "可通过 `uv run recsys-dataset-eda --help` 查看 CLI 选项。*\n"
    )

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")

    logger.info("Report written to %s", output_path)
    return output_path


def generate_vector_report(
    vector_result: Any,
    metadata: SampleMetadata,
    chart_files: Dict[str, Path],
    output_path: Path,
    ctx: Any = None,
) -> Path:
    """Generate a vector analysis Markdown report (for mm_emb_* subsets)."""
    from recsys.data.eda.stats.vector import VectorResult

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    has_chart = lambda name: name in chart_files  # noqa: E731
    assets_dir = ctx.assets_dir_rel if ctx else "assets/figures/eda"

    lines.append("# 向量分析报告\n")

    if ctx:
        lines.append(f"> **数据集**: {ctx.dataset_label} / {ctx.subset}")
        lines.append(f"> **生成时间**: {ctx.generated_at}")
        lines.append(f"> **模态**: {getattr(vector_result, 'modality', 'unknown')}\n")

    if not isinstance(vector_result, VectorResult) or vector_result.skipped:
        lines.append(f"> !!! warning \"分析跳过\"\n> {vector_result.skip_reason if vector_result else 'No data'}\n")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    # Vector overview
    lines.append("## 向量概况\n")
    lines.append(f"- **维度**: {vector_result.dim}")
    lines.append(f"- **向量数**: {vector_result.n_vectors:,}")
    if vector_result.sampled_at_load:
        lines.append(f"- **原始向量数**: {vector_result.original_count:,} (加载时预采样)")
    lines.append(f"- **零向量**: {vector_result.zero_vector_count:,} ({vector_result.zero_vector_ratio:.2%})")
    lines.append(f"- **重复率**: {vector_result.duplicate_ratio:.2%}\n")

    # Norm distribution chart
    if has_chart("vector_norms"):
        lines.append(_echarts_div("vector_norms", "范数分布", assets_dir))

    ns = vector_result.norm_stats
    if ns:
        lines.append("### 范数统计\n")
        for key in ["mean", "std", "min", "p25", "p50", "p75", "p95", "max"]:
            val = ns.get(key, 0.0)
            lines.append(f"- **{key}**: {val:.4f}")
        lines.append("")

    # Dimension variance chart
    if has_chart("vector_dim_variance"):
        lines.append(_echarts_div("vector_dim_variance", "维度方差", assets_dir))

    dv = vector_result.dim_variance
    if dv:
        lines.append("### 维度方差统计\n")
        for key, label in [("mean_var", "平均方差"), ("min_var", "最小方差"),
                            ("max_var", "最大方差"), ("mean_abs_mean", "平均|均值|")]:
            val = dv.get(key, 0.0)
            lines.append(f"- **{label}**: {val:.6f}")
        lines.append("")

    lines.append("---\n")
    lines.append("*本报告由 `recsys.data.eda.stats.vector` 模块自动生成。*\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Vector report written to %s", output_path)
    return output_path
