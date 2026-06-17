"""ECharts JSON renderer — converts stats/* dataclass results into ECharts chart specs.

Design:
    - Each chart function takes a specific result dataclass + SampleMetadata
    - Output is a dict with "echarts_option" (ECharts config) and "_eda_metadata" (audit trail)
    - Renderer does NOT perform any statistical computation
    - Skipped results produce no chart
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from recsys.data.eda.sampler import SampleMetadata
from recsys.data.eda.stats.distribution import DistributionResult
from recsys.data.eda.stats.effectiveness import EffectivenessResult
from recsys.data.eda.stats.missing import MissingResult
from recsys.data.eda.stats.overview import OverviewResult
from recsys.data.eda.stats.sequence import SequenceResult
from recsys.data.eda.stats.user_item import UserItemResult

logger = logging.getLogger(__name__)


@dataclass
class RenderOutput:
    """Renderer output — mapping of chart names to file paths."""

    chart_files: Dict[str, Path]
    chart_count: int


def _make_metadata(ctx: Any, metadata: SampleMetadata) -> Dict[str, Any]:
    """Build the _eda_metadata dictionary from RunContext and SampleMetadata.

    Falls back to sample-only metadata when ctx is None (backward-compatible).
    """
    ctx_fields: Dict[str, Any] = {}
    if ctx is not None:
        ctx_fields = {
            "dataset_id": ctx.dataset_id,
            "dataset_label": ctx.dataset_label,
            "run_tag": ctx.run_tag,
            "generated_at": ctx.generated_at,
        }
    return {
        **ctx_fields,
        "sample_strategy": metadata.sample_strategy,
        "total_rows": metadata.total_rows,
        "sample_ratio": metadata.sample_ratio,
        "strat_rows": metadata.strat_rows,
        "tail_rows": metadata.tail_rows,
        "union_rows": metadata.union_rows,
        "seed": metadata.seed,
        "total_users": metadata.total_users,
        "total_items": metadata.total_items,
    }


def _write_chart(output_dir: Path, filename: str, chart_data: Dict[str, Any]) -> Path:
    """Write a single ECharts JSON file. Returns the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.echarts.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chart_data, f, indent=2, ensure_ascii=False)
    return filepath


# ---------------------------------------------------------------------------
# Chart renderers
# ---------------------------------------------------------------------------


def _render_column_layout(
    overview: OverviewResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Column layout bar chart — count of columns per group."""
    if overview.skipped:
        return None
    groups = overview.column_groups
    categories = list(groups.keys())
    values = [len(groups[g]) for g in categories]
    return {
        "echarts_option": {
            "title": {"text": "Column Layout Overview"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": categories},
            "yAxis": {"type": "value", "name": "Column Count"},
            "series": [{"type": "bar", "data": values}],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_label_distribution(
    distribution: DistributionResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Label type pie chart."""
    if distribution.skipped or not distribution.label_distribution:
        return None
    data = [
        {"name": f"type_{k}", "value": v}
        for k, v in sorted(distribution.label_distribution.items())
    ]
    return {
        "echarts_option": {
            "title": {"text": "Label Type Distribution"},
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "series": [{"type": "pie", "radius": "60%", "data": data}],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_null_rates(
    missing: MissingResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Feature null rate bar chart (descending)."""
    if missing.skipped or not missing.column_missing_rates:
        return None
    sorted_items = sorted(
        missing.column_missing_rates.items(), key=lambda x: x[1], reverse=True
    )
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    return {
        "echarts_option": {
            "title": {"text": "Feature Missing Rates"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": names, "axisLabel": {"rotate": 45}},
            "yAxis": {"type": "value", "name": "Missing Rate"},
            "series": [{"type": "bar", "data": values}],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_cardinality(
    distribution: DistributionResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Feature cardinality bar chart."""
    if distribution.skipped or not distribution.feature_cardinality:
        return None
    sorted_items = sorted(
        distribution.feature_cardinality.items(), key=lambda x: x[1], reverse=True
    )
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    return {
        "echarts_option": {
            "title": {"text": "Feature Cardinality"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": names, "axisLabel": {"rotate": 45}},
            "yAxis": {"type": "value", "name": "Unique Values"},
            "series": [{"type": "bar", "data": values}],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_cardinality_bins(
    distribution: DistributionResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Cardinality bin distribution pie chart."""
    if distribution.skipped or not distribution.cardinality_bins:
        return None
    data = [
        {"name": k, "value": v}
        for k, v in distribution.cardinality_bins.items()
        if v > 0
    ]
    if not data:
        return None
    return {
        "echarts_option": {
            "title": {"text": "Cardinality Bin Distribution"},
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} columns"},
            "series": [{"type": "pie", "radius": "60%", "data": data}],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_coverage_heatmap(
    missing: MissingResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Feature coverage heatmap (1D)."""
    if missing.skipped or not missing.coverage_matrix:
        return None
    sorted_items = sorted(missing.coverage_matrix.items(), key=lambda x: x[1])
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    return {
        "echarts_option": {
            "title": {"text": "Feature Coverage Heatmap"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": names, "axisLabel": {"rotate": 45}},
            "yAxis": {"type": "value", "name": "Coverage", "max": 1.0},
            "series": [{"type": "bar", "data": values}],
            "visualMap": {
                "min": 0,
                "max": 1,
                "orient": "horizontal",
                "text": ["High", "Low"],
            },
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_co_missing(
    missing: MissingResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Co-missing pair chart."""
    if missing.skipped or not missing.co_missing_pairs:
        return None
    pairs = [f"{a} & {b}" for a, b, _ in missing.co_missing_pairs]
    rates = [r for _, _, r in missing.co_missing_pairs]
    return {
        "echarts_option": {
            "title": {"text": "Top Co-Missing Feature Pairs"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": pairs, "axisLabel": {"rotate": 45}},
            "yAxis": {"type": "value", "name": "Co-Missing Rate"},
            "series": [{"type": "bar", "data": rates}],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_null_rate_by_label(
    missing: MissingResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Null rate by label grouped bar chart."""
    if missing.skipped or missing.null_rate_by_label is None:
        return None
    labels = sorted(missing.null_rate_by_label.keys())
    # Pick top-10 columns by average null rate for visual clarity
    col_rates: Dict[str, float] = {}
    for lbl in labels:
        for col, rate in missing.null_rate_by_label[lbl].items():
            if col not in col_rates:
                col_rates[col] = 0.0
            col_rates[col] += rate / len(labels)
    top_cols = sorted(col_rates, key=col_rates.get, reverse=True)[:10]
    if not top_cols:
        return None

    series = []
    for lbl in labels:
        values = [missing.null_rate_by_label[lbl].get(c, 0.0) for c in top_cols]
        series.append({"name": f"label={lbl}", "type": "bar", "data": values})

    return {
        "echarts_option": {
            "title": {"text": "Null Rate by Label"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": [f"label={label_val}" for label_val in labels]},
            "xAxis": {"type": "category", "data": top_cols, "axisLabel": {"rotate": 45}},
            "yAxis": {"type": "value", "name": "Missing Rate"},
            "series": series,
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_label_null_diff(
    missing: MissingResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Top features with largest null rate gap across labels."""
    if missing.skipped or not missing.label_null_diff:
        return None
    diffs = missing.label_null_diff[:15]  # Top 15
    labels = [f"{c}\n({a}↔{b})" for c, a, b, _ in diffs]
    values = [d for _, _, _, d in diffs]
    return {
        "echarts_option": {
            "title": {"text": "Null Rate Gap by Label"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": labels, "axisLabel": {"rotate": 45, "fontSize": 10}},
            "yAxis": {"type": "value", "name": "|Δ Null Rate|"},
            "series": [{"type": "bar", "data": values}],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_dense_distributions(
    distribution: DistributionResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Dense feature statistics table/summary."""
    if distribution.skipped or not distribution.dense_stats:
        return None
    cols = list(distribution.dense_stats.keys())
    means = [distribution.dense_stats[c]["mean"] for c in cols]
    stds = [distribution.dense_stats[c]["std"] for c in cols]
    return {
        "echarts_option": {
            "title": {"text": "Dense Feature Distributions"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": cols, "axisLabel": {"rotate": 45}},
            "yAxis": {"type": "value"},
            "series": [
                {"name": "Mean", "type": "bar", "data": means},
                {"name": "Std", "type": "bar", "data": stds},
            ],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_sequence_lengths(
    sequence: SequenceResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Sequence length distribution per domain."""
    if sequence.skipped or not sequence.has_sequences:
        return None
    domains = list(sequence.domain_lengths.keys())
    means = [sequence.domain_lengths[d]["mean"] for d in domains]
    p95s = [sequence.domain_lengths[d]["p95"] for d in domains]
    return {
        "echarts_option": {
            "title": {"text": "Sequence Length Distribution"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": domains},
            "yAxis": {"type": "value", "name": "Length"},
            "series": [
                {"name": "Mean", "type": "bar", "data": means},
                {"name": "P95", "type": "bar", "data": p95s},
            ],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_seq_length_summary(
    sequence: SequenceResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Sequence length summary table (detailed per-domain stats)."""
    if sequence.skipped or not sequence.has_sequences:
        return None
    domains = list(sequence.domain_lengths.keys())
    # Build a simple radar/table representation
    indicators = [
        "mean", "std", "min", "max", "p50", "p95", "p99", "empty_rate",
    ]
    series_data = []
    for domain in domains:
        vals = [sequence.domain_lengths[domain].get(k, 0.0) for k in indicators]
        series_data.append({"name": domain, "value": vals})
    return {
        "echarts_option": {
            "title": {"text": "Sequence Length Summary"},
            "tooltip": {},
            "legend": {"data": [d.replace("domain_", "") for d in domains]},
            "radar": {
                "indicator": [{"name": i, "max": 1.0} for i in indicators],
            },
            "series": [{"type": "radar", "data": series_data}],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_seq_repeat_rate(
    sequence: SequenceResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Sequence repeat rate per domain."""
    if sequence.skipped or not sequence.has_sequences:
        return None
    domains = list(sequence.seq_repeat_rates.keys())
    values = [sequence.seq_repeat_rates[d] for d in domains]
    return {
        "echarts_option": {
            "title": {"text": "Sequence Repeat Rate"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": domains},
            "yAxis": {"type": "value", "name": "Repeat Rate"},
            "series": [{"type": "bar", "data": values}],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_feature_auc(
    effectiveness: EffectivenessResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Single-feature AUC ranking chart."""
    if effectiveness.skipped or not effectiveness.feature_auc:
        return None
    sorted_items = sorted(
        effectiveness.feature_auc.items(), key=lambda x: x[1], reverse=True
    )
    names = [item[0] for item in sorted_items[:30]]  # Top 30
    values = [item[1] for item in sorted_items[:30]]
    return {
        "echarts_option": {
            "title": {"text": "Single Feature AUC"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": names, "axisLabel": {"rotate": 45}},
            "yAxis": {"type": "value", "name": "AUC", "min": 0, "max": 1},
            "series": [{"type": "bar", "data": values}],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_user_activity(
    user_item: UserItemResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """User activity distribution chart."""
    if user_item.skipped or not user_item.user_activity:
        return None
    keys = [k for k in ["mean", "p50", "p75", "p95", "p99"] if k in user_item.user_activity]
    values = [user_item.user_activity[k] for k in keys]
    return {
        "echarts_option": {
            "title": {"text": "User Activity Distribution"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": keys},
            "yAxis": {"type": "value", "name": "Interactions"},
            "series": [{"type": "bar", "data": values}],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


def _render_cross_domain_overlap(
    user_item: UserItemResult, metadata: SampleMetadata, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Cross-domain user overlap chart."""
    if user_item.skipped or user_item.cross_domain_overlap is None:
        return None
    if not user_item.cross_domain_overlap:
        return None
    pairs = list(user_item.cross_domain_overlap.keys())
    values = [user_item.cross_domain_overlap[p] for p in pairs]
    return {
        "echarts_option": {
            "title": {"text": "Cross-Domain User Overlap"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": pairs, "axisLabel": {"rotate": 45}},
            "yAxis": {"type": "value", "name": "Overlap Ratio", "min": 0, "max": 1},
            "series": [{"type": "bar", "data": values}],
        },
        "_eda_metadata": _make_metadata(kwargs.get("ctx"), metadata),
    }


# ---------------------------------------------------------------------------
# Main render pipeline
# ---------------------------------------------------------------------------

# Chart name �?(renderer function, condition flag)
_CHART_REGISTRY: Dict[str, Any] = {
    "column_layout": _render_column_layout,
    "label_distribution": _render_label_distribution,
    "null_rates": _render_null_rates,
    "cardinality": _render_cardinality,
    "cardinality_bins": _render_cardinality_bins,
    "coverage_heatmap": _render_coverage_heatmap,
    "co_missing": _render_co_missing,
    "null_rate_by_label": _render_null_rate_by_label,
    "label_null_diff": _render_label_null_diff,
    "dense_distributions": _render_dense_distributions,
    "sequence_lengths": _render_sequence_lengths,
    "seq_length_summary": _render_seq_length_summary,
    "seq_repeat_rate": _render_seq_repeat_rate,
    "feature_auc": _render_feature_auc,
    "user_activity": _render_user_activity,
    "cross_domain_overlap": _render_cross_domain_overlap,
}


def render_to_echarts(
    overview: OverviewResult,
    missing: MissingResult,
    distribution: DistributionResult,
    sequence: SequenceResult,
    effectiveness: EffectivenessResult,
    user_item: UserItemResult,
    metadata: SampleMetadata,
    output_dir: Path,
    ctx: Any = None,
) -> RenderOutput:
    """Convert all statistical results to ECharts JSON files.

    Parameters
    ----------
    overview : OverviewResult
    missing : MissingResult
    distribution : DistributionResult
    sequence : SequenceResult
    effectiveness : EffectivenessResult
    user_item : UserItemResult
    metadata : SampleMetadata
        Sampling metadata to embed in each chart.
    output_dir : Path
        Directory to write ECharts JSON files.
    ctx : Any, optional
        RunContext for enriched metadata (dataset_id, run_tag, etc.).

    Returns
    -------
    RenderOutput
    """
    chart_files: Dict[str, Path] = {}
    chart_count = 0

    for name, render_fn in _CHART_REGISTRY.items():
        chart_data = render_fn(
            overview=overview,
            missing=missing,
            distribution=distribution,
            sequence=sequence,
            effectiveness=effectiveness,
            user_item=user_item,
            metadata=metadata,
            ctx=ctx,
        )
        if chart_data is not None:
            filepath = _write_chart(output_dir, name, chart_data)
            chart_files[name] = filepath
            chart_count += 1

    logger.info(
        "Rendered %d ECharts JSON chart(s) to %s", chart_count, output_dir
    )
    return RenderOutput(chart_files=chart_files, chart_count=chart_count)
