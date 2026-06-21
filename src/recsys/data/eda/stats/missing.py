"""Missing value analysis — null rates, co-missing patterns, label-conditional null rates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MissingResult:
    """Missing value analysis results."""

    column_missing_rates: Dict[str, float]  # per-column null proportion
    overall_missing_rate: float  # overall null proportion across all cells
    co_missing_pairs: List[Tuple[str, str, float]]  # (col_a, col_b, co_missing_rate), Top-N
    null_rate_by_label: Optional[Dict[int, Dict[str, float]]]  # per-label null rates
    coverage_matrix: Dict[str, float]  # per-column non-null proportion
    label_null_diff: List[Tuple[str, int, int, float]]  # (col, label_a, label_b, |diff|), Top-N — largest null rate gaps across labels
    skipped: bool = False
    skip_reason: Optional[str] = None


def analyze(
    df: pd.DataFrame,
    label_col: Optional[str] = "label_type",
    top_n_co_missing: int = 10,
) -> MissingResult:
    """Analyze missing value patterns across features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    label_col : Optional[str]
        Label column for per-label null rate breakdown. If None or missing,
        null_rate_by_label is set to None.
    top_n_co_missing : int
        Top-N co-missing pairs to return.

    Returns
    -------
    MissingResult
    """
    if df.empty:
        return MissingResult(
            column_missing_rates={},
            overall_missing_rate=0.0,
            co_missing_pairs=[],
            null_rate_by_label=None,
            coverage_matrix={},
            label_null_diff=[],
            skipped=True,
            skip_reason="DataFrame is empty.",
        )

    # ---- per-column missing rates ----
    null_counts = df.isnull().sum()
    total = len(df)
    column_missing_rates = {
        col: round(null_counts[col] / total, 4) for col in df.columns
    }

    # ---- overall missing rate ----
    total_cells = df.size
    total_nulls = null_counts.sum()
    overall_missing_rate = round(total_nulls / total_cells, 4) if total_cells > 0 else 0.0

    # ---- coverage (non-null proportion) ----
    coverage_matrix = {
        col: round(1.0 - column_missing_rates[col], 4) for col in df.columns
    }

    # ---- co-missing pairs ----
    # Only compute for columns with moderate missing (5% - 95%) to avoid trivial pairs
    moderate_missing_cols = [
        col
        for col, rate in column_missing_rates.items()
        if 0.05 < rate < 0.95
    ]
    co_missing_pairs: List[Tuple[str, str, float]] = []

    if len(moderate_missing_cols) >= 2:
        null_mask = df[moderate_missing_cols].isnull()
        for i, col_a in enumerate(moderate_missing_cols):
            for col_b in moderate_missing_cols[i + 1 :]:
                both_null = (null_mask[col_a] & null_mask[col_b]).sum()
                co_rate = round(both_null / total, 4)
                if co_rate > 0:
                    co_missing_pairs.append((col_a, col_b, co_rate))
        # Sort by co-missing rate descending, take top N
        co_missing_pairs.sort(key=lambda x: x[2], reverse=True)
        co_missing_pairs = co_missing_pairs[:top_n_co_missing]

    # ---- per-label null rates ----
    null_rate_by_label: Optional[Dict[int, Dict[str, float]]] = None
    label_null_diff: List[Tuple[str, int, int, float]] = []
    has_label = label_col is not None and label_col in df.columns
    if has_label and label_col is not None:
        null_rate_by_label = {}
        for label_val, group in df.groupby(label_col, observed=False):
            n_group = len(group)
            if n_group > 0:
                group_nulls = group.isnull().sum()
                null_rate_by_label[int(label_val)] = {
                    col: round(group_nulls[col] / n_group, 4) for col in df.columns
                }
        # --- compute per-label null rate differences ---
        label_null_diff = _compute_label_null_diff(null_rate_by_label)

    logger.info(
        "Missing analysis: overall_missing=%.4f, %d moderate-missing cols, "
        "%d co-missing pairs, per-label=%s, %d label-null-diffs",
        overall_missing_rate,
        len(moderate_missing_cols),
        len(co_missing_pairs),
        null_rate_by_label is not None,
        len(label_null_diff),
    )

    return MissingResult(
        column_missing_rates=column_missing_rates,
        overall_missing_rate=overall_missing_rate,
        co_missing_pairs=co_missing_pairs,
        null_rate_by_label=null_rate_by_label,
        coverage_matrix=coverage_matrix,
        label_null_diff=label_null_diff,
    )


def _compute_label_null_diff(
    null_rate_by_label: Dict[int, Dict[str, float]],
) -> List[Tuple[str, int, int, float]]:
    """Compute max null-rate gap per column across labels, return Top-15 largest gaps."""
    if not null_rate_by_label or len(null_rate_by_label) < 2:
        return []

    label_ids = sorted(null_rate_by_label.keys())
    diffs: List[Tuple[str, int, int, float]] = []

    # Get all columns from the first label's key set
    first_label = label_ids[0]
    columns = list(null_rate_by_label[first_label].keys())

    for col in columns:
        rates = [(lid, null_rate_by_label[lid].get(col, 0.0)) for lid in label_ids]
        # Find the pair with max absolute difference
        max_diff = 0.0
        best_pair = (label_ids[0], label_ids[1])
        for i, (lid_a, rate_a) in enumerate(rates):
            for lid_b, rate_b in rates[i + 1 :]:
                diff = abs(rate_a - rate_b)
                if diff > max_diff:
                    max_diff = diff
                    best_pair = (lid_a, lid_b)
        if max_diff >= 0.05:  # only report meaningful gaps (>5%)
            diffs.append((col, best_pair[0], best_pair[1], round(max_diff, 4)))

    # Sort by difference descending, return top 15
    diffs.sort(key=lambda x: x[3], reverse=True)
    return diffs[:15]
