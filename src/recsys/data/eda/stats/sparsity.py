"""Sparsity & cold-start analysis — matrix sparsity, Gini coefficient, concentration.

Provides:
    - Matrix sparsity (interaction density)
    - User / item interaction distribution histograms
    - Gini coefficient and Lorenz curve data for item popularity
    - Cold-start user / item ratio
    - User interaction concentration (top-1%/5%/10%)
    - Long-tail coverage (bottom 50% items' interaction share)

Works with any DataFrame containing user_id and item_id columns.
Gracefully skips when columns are missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from recsys.data.eda.stats.user_item import activity_stats

logger = logging.getLogger(__name__)


@dataclass
class SparsityResult:
    """Sparsity and cold-start analysis results."""

    matrix_sparsity: float  # 1 - n_interactions / (n_users * n_items), range [0,1]
    user_interaction_stats: Dict[str, float]  # {mean, p50, p75, p95, p99, max, total_users}
    item_popularity_stats: Dict[str, float]  # {mean, p50, p95, p99, max, total_items}
    item_gini: float  # Gini coefficient of item popularity, range [0,1]
    cold_start_user_ratio: float  # fraction of users below quantile threshold
    cold_start_item_ratio: float  # fraction of items below quantile threshold
    user_concentration: Dict[str, float]  # {top1pct, top5pct, top10pct} interaction share
    long_tail_coverage: float  # bottom 50% items' interaction share
    item_popularity_sorted: Optional[List[float]]  # sorted counts for Lorenz curve
    skipped: bool = False
    skip_reason: Optional[str] = None


def _compute_gini(sorted_counts: np.ndarray) -> float:
    """Compute Gini coefficient from sorted item popularity counts.

    G = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n + 1) / n

    Parameters
    ----------
    sorted_counts : np.ndarray
        Item popularity counts sorted in ascending order.

    Returns
    -------
    float
        Gini coefficient in [0, 1]. 0 = perfect equality, 1 = extreme inequality.
    """
    n = len(sorted_counts)
    if n == 0 or sorted_counts.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    numerator = 2.0 * (index * sorted_counts).sum()
    denominator = n * sorted_counts.sum()
    gini = numerator / denominator - (n + 1.0) / n
    return float(gini)


def _compute_histogram(counts: np.ndarray, bins: int = 10) -> Dict[str, int]:
    """Compute equal-width histogram bins for a count array.

    Parameters
    ----------
    counts : np.ndarray
        1D array of counts (e.g. user interaction counts).
    bins : int
        Number of equal-width bins.

    Returns
    -------
    Dict[str, int]
        {bin_label: count} where bin_label is "0-5", "6-10", etc.
    """
    if len(counts) == 0 or counts.max() == 0:
        return {}
    bin_width = max(1, int(np.ceil(counts.max() / bins)))
    edges = np.arange(0, counts.max() + bin_width, bin_width)
    hist, _ = np.histogram(counts, bins=edges)
    result: Dict[str, int] = {}
    for i in range(len(hist)):
        if hist[i] > 0:
            label = f"{int(edges[i])}-{int(edges[i + 1]) - 1}"
            result[label] = int(hist[i])
    return result


def analyze(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    cold_start_quantile: float = 0.05,
) -> SparsityResult:
    """Analyze matrix sparsity, Gini coefficient, cold-start, and concentration.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with user-item interactions (one row per interaction).
    user_col : str
        Column name for user ID.
    item_col : str
        Column name for item ID.
    cold_start_quantile : float
        Quantile threshold for cold-start classification (default P5 = 0.05).
        Users/items with interaction count below this quantile are considered
        cold-start.

    Returns
    -------
    SparsityResult
    """
    if df.empty:
        return SparsityResult(
            matrix_sparsity=0.0,
            user_interaction_stats={},
            item_popularity_stats={},
            item_gini=0.0,
            cold_start_user_ratio=0.0,
            cold_start_item_ratio=0.0,
            user_concentration={},
            long_tail_coverage=0.0,
            item_popularity_sorted=None,
            skipped=True,
            skip_reason="DataFrame is empty.",
        )

    has_user = user_col in df.columns
    has_item = item_col in df.columns

    if not has_user or not has_item:
        missing = []
        if not has_user:
            missing.append(user_col)
        if not has_item:
            missing.append(item_col)
        return SparsityResult(
            matrix_sparsity=0.0,
            user_interaction_stats={},
            item_popularity_stats={},
            item_gini=0.0,
            cold_start_user_ratio=0.0,
            cold_start_item_ratio=0.0,
            user_concentration={},
            long_tail_coverage=0.0,
            item_popularity_sorted=None,
            skipped=True,
            skip_reason=f"Required columns not found: {', '.join(missing)}.",
        )

    n_interactions = len(df)
    n_users = int(df[user_col].nunique())
    n_items = int(df[item_col].nunique())

    # ---- Matrix sparsity ----
    total_possible = n_users * n_items
    matrix_sparsity = round(1.0 - n_interactions / total_possible, 6) if total_possible > 0 else 1.0

    # ---- User interaction distribution ----
    user_counts = df[user_col].value_counts().values.astype(np.float64)
    user_stats = activity_stats(user_counts)
    user_stats["total_users"] = float(n_users)
    # add P75 which activity_stats already computes
    user_stats["total_interactions"] = float(n_interactions)

    # ---- Item popularity distribution ----
    item_counts = df[item_col].value_counts().values.astype(np.float64)
    item_stats = activity_stats(item_counts)
    item_stats["total_items"] = float(n_items)
    item_stats["total_interactions"] = float(n_interactions)

    # ---- Gini coefficient ----
    sorted_item_counts = np.sort(item_counts)
    item_gini = round(_compute_gini(sorted_item_counts), 4)

    # ---- Item popularity sorted (for Lorenz curve) ----
    # Cap at 1000 to avoid excessive chart data
    popularity_sorted = sorted_item_counts[-1000:].tolist() if len(sorted_item_counts) > 0 else [0.0]

    # ---- Cold-start ratios ----
    user_quantile_val = float(np.percentile(user_counts, cold_start_quantile * 100))
    item_quantile_val = float(np.percentile(item_counts, cold_start_quantile * 100))
    cold_user_ratio = round(
        float((user_counts <= user_quantile_val).sum() / len(user_counts)), 4
    )
    cold_item_ratio = round(
        float((item_counts <= item_quantile_val).sum() / len(item_counts)), 4
    )

    # ---- User concentration ----
    sorted_users = np.sort(user_counts)[::-1]  # descending
    total = sorted_users.sum()
    concentration: Dict[str, float] = {}
    for pct in [1, 5, 10]:
        k = max(1, int(len(sorted_users) * pct / 100))
        share = round(float(sorted_users[:k].sum() / total), 4) if total > 0 else 0.0
        concentration[f"top{pct}pct"] = share

    # ---- Long-tail coverage ----
    mid = len(sorted_item_counts) // 2
    bottom_half_sum = sorted_item_counts[:mid].sum()
    total_item_interactions = sorted_item_counts.sum()
    long_tail = round(
        float(bottom_half_sum / total_item_interactions), 4
    ) if total_item_interactions > 0 else 0.0

    logger.info(
        "Sparsity: %.4f sparsity, gini=%.4f, %d users, %d items, "
        "cold_users=%.2f%%, cold_items=%.2f%%, long_tail=%.2f%%",
        matrix_sparsity,
        item_gini,
        n_users,
        n_items,
        cold_user_ratio * 100,
        cold_item_ratio * 100,
        long_tail * 100,
    )

    return SparsityResult(
        matrix_sparsity=matrix_sparsity,
        user_interaction_stats=user_stats,
        item_popularity_stats=item_stats,
        item_gini=item_gini,
        cold_start_user_ratio=cold_user_ratio,
        cold_start_item_ratio=cold_item_ratio,
        user_concentration=concentration,
        long_tail_coverage=long_tail,
        item_popularity_sorted=popularity_sorted,
    )
