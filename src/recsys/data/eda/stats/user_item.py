"""User & item analysis — user activity distributions, item popularity, cross-domain overlap."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class UserItemResult:
    """User and item analysis results."""

    user_activity: Dict[str, float]  # {mean, p50, p75, p95, p99, max, total_users}
    item_popularity: Dict[str, float]  # {mean, p50, p95, p99, max, total_items}
    cross_domain_overlap: Optional[Dict[str, float]]  # domain pair → overlap ratio
    item_popularity_sorted: Optional[List[float]] = None  # sorted counts for Lorenz curve
    user_activity_histogram: Optional[Dict[str, int]] = None  # 10-bin histogram
    item_popularity_histogram: Optional[Dict[str, int]] = None  # 10-bin histogram
    skipped: bool = False
    skip_reason: Optional[str] = None


def activity_stats(counts: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for user/item counts."""
    if len(counts) == 0:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "mean": round(float(counts.mean()), 2),
        "p50": round(float(np.percentile(counts, 50)), 2),
        "p75": round(float(np.percentile(counts, 75)), 2),
        "p95": round(float(np.percentile(counts, 95)), 2),
        "p99": round(float(np.percentile(counts, 99)), 2),
        "max": float(counts.max()),
    }


def _compute_histogram(counts: np.ndarray, bins: int = 10) -> Dict[str, int]:
    """Compute equal-width histogram bins."""
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
    domain_pattern: str = "domain_",
) -> UserItemResult:
    """Analyze user activity distribution, item popularity, and cross-domain user overlap.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    user_col : str
        Column name for user ID.
    item_col : str
        Column name for item ID.
    domain_pattern : str
        Prefix pattern for domain sequence columns (cross-domain overlap).

    Returns
    -------
    UserItemResult
    """
    if df.empty:
        return UserItemResult(
            user_activity={},
            item_popularity={},
            cross_domain_overlap=None,
            skipped=True,
            skip_reason="DataFrame is empty.",
        )

    # ---- user activity ----
    user_activity: Dict[str, float] = {}
    if user_col in df.columns:
        user_counts = df[user_col].value_counts().values.astype(np.float64)
        user_activity = activity_stats(user_counts)
        user_activity["total_users"] = float(len(user_counts))
        logger.info("User activity: %d users analyzed.", len(user_counts))
    else:
        logger.info("User column '%s' not found, skipping user analysis.", user_col)

    # ---- item popularity ----
    item_popularity: Dict[str, float] = {}
    if item_col in df.columns:
        item_counts = df[item_col].value_counts().values.astype(np.float64)
        item_popularity = activity_stats(item_counts)
        item_popularity["total_items"] = float(len(item_counts))
        logger.info("Item popularity: %d items analyzed.", len(item_counts))
    else:
        logger.info("Item column '%s' not found, skipping item analysis.", item_col)

    # ---- cross-domain user overlap ----
    cross_domain_overlap: Optional[Dict[str, float]] = None
    if user_col in df.columns:
        seq_cols = [c for c in df.columns if c.startswith(domain_pattern)]
        if len(seq_cols) >= 2 and user_col in df.columns:
            # For each domain column, identify users with non-empty sequences
            domain_users: Dict[str, set] = {}
            for col in seq_cols:
                # Users that have non-null, non-empty sequence values
                mask = df[col].apply(
                    lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
                    if x is not None and not (isinstance(x, float) and np.isnan(x))
                    else False
                )
                users_in_domain = set(df.loc[mask, user_col].unique())
                if users_in_domain:
                    domain_users[col] = users_in_domain

            if len(domain_users) >= 2:
                cross_domain_overlap = {}
                domain_names = list(domain_users.keys())
                for i, da in enumerate(domain_names):
                    for db in domain_names[i + 1 :]:
                        users_a = domain_users[da]
                        users_b = domain_users[db]
                        union = len(users_a | users_b)
                        if union > 0:
                            overlap = len(users_a & users_b) / union
                            key = f"{da},{db}"
                            cross_domain_overlap[key] = round(overlap, 4)
                logger.info(
                    "Cross-domain overlap: %d domain pairs analyzed.",
                    len(cross_domain_overlap),
                )
            else:
                logger.info(
                    "Less than 2 domains with users, skipping cross-domain overlap."
                )

    # ---- histograms and sorted popularity ----
    user_activity_histogram: Optional[Dict[str, int]] = None
    item_popularity_histogram: Optional[Dict[str, int]] = None
    item_pop_sorted: Optional[List[float]] = None

    if user_col in df.columns and user_activity:
        user_counts = df[user_col].value_counts().values.astype(np.float64)
        user_activity_histogram = _compute_histogram(user_counts)

    if item_col in df.columns:
        item_counts = df[item_col].value_counts().values.astype(np.float64)
        item_popularity_histogram = _compute_histogram(item_counts)
        sorted_counts = np.sort(item_counts)
        item_pop_sorted = sorted_counts[-1000:].tolist() if len(sorted_counts) > 0 else None

    return UserItemResult(
        user_activity=user_activity,
        item_popularity=item_popularity,
        cross_domain_overlap=cross_domain_overlap,
        item_popularity_sorted=item_pop_sorted,
        user_activity_histogram=user_activity_histogram,
        item_popularity_histogram=item_popularity_histogram,
    )
