"""Rating analysis — rating distribution, user/item bias, density.

Provides:
    - Rating value distribution (count and proportion per rating value)
    - User average rating histogram
    - Item average rating histogram
    - User-level rating bias (deviation from global mean, top-N)
    - Item-level rating bias
    - Rating density (per-user rating count percentiles)

Auto-detects rating columns and gracefully skips when absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RATING_COL_CANDIDATES = ["rating", "Rating", "score", "stars"]


def _detect_rating_col(df: pd.DataFrame) -> Optional[str]:
    """Return the first matching rating column name, or None."""
    for cand in _RATING_COL_CANDIDATES:
        if cand in df.columns:
            return cand
    return None


@dataclass
class RatingResult:
    """Rating distribution and bias analysis results."""

    rating_distribution: Dict[float, float]  # rating_value → proportion
    user_avg_rating_stats: Dict[str, float]  # {mean, std, p25, p50, p75, p95}
    item_avg_rating_stats: Dict[str, float]
    global_mean: float
    user_bias_top: List[Tuple[Any, float]]  # [(user_id, bias), ...] top 10 most generous
    user_bias_bottom: List[Tuple[Any, float]]  # top 10 most critical
    item_bias_top: List[Tuple[Any, float]]
    item_bias_bottom: List[Tuple[Any, float]]
    rating_density: Dict[str, float]  # {p50, p75, p95, p99} per-user rating count
    skipped: bool = False
    skip_reason: Optional[str] = None


def analyze(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: Optional[str] = None,
) -> RatingResult:
    """Analyze rating distribution, bias, and density.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    user_col : str
        User ID column name.
    item_col : str
        Item ID column name.
    rating_col : str, optional
        Rating column name. Auto-detected if None.

    Returns
    -------
    RatingResult
    """
    if df.empty:
        return RatingResult(
            rating_distribution={}, user_avg_rating_stats={}, item_avg_rating_stats={},
            global_mean=0.0, user_bias_top=[], user_bias_bottom=[],
            item_bias_top=[], item_bias_bottom=[], rating_density={},
            skipped=True, skip_reason="DataFrame is empty.",
        )

    r_col = rating_col or _detect_rating_col(df)
    if r_col is None or r_col not in df.columns:
        return RatingResult(
            rating_distribution={}, user_avg_rating_stats={}, item_avg_rating_stats={},
            global_mean=0.0, user_bias_top=[], user_bias_bottom=[],
            item_bias_top=[], item_bias_bottom=[], rating_density={},
            skipped=True,
            skip_reason=f"No rating column found (looked for: {', '.join(_RATING_COL_CANDIDATES)}).",
        )

    ratings = pd.to_numeric(df[r_col], errors="coerce").dropna()
    if len(ratings) == 0:
        return RatingResult(
            rating_distribution={}, user_avg_rating_stats={}, item_avg_rating_stats={},
            global_mean=0.0, user_bias_top=[], user_bias_bottom=[],
            item_bias_top=[], item_bias_bottom=[], rating_density={},
            skipped=True, skip_reason="All rating values are non-numeric or NaN.",
        )

    # ---- Rating distribution ----
    counts = ratings.value_counts(normalize=True)
    rating_distribution = {float(k): round(float(v), 4) for k, v in counts.items()}

    # ---- Global mean ----
    global_mean = round(float(ratings.mean()), 4)

    # ---- User avg rating stats ----
    user_avg = pd.DataFrame({user_col: df[user_col], r_col: pd.to_numeric(df[r_col], errors="coerce")})
    user_avg = user_avg.dropna(subset=[r_col])
    user_means = user_avg.groupby(user_col)[r_col].mean()
    user_avg_rating_stats = _compute_distribution_stats(user_means.values)

    # ---- Item avg rating stats ----
    item_avg = pd.DataFrame({item_col: df[item_col], r_col: pd.to_numeric(df[r_col], errors="coerce")})
    item_avg = item_avg.dropna(subset=[r_col])
    item_means = item_avg.groupby(item_col)[r_col].mean()
    item_avg_rating_stats = _compute_distribution_stats(item_means.values)

    # ---- User bias ----
    user_biases = user_means - global_mean
    user_bias_sorted = user_biases.sort_values()
    user_bias_bottom = [(uid, round(float(b), 4)) for uid, b in user_bias_sorted.head(10).items()]
    user_bias_top = [(uid, round(float(b), 4)) for uid, b in user_bias_sorted.tail(10).items()][::-1]

    # ---- Item bias ----
    item_biases = item_means - global_mean
    item_bias_sorted = item_biases.sort_values()
    item_bias_bottom = [(iid, round(float(b), 4)) for iid, b in item_bias_sorted.head(10).items()]
    item_bias_top = [(iid, round(float(b), 4)) for iid, b in item_bias_sorted.tail(10).items()][::-1]

    # ---- Rating density ----
    user_rating_counts = user_avg.groupby(user_col).size()
    density = _compute_distribution_stats(user_rating_counts.values.astype(np.float64))

    logger.info(
        "Rating: %d unique values, global_mean=%.2f, %d users, %d items",
        len(rating_distribution), global_mean,
        len(user_means) if len(user_means) > 0 else 0,
        len(item_means) if len(item_means) > 0 else 0,
    )

    return RatingResult(
        rating_distribution=rating_distribution,
        user_avg_rating_stats=user_avg_rating_stats,
        item_avg_rating_stats=item_avg_rating_stats,
        global_mean=global_mean,
        user_bias_top=user_bias_top,
        user_bias_bottom=user_bias_bottom,
        item_bias_top=item_bias_top,
        item_bias_bottom=item_bias_bottom,
        rating_density=density,
    )


def _compute_distribution_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute mean, std, and percentiles for a numeric array."""
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
    return {
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4) if len(values) > 1 else 0.0,
        "p25": round(float(np.percentile(values, 25)), 4),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p75": round(float(np.percentile(values, 75)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
    }
