"""Distribution analysis — label distribution, feature cardinality, dense feature stats."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Cardinality bin boundaries
_CARDINALITY_BINS = [
    (1, 10, "1-10"),
    (11, 100, "11-100"),
    (101, 1000, "101-1K"),
    (1001, 10000, "1K-10K"),
    (10001, 100000, "10K-100K"),
    (100001, float("inf"), "100K+"),
]


@dataclass
class DistributionResult:
    """Distribution analysis results."""

    label_distribution: Dict[int, float]  # label_value → proportion
    feature_cardinality: Dict[str, int]  # per-column unique count
    cardinality_bins: Dict[str, int]  # bin_label → column count
    dense_stats: Dict[str, Dict[str, float]]  # dense col → {mean, std, min, max, skew, zeros_ratio}
    skipped: bool = False
    skip_reason: Optional[str] = None


def _compute_skew(series: pd.Series) -> float:
    """Compute skewness, returning 0.0 for constant or empty series."""
    n = len(series.dropna())
    if n < 3:
        return 0.0
    return float(series.skew())


def analyze(
    df: pd.DataFrame,
    label_col: str = "label_type",
    dense_pattern: str = "user_dense_",
) -> DistributionResult:
    """Analyze label distribution, feature cardinality, and dense feature statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    label_col : str
        Name of the label column for distribution analysis.
    dense_pattern : str
        Prefix pattern to identify dense (continuous) feature columns.

    Returns
    -------
    DistributionResult
    """
    if df.empty:
        return DistributionResult(
            label_distribution={},
            feature_cardinality={},
            cardinality_bins={},
            dense_stats={},
            skipped=True,
            skip_reason="DataFrame is empty.",
        )

    # ---- label distribution ----
    label_distribution: Dict[int, float] = {}
    if label_col in df.columns:
        label_counts = df[label_col].value_counts(normalize=True)
        for val, prop in label_counts.items():
            label_distribution[int(val)] = round(float(prop), 4)

    # ---- feature cardinality ----
    feature_cardinality: Dict[str, int] = {}
    for col in df.columns:
        try:
            feature_cardinality[col] = int(df[col].nunique(dropna=True))
        except TypeError:
            # Handle unhashable types (e.g. list, numpy.ndarray) by
            # converting to tuples for unique counting.
            feature_cardinality[col] = int(
                df[col].dropna().apply(lambda x: tuple(x) if hasattr(x, "__iter__") else x).nunique()  # type: ignore[arg-type]
            )

    # ---- cardinality bins ----
    cardinality_bins: Dict[str, int] = {label: 0 for _, _, label in _CARDINALITY_BINS}
    for _col, card in feature_cardinality.items():
        for low, high, label in _CARDINALITY_BINS:
            if low <= card <= high:
                cardinality_bins[label] += 1
                break

    # ---- dense feature statistics ----
    dense_stats: Dict[str, Dict[str, float]] = {}
    dense_cols = [c for c in df.columns if c.startswith(dense_pattern)]
    for col in dense_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        if len(valid) == 0:
            dense_stats[col] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "skew": 0.0,
                "zeros_ratio": 1.0,
            }
        else:
            zeros_ratio = round(float((series == 0).sum() / len(series)), 4)
            dense_stats[col] = {
                "mean": round(float(valid.mean()), 4),
                "std": round(float(valid.std()), 4),
                "min": round(float(valid.min()), 4),
                "max": round(float(valid.max()), 4),
                "skew": round(_compute_skew(series), 4),
                "zeros_ratio": zeros_ratio,
            }

    logger.info(
        "Distribution: %d labels, %d features, %d cardinality bins, %d dense cols.",
        len(label_distribution),
        len(feature_cardinality),
        len(cardinality_bins),
        len(dense_stats),
    )

    return DistributionResult(
        label_distribution=label_distribution,
        feature_cardinality=feature_cardinality,
        cardinality_bins=cardinality_bins,
        dense_stats=dense_stats,
    )
