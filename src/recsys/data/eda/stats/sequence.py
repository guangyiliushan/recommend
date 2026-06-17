"""Sequence analysis — domain sequence lengths, repeat rates, and domain comparisons."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SequenceResult:
    """Sequence analysis results."""

    domain_lengths: Dict[str, Dict[str, float]]  # domain_name → {mean, p50, p95, empty_rate, ...}
    seq_repeat_rates: Dict[str, float]  # domain_name → repeat_rate
    has_sequences: bool  # True if domain_* columns were found
    skipped: bool = False
    skip_reason: Optional[str] = None


def _compute_length_stats(lengths: np.ndarray, total_rows: int) -> Dict[str, float]:
    """Compute summary statistics for sequence lengths.

    Parameters
    ----------
    lengths : np.ndarray
        1D array of sequence lengths (may contain zeros for empty sequences).
    total_rows : int
        Total number of rows in the DataFrame.

    Returns
    -------
    Dict[str, float]
        Statistics including mean, std, min, max, percentiles, and empty_rate.
    """
    if len(lengths) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "empty_rate": 1.0,
        }

    nonzero = lengths[lengths > 0]
    empty_count = total_rows - len(nonzero)

    return {
        "mean": round(float(lengths.mean()), 2),
        "std": round(float(lengths.std()), 2),
        "min": round(float(lengths.min()), 2),
        "max": round(float(lengths.max()), 2),
        "p50": round(float(np.percentile(lengths, 50)), 2),
        "p95": round(float(np.percentile(lengths, 95)), 2),
        "p99": round(float(np.percentile(lengths, 99)), 2),
        "empty_rate": round(empty_count / total_rows, 4) if total_rows > 0 else 1.0,
    }


def _compute_repeat_rate(series: pd.Series) -> float:
    """Compute intra-sequence item repeat rate.

    repeat_rate = 1 - (unique_items_per_sequence / sequence_length)
    Average across all non-empty sequences.
    """
    total_repeat = 0.0
    count = 0
    for val in series.dropna():
        if isinstance(val, (list, np.ndarray)):
            seq = list(val)
            if len(seq) > 0:
                unique_count = len(set(seq))
                total_repeat += 1.0 - (unique_count / len(seq))
                count += 1
        elif isinstance(val, str):
            # Try parsing as list-like string: "[1, 2, 3]"
            stripped = val.strip("[]")
            items = [x.strip() for x in stripped.split(",") if x.strip()]
            if items:
                unique_count = len(set(items))
                total_repeat += 1.0 - (unique_count / len(items))
                count += 1
    if count == 0:
        return 0.0
    return round(total_repeat / count, 4)


def _try_parse_sequence_value(val) -> Optional[List]:
    """Attempt to parse a single value into a list of ints.

    Handles:
        - Python lists: [1, 2, 3]
        - numpy arrays
        - String representations: "[1, 2, 3]"
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    if isinstance(val, str):
        stripped = val.strip("[]")
        if not stripped:
            return []
        try:
            return [int(x.strip()) for x in stripped.split(",")]
        except (ValueError, TypeError):
            return None
    return None


def analyze(
    df: pd.DataFrame,
    domain_pattern: str = "domain_",
) -> SequenceResult:
    """Analyze domain sequence lengths and intra-sequence repeat rates.

    Automatically detects columns starting with ``domain_pattern``.
    Returns ``has_sequences=False`` and ``skipped=True`` if no such columns exist.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    domain_pattern : str
        Prefix pattern for domain sequence columns (e.g. "domain_").

    Returns
    -------
    SequenceResult
    """
    if df.empty:
        return SequenceResult(
            domain_lengths={},
            seq_repeat_rates={},
            has_sequences=False,
            skipped=True,
            skip_reason="DataFrame is empty.",
        )

    # Detect domain sequence columns
    seq_cols = [c for c in df.columns if c.startswith(domain_pattern)]

    if not seq_cols:
        return SequenceResult(
            domain_lengths={},
            seq_repeat_rates={},
            has_sequences=False,
            skipped=True,
            skip_reason=f"No columns found matching pattern '{domain_pattern}'.",
        )

    total_rows = len(df)

    # ---- per-domain sequence lengths ----
    domain_lengths: Dict[str, Dict[str, float]] = {}
    for col in seq_cols:
        # Parse each value to list and compute length
        lengths_list: List[int] = []
        for val in df[col]:
            parsed = _try_parse_sequence_value(val)
            if parsed is not None:
                lengths_list.append(len(parsed))
            else:
                lengths_list.append(0)
        lengths = np.array(lengths_list, dtype=np.float64)
        domain_lengths[col] = _compute_length_stats(lengths, total_rows)

    # ---- intra-sequence repeat rates ----
    seq_repeat_rates: Dict[str, float] = {}
    for col in seq_cols:
        seq_repeat_rates[col] = _compute_repeat_rate(df[col])

    logger.info(
        "Sequence analysis: %d domain cols found, %d with length stats.",
        len(seq_cols),
        len(domain_lengths),
    )

    return SequenceResult(
        domain_lengths=domain_lengths,
        seq_repeat_rates=seq_repeat_rates,
        has_sequences=True,
    )
