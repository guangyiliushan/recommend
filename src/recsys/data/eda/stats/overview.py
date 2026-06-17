"""Dataset overview statistics — row/column counts, column group detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Core columns that are not features (TAAC2026 naming convention)
_CORE_COLS = {"user_id", "item_id", "label_type", "label_time", "timestamp"}


@dataclass
class OverviewResult:
    """Dataset overview statistics."""

    total_rows: int
    total_columns: int
    column_groups: Dict[str, List[str]]
    memory_usage_mb: float
    has_label: bool
    has_timestamp: bool
    suspected_multimodal_embeddings: List[str]  # cols with high cardinality + high missing — likely multi-modal embedding lookup keys
    skipped: bool = False
    skip_reason: Optional[str] = None


def analyze(
    df: pd.DataFrame,
    label_col: str = "label_type",
) -> OverviewResult:
    """Analyze dataset overview: shape, column groups, memory footprint.

    Column grouping follows TAAC2026 naming convention:
        - core: user_id, item_id, label_type, label_time, timestamp
        - user_feat: columns starting with user_
        - item_feat: columns starting with item_
        - domain_seq: columns starting with domain_
        - other: columns not matching any pattern

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    label_col : str
        Name of the label column.

    Returns
    -------
    OverviewResult
    """
    if df.empty:
        return OverviewResult(
            total_rows=0,
            total_columns=0,
            column_groups={},
            memory_usage_mb=0.0,
            has_label=False,
            has_timestamp=False,
            suspected_multimodal_embeddings=[],
            skipped=True,
            skip_reason="DataFrame is empty.",
        )

    total_rows = len(df)
    total_columns = len(df.columns)
    columns = list(df.columns)

    # Group columns by naming pattern
    core_cols: List[str] = []
    user_feat_cols: List[str] = []
    item_feat_cols: List[str] = []
    domain_seq_cols: List[str] = []
    other_cols: List[str] = []

    for col in columns:
        if col in _CORE_COLS:
            core_cols.append(col)
        elif col.startswith("user_"):
            user_feat_cols.append(col)
        elif col.startswith("item_"):
            item_feat_cols.append(col)
        elif col.startswith("domain_"):
            domain_seq_cols.append(col)
        else:
            other_cols.append(col)

    column_groups = {
        "core": core_cols,
        "user_feat": user_feat_cols,
        "item_feat": item_feat_cols,
        "domain_seq": domain_seq_cols,
    }
    if other_cols:
        column_groups["other"] = other_cols

    # Memory usage (deep for accurate object-type accounting)
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    has_label = label_col in df.columns
    has_timestamp = "timestamp" in df.columns

    # ---- suspected multi-modal embedding lookups ----
    # Heuristic: item/user integer features with very high missing rate (>80%)
    # and non-trivial cardinality (>10) may be cross-modal embedding IDs
    # that are only populated for one modality.
    suspected_multimodal_embeddings: List[str] = []
    feat_cols = user_feat_cols + item_feat_cols
    for col in feat_cols:
        if not col.startswith(("user_int_", "item_int_")):
            continue  # skip dense features
        null_rate = df[col].isnull().sum() / total_rows if total_rows > 0 else 0.0
        if null_rate > 0.80:
            try:
                card = int(df[col].nunique(dropna=True))
            except TypeError:
                card = int(df[col].dropna().apply(tuple).nunique())
            if card > 10:
                suspected_multimodal_embeddings.append(col)

    logger.info(
        "Overview: %d rows, %d cols, %d user feat, %d item feat, %d domain seq, "
        "%.1f MB, has_label=%s, has_timestamp=%s, %d suspected multimodal emb",
        total_rows,
        total_columns,
        len(user_feat_cols),
        len(item_feat_cols),
        len(domain_seq_cols),
        memory_usage_mb,
        has_label,
        has_timestamp,
        len(suspected_multimodal_embeddings),
    )

    return OverviewResult(
        total_rows=total_rows,
        total_columns=total_columns,
        column_groups=column_groups,
        memory_usage_mb=round(memory_usage_mb, 2),
        has_label=has_label,
        has_timestamp=has_timestamp,
        suspected_multimodal_embeddings=suspected_multimodal_embeddings,
    )
