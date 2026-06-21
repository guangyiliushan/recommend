"""Hybrid tail-preserving stratified sampler for large-scale EDA.

Strategy (stratified_tail_preserving):
    1. df_strat: stratified random sample by label_col, preserving class proportions
    2. df_tail: keep ALL rows where item_id frequency < tail threshold (capture long-tail)
    3. Union → drop_duplicates → if over max_rows, uniform trim

Falls back to random + tail-preserving when no label column is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SampleMetadata:
    """Sampling audit metadata — embedded in each ECharts JSON's _eda_metadata field."""

    sample_strategy: str  # "stratified_tail_preserving"
    total_rows: int  # original row count
    sample_ratio: float  # union_rows / total_rows
    strat_rows: int  # rows from stratified sampling
    tail_rows: int  # rows from tail-preserving sampling
    union_rows: int  # rows after union + dedup
    seed: int  # random seed
    total_users: Optional[int] = None
    total_items: Optional[int] = None


def hybrid_sample(
    df: pd.DataFrame,
    max_rows: int = 500_000,
    label_col: Optional[str] = "label_type",
    item_col: str = "item_id",
    user_col: str = "user_id",
    seed: int = 42,
    tail_quantile: float = 0.95,
) -> Tuple[pd.DataFrame, SampleMetadata]:
    """Hybrid tail-preserving stratified sampling.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    max_rows : int
        Maximum rows in the output sample (must be > 0).
    label_col : Optional[str]
        Column used for stratified sampling. If None or missing, falls back
        to uniform random sampling + tail protection.
    item_col : str
        Item ID column for tail detection.
    user_col : str
        User ID column for optional user count tracking.
    seed : int
        Random seed for reproducibility (must be >= 0).
    tail_quantile : float
        Quantile threshold for tail detection (0 < tail_quantile < 1).
        Items with frequency below this quantile are fully preserved.

    Returns
    -------
    Tuple[pd.DataFrame, SampleMetadata]
        Sampled DataFrame and sampling metadata.

    Raises
    ------
    ValueError
        If max_rows <= 0, tail_quantile not in (0, 1), or seed < 0.
    """
    # ---- parameter validation ----
    if max_rows <= 0:
        raise ValueError(f"max_rows must be > 0, got {max_rows}")
    if not (0 < tail_quantile < 1):
        raise ValueError(f"tail_quantile must be in (0, 1), got {tail_quantile}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    total_rows = len(df)

    # Compute total users / items early for metadata
    total_users: Optional[int] = None
    total_items: Optional[int] = None
    if user_col in df.columns:
        total_users = df[user_col].nunique()
    if item_col in df.columns:
        total_items = df[item_col].nunique()

    # If data is already small enough, return as-is
    if total_rows <= max_rows:
        metadata = SampleMetadata(
            sample_strategy="none",
            total_rows=total_rows,
            sample_ratio=1.0,
            strat_rows=total_rows,
            tail_rows=0,
            union_rows=total_rows,
            seed=seed,
            total_users=total_users,
            total_items=total_items,
        )
        logger.info(
            "Data size (%d) within max_rows (%d), skipping sampling.",
            total_rows,
            max_rows,
        )
        return df, metadata

    # ---- Step 1: Stratified sampling ----
    strat_fraction = 0.5  # allocate half budget to stratified
    strat_budget = int(max_rows * strat_fraction)

    has_label = label_col is not None and label_col in df.columns
    if has_label:
        # Stratified sample: proportional to label distribution
        label_groups = df.groupby(label_col, observed=False)
        strat_parts = []
        for _, group in label_groups:
            frac = min(1.0, strat_budget * len(group) / total_rows)
            n_sample = max(1, int(len(group) * frac))
            n_sample = min(n_sample, len(group))
            strat_parts.append(group.sample(n=n_sample, random_state=seed))
        df_strat = pd.concat(strat_parts, ignore_index=True)
        # Trim if overshot
        if len(df_strat) > strat_budget:
            df_strat = df_strat.sample(n=strat_budget, random_state=seed)
    else:
        logger.info(
            "Label column '%s' not found, falling back to uniform random "
            "for stratified portion.",
            label_col,
        )
        df_strat = df.sample(n=min(strat_budget, total_rows), random_state=seed)

    # ---- Step 2: Tail-preserving sampling ----
    tail_budget = max_rows - len(df_strat)

    if item_col in df.columns:
        item_counts = df[item_col].value_counts()
        tail_threshold = item_counts.quantile(1.0 - tail_quantile)
        tail_items = set(item_counts[item_counts <= tail_threshold].index)
        df_tail = df[df[item_col].isin(tail_items)]
        if len(df_tail) > tail_budget:
            df_tail = df_tail.sample(n=tail_budget, random_state=seed)
    else:
        logger.info(
            "Item column '%s' not found, tail preservation skipped.",
            item_col,
        )
        df_tail = pd.DataFrame(columns=df.columns)

    # ---- Step 3: Union and dedup ----
    if len(df_tail) > 0:
        df_union = pd.concat([df_strat, df_tail], ignore_index=True).drop_duplicates()
    else:
        df_union = df_strat

    # ---- Step 4: Final trim ----
    if len(df_union) > max_rows:
        df_union = df_union.sample(n=max_rows, random_state=seed)

    metadata = SampleMetadata(
        sample_strategy="stratified_tail_preserving",
        total_rows=total_rows,
        sample_ratio=len(df_union) / total_rows,
        strat_rows=len(df_strat),
        tail_rows=len(df_tail),
        union_rows=len(df_union),
        seed=seed,
        total_users=total_users,
        total_items=total_items,
    )

    logger.info(
        "Hybrid sampling: %d total → %d sampled (%.1f%%). "
        "strat=%d, tail=%d, strategy=%s",
        total_rows,
        len(df_union),
        100.0 * len(df_union) / total_rows,
        len(df_strat),
        len(df_tail),
        metadata.sample_strategy,
    )
    return df_union, metadata
