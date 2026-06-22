"""Temporal analysis — time span, monthly volume, retention, interaction gaps.

Provides:
    - Global time span (min/max timestamp, covered days)
    - Monthly interaction volume trend
    - User retention curve (day-N active user ratio after first interaction)
    - Interaction gap distribution (time between consecutive user actions)
    - Daily interaction density

Works with DataFrames containing a timestamp column.
Gracefully skips when the column is missing or cannot be parsed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Column name candidates for timestamp detection (in priority order)
_TIMESTAMP_CANDIDATES = ["timestamp", "time", "created_at"]


def _detect_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    """Return the first matching timestamp column name, or None."""
    for cand in _TIMESTAMP_CANDIDATES:
        if cand in df.columns:
            return cand
    return None


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse a timestamp series, handling Unix seconds and ISO strings.

    Returns a DatetimeIndex-backed Series or None on failure.
    """
    sample = series.dropna().iloc[:5] if len(series) > 5 else series.dropna()
    if len(sample) == 0:
        return pd.Series(dtype="datetime64[ns]")

    # Heuristic: if all values are numeric and in Unix-second range (~1970-2090),
    # treat as Unix seconds.
    try:
        numeric = pd.to_numeric(sample, errors="coerce")
        if numeric.notna().all():
            mn, mx = numeric.min(), numeric.max()
            # Unix seconds range: ~0 (1970-01-01) to ~4e9 (2090+)
            if mn >= 0 and mx < 5e9:
                return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="s", errors="coerce")
    except Exception:
        pass

    # Fallback: try string parsing
    return pd.to_datetime(series, errors="coerce")


@dataclass
class TemporalResult:
    """Temporal behavior analysis results."""

    time_span: Dict[str, Any]  # {min_ts, max_ts, span_days}
    monthly_volume: Dict[str, int]  # {"2023-01": 1234, ...}
    retention_curve: Optional[Dict[int, float]]  # {day_1: 0.85, day_7: 0.42, ...}
    interaction_gap_stats: Optional[Dict[str, float]]  # {mean_hours, p50_hours, p95_hours, max_hours}
    daily_avg_interactions: float
    peak_day_interactions: int
    skipped: bool = False
    skip_reason: Optional[str] = None


def analyze(
    df: pd.DataFrame,
    user_col: str = "user_id",
    timestamp_col: Optional[str] = None,
    max_retention_days: int = 30,
) -> TemporalResult:
    """Analyze temporal patterns in user-item interactions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with user-item interactions.
    user_col : str
        Column name for user ID (needed for retention and gap analysis).
    timestamp_col : str, optional
        Column name for timestamps. Auto-detected if None.
    max_retention_days : int
        Maximum days to compute for retention curve (default 30).

    Returns
    -------
    TemporalResult
    """
    if df.empty:
        return TemporalResult(
            time_span={}, monthly_volume={},
            retention_curve=None, interaction_gap_stats=None,
            daily_avg_interactions=0.0, peak_day_interactions=0,
            skipped=True, skip_reason="DataFrame is empty.",
        )

    # Detect timestamp column
    ts_col = timestamp_col or _detect_timestamp_col(df)
    if ts_col is None or ts_col not in df.columns:
        return TemporalResult(
            time_span={}, monthly_volume={},
            retention_curve=None, interaction_gap_stats=None,
            daily_avg_interactions=0.0, peak_day_interactions=0,
            skipped=True,
            skip_reason=f"No timestamp column found (looked for: {', '.join(_TIMESTAMP_CANDIDATES)}).",
        )

    # Parse timestamps
    timestamps = _parse_timestamps(df[ts_col])
    valid_ts = timestamps.dropna()
    if len(valid_ts) == 0:
        return TemporalResult(
            time_span={}, monthly_volume={},
            retention_curve=None, interaction_gap_stats=None,
            daily_avg_interactions=0.0, peak_day_interactions=0,
            skipped=True, skip_reason="Could not parse timestamp values.",
        )

    # ---- Time span ----
    min_ts = valid_ts.min()
    max_ts = valid_ts.max()
    span_days = int((max_ts - min_ts).total_seconds() / 86400) if pd.notna(min_ts) and pd.notna(max_ts) else 0
    time_span = {
        "min_ts": str(min_ts) if pd.notna(min_ts) else "N/A",
        "max_ts": str(max_ts) if pd.notna(max_ts) else "N/A",
        "span_days": span_days,
    }

    # ---- Monthly volume ----
    monthly = valid_ts.dt.to_period("M").value_counts().sort_index()
    monthly_volume = {str(k): int(v) for k, v in monthly.items()}

    # ---- Daily average / peak ----
    daily = valid_ts.dt.date.value_counts()
    daily_avg = round(float(daily.mean()), 1) if len(daily) > 0 else 0.0
    peak_day = int(daily.max()) if len(daily) > 0 else 0

    # ---- Retention curve ----
    retention_curve: Optional[Dict[int, float]] = None
    has_user = user_col in df.columns
    if has_user and span_days > 0:
        try:
            df_with_ts = df[[user_col, ts_col]].copy()
            df_with_ts["_ts"] = _parse_timestamps(df_with_ts[ts_col])
            df_with_ts = df_with_ts.dropna(subset=["_ts"])

            if len(df_with_ts) > 0:
                # First interaction date per user
                first_dates = df_with_ts.groupby(user_col)["_ts"].min().dt.date
                all_users = set(first_dates.index)
                n_total = len(all_users)

                if n_total > 1:
                    retention_curve = {}
                    for day_offset in range(1, max_retention_days + 1):
                        # Users active on or after day_offset from their first date
                        active_set: set = set()
                        for uid in all_users:
                            first = first_dates[uid]
                            user_rows = df_with_ts[df_with_ts[user_col] == uid]
                            if len(user_rows) > 0:
                                max_date = user_rows["_ts"].dt.date.max()
                                if (max_date - first).days >= day_offset:
                                    active_set.add(uid)
                        ratio = round(len(active_set) / n_total, 4)
                        if ratio > 0:
                            retention_curve[day_offset] = ratio
        except Exception as e:
            logger.debug("Retention curve computation failed: %s", e)

    # ---- Interaction gap ----
    interaction_gap_stats: Optional[Dict[str, float]] = None
    if has_user:
        try:
            df_sorted = df[[user_col, ts_col]].copy()
            df_sorted["_ts"] = _parse_timestamps(df_sorted[ts_col])
            df_sorted = df_sorted.dropna(subset=["_ts"])
            if len(df_sorted) > 1:
                df_sorted = df_sorted.sort_values([user_col, "_ts"])
                gaps = df_sorted.groupby(user_col)["_ts"].diff().dropna()
                gaps_hours = gaps.dt.total_seconds() / 3600.0
                if len(gaps_hours) > 0:
                    interaction_gap_stats = {
                        "mean_hours": round(float(gaps_hours.mean()), 1),
                        "p50_hours": round(float(gaps_hours.quantile(0.5)), 1),
                        "p95_hours": round(float(gaps_hours.quantile(0.95)), 1),
                        "max_hours": round(float(gaps_hours.max()), 1),
                    }
        except Exception as e:
            logger.debug("Interaction gap computation failed: %s", e)

    logger.info(
        "Temporal: %d days span, %d months, daily_avg=%.1f, peak=%d, retention=%s",
        span_days, len(monthly_volume), daily_avg, peak_day,
        f"{len(retention_curve)} days" if retention_curve else "N/A",
    )

    return TemporalResult(
        time_span=time_span,
        monthly_volume=monthly_volume,
        retention_curve=retention_curve,
        interaction_gap_stats=interaction_gap_stats,
        daily_avg_interactions=daily_avg,
        peak_day_interactions=peak_day,
    )
