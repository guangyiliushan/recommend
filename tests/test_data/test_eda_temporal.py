"""Temporal analysis tests."""

import numpy as np
import pandas as pd
import pytest

from recsys.data.eda.stats.temporal import analyze


@pytest.fixture
def df_temporal() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    base_ts = 1700000000
    return pd.DataFrame({
        "user_id": rng.integers(0, 10, n),
        "item_id": rng.integers(0, 30, n),
        "timestamp": [base_ts + i * 86400 for i in range(n)],
    })


class TestTemporal:
    def test_time_span(self, df_temporal):
        result = analyze(df_temporal)
        assert not result.skipped
        assert result.time_span["span_days"] > 0
        assert result.time_span["min_ts"] != result.time_span["max_ts"]

    def test_monthly_volume(self, df_temporal):
        result = analyze(df_temporal)
        assert len(result.monthly_volume) > 0

    def test_daily_avg(self, df_temporal):
        result = analyze(df_temporal)
        assert result.daily_avg_interactions > 0

    def test_peak_day(self, df_temporal):
        result = analyze(df_temporal)
        assert result.peak_day_interactions > 0

    def test_retention_curve(self, df_temporal):
        """Retention curve may be empty for toy data but should not crash."""
        result = analyze(df_temporal, max_retention_days=7)
        assert not result.skipped

    def test_no_timestamp(self):
        df = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20]})
        result = analyze(df)
        assert result.skipped
        assert "timestamp" in result.skip_reason.lower()

    def test_auto_detect_time_col(self):
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "item_id": [10, 20, 30],
            "time": [1700000000, 1700086400, 1700172800],
        })
        result = analyze(df)
        assert not result.skipped

    def test_auto_detect_created_at_col(self):
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "item_id": [10, 20, 30],
            "created_at": [1700000000, 1700086400, 1700172800],
        })
        result = analyze(df)
        assert not result.skipped

    def test_explicit_timestamp_col_override(self):
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "item_id": [10, 20, 30],
            "event_time": [1700000000, 1700086400, 1700172800],
        })
        result = analyze(df, timestamp_col="event_time")
        assert not result.skipped
        assert result.time_span["span_days"] > 0

    def test_empty_df(self):
        result = analyze(pd.DataFrame())
        assert result.skipped
