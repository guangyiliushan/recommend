"""Sparsity analysis tests."""

import numpy as np
import pandas as pd
import pytest

from recsys.data.eda.stats.sparsity import analyze


@pytest.fixture
def df_interactions() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "user_id": rng.integers(0, 20, n),
        "item_id": rng.integers(0, 50, n),
    })


class TestSparsity:
    def test_basic_sparsity(self, df_interactions):
        result = analyze(df_interactions)
        assert not result.skipped
        assert 0 <= result.matrix_sparsity <= 1
        assert result.user_interaction_stats["total_users"] == 20
        assert result.item_popularity_stats["total_items"] <= 50

    def test_gini_range(self, df_interactions):
        result = analyze(df_interactions)
        assert 0 <= result.item_gini <= 1

    def test_cold_start_ratios(self, df_interactions):
        result = analyze(df_interactions, cold_start_quantile=0.10)
        assert 0 <= result.cold_start_user_ratio <= 1
        assert 0 <= result.cold_start_item_ratio <= 1

    def test_concentration(self, df_interactions):
        result = analyze(df_interactions)
        conc = result.user_concentration
        assert 0 <= conc["top1pct"] <= conc["top5pct"] <= conc["top10pct"] <= 1

    def test_long_tail_coverage(self, df_interactions):
        result = analyze(df_interactions)
        assert 0 <= result.long_tail_coverage <= 1

    def test_popularity_sorted(self, df_interactions):
        result = analyze(df_interactions)
        assert result.item_popularity_sorted is not None
        assert len(result.item_popularity_sorted) > 0

    def test_no_user_col(self):
        df = pd.DataFrame({"item_id": [1, 2, 3, 4, 5]})
        result = analyze(df)
        assert result.skipped
        assert "user_id" in result.skip_reason

    def test_no_item_col(self):
        df = pd.DataFrame({"user_id": [1, 2, 3, 4, 5]})
        result = analyze(df)
        assert result.skipped
        assert "item_id" in result.skip_reason

    def test_empty_df(self):
        result = analyze(pd.DataFrame())
        assert result.skipped
        assert "empty" in result.skip_reason.lower()
