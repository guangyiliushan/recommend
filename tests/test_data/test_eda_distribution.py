"""Distribution analysis tests."""

import numpy as np
import pandas as pd
import pytest

from recsys.data.eda.stats.distribution import analyze


@pytest.fixture
def df_with_dense() -> pd.DataFrame:
    """DataFrame with label, features, and dense columns."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "label_type": rng.integers(0, 2, n),
            "user_int_feats_1": rng.integers(0, 100, n),
            "user_int_feats_2": rng.integers(0, 10, n),
            "user_dense_feats_1": rng.normal(0, 1, n),
            "user_dense_feats_2": rng.normal(5, 2, n),
        }
    )


class TestDistribution:
    def test_label_distribution(self, df_with_dense):
        result = analyze(df_with_dense)
        total = sum(result.label_distribution.values())
        assert abs(total - 1.0) < 0.01

    def test_feature_cardinality(self, df_with_dense):
        result = analyze(df_with_dense)
        assert "user_int_feats_1" in result.feature_cardinality
        assert result.feature_cardinality["user_int_feats_1"] <= 100

    def test_cardinality_bins(self, df_with_dense):
        result = analyze(df_with_dense)
        total_in_bins = sum(result.cardinality_bins.values())
        # Should account for all columns (minus label)
        assert total_in_bins <= len(df_with_dense.columns)

    def test_dense_stats(self, df_with_dense):
        result = analyze(df_with_dense, dense_pattern="user_dense_")
        assert len(result.dense_stats) == 2
        for col in ["user_dense_feats_1", "user_dense_feats_2"]:
            stats = result.dense_stats[col]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert "skew" in stats
            assert "zeros_ratio" in stats

    def test_no_dense_cols(self):
        df = pd.DataFrame({"label_type": [0, 1], "x": [1, 2]})
        result = analyze(df, dense_pattern="user_dense_")
        assert result.dense_stats == {}

    def test_empty_dataframe(self):
        result = analyze(pd.DataFrame())
        assert result.skipped
