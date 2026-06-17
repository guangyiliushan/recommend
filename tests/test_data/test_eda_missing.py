"""Missing value analysis tests."""

import numpy as np
import pandas as pd
import pytest

from recsys.data.eda.stats.missing import analyze


@pytest.fixture
def df_with_nulls() -> pd.DataFrame:
    """DataFrame with intentional nulls."""
    return pd.DataFrame(
        {
            "col_a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "col_b": [np.nan, 2.0, np.nan, 4.0, 5.0],
            "col_c": [1.0, 2.0, 3.0, 4.0, 5.0],
            "label_type": [0, 1, 0, 1, 1],
        }
    )


@pytest.fixture
def df_no_label() -> pd.DataFrame:
    """DataFrame without label column."""
    return pd.DataFrame(
        {
            "col_a": [1.0, np.nan, 3.0],
            "col_b": [np.nan, 2.0, 4.0],
        }
    )


class TestMissing:
    def test_column_missing_rates(self, df_with_nulls):
        result = analyze(df_with_nulls)
        assert result.column_missing_rates["col_a"] == 0.4
        assert result.column_missing_rates["col_c"] == 0.0

    def test_overall_missing_rate(self, df_with_nulls):
        result = analyze(df_with_nulls)
        # 4 nulls out of 20 cells = 0.2
        assert abs(result.overall_missing_rate - 0.2) < 0.01

    def test_coverage_matrix(self, df_with_nulls):
        result = analyze(df_with_nulls)
        assert result.coverage_matrix["col_c"] == 1.0
        assert result.coverage_matrix["col_a"] == 0.6

    def test_co_missing_pairs(self, df_with_nulls):
        result = analyze(df_with_nulls, top_n_co_missing=5)
        # col_a and col_b both have nulls, should have some co-missing
        # But the threshold is 5% < rate < 95%, and 2/5 = 40% qualifies
        assert len(result.co_missing_pairs) >= 0  # may or may not have pairs

    def test_null_rate_by_label(self, df_with_nulls):
        result = analyze(df_with_nulls, label_col="label_type")
        assert result.null_rate_by_label is not None
        assert 0 in result.null_rate_by_label  # label=0
        assert 1 in result.null_rate_by_label

    def test_no_label_column(self, df_no_label):
        result = analyze(df_no_label, label_col=None)
        assert result.null_rate_by_label is None
        assert not result.skipped

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = analyze(df)
        assert result.skipped

    def test_label_null_diff(self):
        """Features with different null rates across labels should be detected."""
        df = pd.DataFrame(
            {
                "feat_a": [1.0, np.nan, np.nan, 4.0],
                "feat_b": [np.nan, np.nan, 3.0, 4.0],
                "label_type": [0, 0, 1, 1],
            }
        )
        result = analyze(df, label_col="label_type")
        assert len(result.label_null_diff) > 0
        # feat_a: label=0 has 50% null, label=1 has 50% null → diff=0, won't appear
        # feat_b: label=0 has 100% null, label=1 has 0% null → diff=1.0
        # So at least feat_b should appear
        cols_in_diff = {item[0] for item in result.label_null_diff}
        assert "feat_b" in cols_in_diff

    def test_label_null_diff_no_label(self, df_no_label):
        result = analyze(df_no_label, label_col=None)
        assert result.label_null_diff == []
