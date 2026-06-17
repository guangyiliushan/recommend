"""User/item analysis tests."""

import numpy as np
import pandas as pd
import pytest

from recsys.data.eda.stats.user_item import analyze


@pytest.fixture
def df_user_item() -> pd.DataFrame:
    """DataFrame with user_id, item_id, and sequence columns."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "user_id": rng.integers(0, 10, n),
            "item_id": rng.integers(0, 30, n),
            "domain_a_seq": [
                list(rng.integers(1, 50, rng.integers(1, 10)))
                for _ in range(n)
            ],
            "domain_b_seq": [
                list(rng.integers(1, 50, rng.integers(1, 10)))
                for _ in range(n)
            ],
        }
    )


class TestUserItem:
    def test_user_activity(self, df_user_item):
        result = analyze(df_user_item)
        assert "total_users" in result.user_activity
        assert "mean" in result.user_activity
        assert result.user_activity["total_users"] <= 10

    def test_item_popularity(self, df_user_item):
        result = analyze(df_user_item)
        assert "total_items" in result.item_popularity
        assert result.item_popularity["total_items"] <= 30

    def test_cross_domain_overlap(self, df_user_item):
        result = analyze(df_user_item, domain_pattern="domain_")
        assert result.cross_domain_overlap is not None
        # Should have at least one pair (domain_a_seq, domain_b_seq)
        assert len(result.cross_domain_overlap) > 0

    def test_no_domain_columns(self):
        df = pd.DataFrame({"user_id": [1, 2, 3], "item_id": [10, 20, 30]})
        result = analyze(df, domain_pattern="domain_")
        assert result.cross_domain_overlap is None

    def test_missing_user_col(self):
        df = pd.DataFrame({"item_id": [10, 20, 30]})
        result = analyze(df)
        assert "total_users" not in result.user_activity

    def test_empty_dataframe(self):
        result = analyze(pd.DataFrame())
        assert result.skipped
