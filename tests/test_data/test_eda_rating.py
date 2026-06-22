"""Rating analysis tests."""

import pandas as pd
import pytest

from recsys.data.eda.stats.rating import analyze


@pytest.fixture
def df_rating() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
        "item_id": [10, 20, 30, 10, 40, 20, 30, 40, 50, 10],
        "rating": [5, 4, 3, 2, 4, 5, 4, 3, 5, 1],
    })


class TestRating:
    def test_rating_distribution(self, df_rating):
        result = analyze(df_rating)
        assert not result.skipped
        total = sum(result.rating_distribution.values())
        assert abs(total - 1.0) < 0.01

    def test_global_mean(self, df_rating):
        result = analyze(df_rating)
        assert 1.0 <= result.global_mean <= 5.0

    def test_user_bias(self, df_rating):
        result = analyze(df_rating)
        assert len(result.user_bias_top) > 0
        assert len(result.user_bias_bottom) > 0
        # Top should have non-negative biases, bottom should have non-positive
        if result.user_bias_top:
            assert result.user_bias_top[0][1] >= 0
        if result.user_bias_bottom:
            assert result.user_bias_bottom[0][1] <= 0

    def test_item_bias(self, df_rating):
        result = analyze(df_rating)
        assert len(result.item_bias_top) > 0
        assert len(result.item_bias_bottom) > 0

    def test_rating_density(self, df_rating):
        result = analyze(df_rating)
        assert "p50" in result.rating_density
        assert "p95" in result.rating_density

    def test_auto_detect_score_col(self):
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "item_id": [10, 20, 30],
            "score": [4.5, 3.0, 5.0],
        })
        result = analyze(df)
        assert not result.skipped

    def test_auto_detect_stars_col(self):
        df = pd.DataFrame({
            "user_id": [1, 2],
            "item_id": [10, 20],
            "stars": [3, 5],
        })
        result = analyze(df)
        assert not result.skipped

    def test_no_rating_col(self):
        df = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20]})
        result = analyze(df)
        assert result.skipped
        assert "rating" in result.skip_reason.lower()

    def test_empty_df(self):
        result = analyze(pd.DataFrame())
        assert result.skipped
