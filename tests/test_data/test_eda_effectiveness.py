"""Feature effectiveness tests."""

import numpy as np
import pandas as pd
import pytest

from recsys.data.eda.stats.effectiveness import analyze


@pytest.fixture
def df_binary() -> pd.DataFrame:
    """DataFrame with binary label and numeric features."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "label_type": [0] * 50 + [1] * 50,
            "feat_good": np.concatenate([rng.normal(0, 1, 50), rng.normal(2, 1, 50)]),
            "feat_noise": rng.normal(0, 1, n),
            "feat_const": [5.0] * n,
        }
    )


class TestEffectiveness:
    def test_binary_auc(self, df_binary):
        result = analyze(df_binary)
        assert not result.skipped
        # Good feature should have AUC > noise
        assert result.feature_auc["feat_good"] > result.feature_auc["feat_noise"]

    def test_auc_range(self, df_binary):
        result = analyze(df_binary)
        for auc in result.feature_auc.values():
            assert 0.0 <= auc <= 1.0

    def test_const_feature_skipped(self, df_binary):
        result = analyze(df_binary)
        assert "feat_const" in result.skipped_features
        assert "constant" in result.skipped_features["feat_const"].lower()

    def test_no_label_column(self):
        df = pd.DataFrame({"feat_a": [1, 2, 3], "feat_b": [4, 5, 6]})
        result = analyze(df, label_col="label_type")
        assert result.skipped
        assert "label" in (result.skip_reason or "").lower()

    def test_single_class(self):
        df = pd.DataFrame({"label_type": [1, 1, 1], "feat_a": [1, 2, 3]})
        result = analyze(df)
        assert result.skipped
        assert "class" in (result.skip_reason or "").lower()

    def test_empty_dataframe(self):
        result = analyze(pd.DataFrame())
        assert result.skipped
