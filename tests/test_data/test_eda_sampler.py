"""Sampler tests — hybrid tail-preserving stratified sampling."""

import numpy as np
import pandas as pd
import pytest

from recsys.data.eda.sampler import hybrid_sample


@pytest.fixture
def df_simple() -> pd.DataFrame:
    """Simple DataFrame with label, user, item."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "user_id": rng.integers(0, 20, n),
            "item_id": rng.integers(0, 50, n),
            "label_type": rng.integers(0, 2, n),
            "feat_a": rng.normal(0, 1, n),
        }
    )


@pytest.fixture
def df_long_tail() -> pd.DataFrame:
    """DataFrame with long-tail item distribution."""
    rng = np.random.default_rng(42)
    n = 500
    # Most items are popular, a few are rare
    item_ids = np.array([0] * 200 + [1] * 150 + [2] * 100 + list(range(3, 53)))
    rng.shuffle(item_ids)
    return pd.DataFrame(
        {
            "user_id": rng.integers(0, 30, n),
            "item_id": item_ids[:n],
            "label_type": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def df_no_label() -> pd.DataFrame:
    """DataFrame without label column."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "user_id": rng.integers(0, 10, n),
            "item_id": rng.integers(0, 30, n),
        }
    )


class TestHybridSample:
    def test_small_data_no_sampling(self, df_simple):
        """Data below max_rows should not be sampled."""
        df, meta = hybrid_sample(df_simple, max_rows=1000, seed=42)
        assert len(df) == len(df_simple)
        assert meta.sample_strategy == "none"

    def test_stratified_proportions(self, df_simple):
        """Label proportions should be approximately preserved."""
        df, meta = hybrid_sample(df_simple, max_rows=50, seed=42)
        original_props = df_simple["label_type"].value_counts(normalize=True)
        sampled_props = df["label_type"].value_counts(normalize=True)
        for label_val in original_props.index:
            assert abs(original_props[label_val] - sampled_props.get(label_val, 0)) < 0.3

    def test_idempotent_same_seed(self, df_simple):
        """Same seed should produce identical samples."""
        df1, _ = hybrid_sample(df_simple, max_rows=50, seed=42)
        df2, _ = hybrid_sample(df_simple, max_rows=50, seed=42)
        pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))

    def test_different_seeds_different(self, df_simple):
        """Different seeds should produce different samples."""
        df1, _ = hybrid_sample(df_simple, max_rows=50, seed=42)
        df2, _ = hybrid_sample(df_simple, max_rows=50, seed=99)
        assert not df1.reset_index(drop=True).equals(df2.reset_index(drop=True))

    def test_tail_items_preserved(self, df_long_tail):
        """Rare items should be preserved in the tail portion."""
        item_counts = df_long_tail["item_id"].value_counts()
        rare_items = set(item_counts[item_counts <= 2].index)
        df, meta = hybrid_sample(df_long_tail, max_rows=200, seed=42, tail_quantile=0.90)
        sampled_items = set(df["item_id"].unique())
        # Check that at least some rare items survived
        preserved = rare_items & sampled_items
        assert len(preserved) > 0, "Expected some rare items to be preserved, got none."

    def test_no_label_fallback(self, df_no_label):
        """Without label column, should fall back gracefully and use random + tail."""
        df, meta = hybrid_sample(
            df_no_label, max_rows=30, label_col=None, seed=42
        )
        assert meta.sample_strategy != "none"
        assert len(df) <= 30

    def test_invalid_max_rows(self, df_simple):
        """max_rows <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="max_rows"):
            hybrid_sample(df_simple, max_rows=0)

    def test_invalid_tail_quantile(self, df_simple):
        """tail_quantile out of range should raise ValueError."""
        with pytest.raises(ValueError, match="tail_quantile"):
            hybrid_sample(df_simple, max_rows=100, tail_quantile=1.5)

    def test_invalid_seed(self, df_simple):
        """Negative seed should raise ValueError."""
        with pytest.raises(ValueError, match="seed"):
            hybrid_sample(df_simple, max_rows=100, seed=-1)

    def test_metadata_fields(self, df_simple):
        """Metadata should contain all expected fields."""
        _, meta = hybrid_sample(df_simple, max_rows=50, seed=42)
        assert meta.sample_strategy == "stratified_tail_preserving"
        assert meta.total_rows == len(df_simple)
        assert 0 < meta.sample_ratio <= 1.0
        assert meta.seed == 42
        assert meta.total_users is not None
        assert meta.total_items is not None
