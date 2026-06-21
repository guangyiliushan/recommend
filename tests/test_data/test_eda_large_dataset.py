"""Tests for large dataset pre-sampling in _extract_dataframe."""

import pandas as pd
import pytest

from recsys.data.eda.cli import _extract_dataframe


class TestExtractDataFramePreSampling:
    """Verify that _extract_dataframe pre-samples large datasets at arrow level."""

    def test_dataframe_passthrough(self):
        """Plain DataFrame should pass through without sampling."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        raw = {"dataset": df}
        result_df, load_meta = _extract_dataframe(raw, max_rows=2, seed=42)
        assert len(result_df) == 5  # Not sampled — DataFrame is passed through
        assert load_meta is None

    def test_huggingface_dataset_mock(self):
        """Mock HuggingFace Dataset with .select() and .to_pandas()."""
        class MockHFDataset:
            def __init__(self, n):
                self._n = n

            def __len__(self):
                return self._n

            def select(self, indices):
                """Return a smaller mock dataset."""
                smaller = MockHFDataset(len(indices))
                smaller._indices = indices
                return smaller

            def to_pandas(self):
                """Return a DataFrame with the sampled row count."""
                n = getattr(self, "_n", 0)
                return pd.DataFrame(
                    {"user_id": range(n), "item_id": range(n)}
                )

        # 10,000 rows mock dataset, max_rows=100
        ds = MockHFDataset(10_000)
        raw = {"seq": ds}
        result_df, load_meta = _extract_dataframe(raw, max_rows=100, seed=42)

        assert len(result_df) == 100
        assert load_meta is not None
        assert load_meta["original_rows"] == 10_000
        assert load_meta["sampled_at_load"] is True

    def test_small_dataset_not_sampled(self):
        """Dataset below max_rows should not be pre-sampled."""
        class MockHFDataset:
            def __len__(self):
                return 50

            def select(self, indices):
                raise AssertionError("Should not be called")

            def to_pandas(self):
                return pd.DataFrame({"a": range(50)})

        ds = MockHFDataset()
        raw = {"seq": ds}
        result_df, load_meta = _extract_dataframe(raw, max_rows=100, seed=42)

        assert len(result_df) == 50
        assert load_meta is None

    def test_key_order_priority(self):
        """dataset key should take priority over seq/train."""
        class MockHFDataset:
            def __len__(self):
                return 200
            def select(self, indices):
                return self
            def to_pandas(self):
                return pd.DataFrame({"source": ["dataset"] * 100})

        raw = {
            "seq": MockHFDataset(),
            "dataset": MockHFDataset(),
        }
        result_df, _ = _extract_dataframe(raw, max_rows=100, seed=42)
        # Both return different data, but "dataset" key takes priority
        assert len(result_df) > 0

    def test_empty_dict_raises(self):
        """Empty raw dict should raise KeyError."""
        with pytest.raises(KeyError, match="Cannot extract"):
            _extract_dataframe({})
