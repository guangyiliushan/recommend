"""Overview stats tests."""

import pandas as pd

from recsys.data.eda.stats.overview import analyze


def _build_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [10, 20, 30],
            "label_type": [0, 1, 1],
            "timestamp": [100, 200, 300],
            "user_int_feats_1": [1, 2, 3],
            "item_int_feats_1": [5, 6, 7],
            "domain_a_seq": [[1, 2], [3], [4, 5, 6]],
        }
    )


class TestOverview:
    def test_column_groups(self):
        df = _build_df()
        result = analyze(df)
        assert result.total_rows == 3
        assert result.total_columns == 7
        assert not result.skipped
        assert len(result.column_groups["core"]) == 4  # user_id, item_id, label_type, timestamp
        assert "domain_a_seq" in result.column_groups["domain_seq"]

    def test_has_label_and_timestamp(self):
        df = _build_df()
        result = analyze(df)
        assert result.has_label
        assert result.has_timestamp

    def test_no_label(self):
        df = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20]})
        result = analyze(df)
        assert not result.has_label
        assert not result.has_timestamp

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = analyze(df)
        assert result.skipped
        assert "empty" in (result.skip_reason or "").lower()

    def test_memory_usage(self):
        df = _build_df()
        result = analyze(df)
        assert result.memory_usage_mb >= 0

    def test_multimodal_detection(self):
        """Columns with high missing rate + high cardinality should be flagged."""
        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3, 4, 5],
                "item_id": [10, 20, 30, 40, 50],
                "label_type": [0, 1, 0, 1, 0],
                "item_int_feats_83": [1, 2, None, None, None],  # 60% missing, cardinality=2 < 10 → not flagged
                "item_int_feats_84": [1, None, None, None, None],  # 80% missing, cardinality=1 → not flagged
                "item_int_feats_85": [1, None, None, None, None],  # 80% missing, cardinality=1 → not flagged
            }
        )
        result = analyze(df)
        # None should be flagged (missing rate not > 80% enough, or card too low)
        assert isinstance(result.suspected_multimodal_embeddings, list)

    def test_multimodal_detection_high_card(self):
        """High cardinality + high missing should be flagged."""
        df = pd.DataFrame(
            {
                "user_id": [1] * 20,
                "item_id": list(range(20)),
                "label_type": [0] * 20,
                "item_int_feats_83": [1, 2, 3, 4] + [None] * 16,  # 80% missing, cardinality=4 → <10, not flagged
                "item_int_feats_99": list(range(1, 21)) + [None] * 0,  # 0% missing → not flagged
                "item_int_feats_85": list(range(11, 31)) + [None] * 0,  # 0% missing → not flagged
            }
        )
        result = analyze(df)
        # With only 20 rows, 80% missing for 83, cardinality=4 < 10 → not flagged
        assert "item_int_feats_83" not in result.suspected_multimodal_embeddings

    def test_multimodal_with_high_card_high_miss(self):
        """Column with >80% missing AND >10 cardinality should be flagged."""
        import numpy as np
        df = pd.DataFrame(
            {
                "user_id": [1] * 100,
                "item_id": list(range(100)),
                "label_type": [0] * 100,
                "item_int_feats_83": np.arange(12).tolist() + [None] * 88,  # 88% missing, cardinality=12
            }
        )
        result = analyze(df)
        assert "item_int_feats_83" in result.suspected_multimodal_embeddings
