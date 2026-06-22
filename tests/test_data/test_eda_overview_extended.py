"""Overview extended type detection tests."""

import pandas as pd

from recsys.data.eda.stats.overview import analyze


class TestOverviewExtended:
    def test_implicit_feedback(self):
        df = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20]})
        result = analyze(df)
        assert result.feedback_type == "implicit"

    def test_implicit_binary(self):
        df = pd.DataFrame({
            "user_id": [1, 2], "item_id": [10, 20],
            "label_type": [0, 1],
        })
        result = analyze(df)
        assert result.feedback_type == "implicit_binary"

    def test_explicit_feedback_via_rating(self):
        df = pd.DataFrame({
            "user_id": [1, 2], "item_id": [10, 20],
            "rating": [4, 5],
        })
        result = analyze(df)
        assert result.feedback_type == "explicit"

    def test_domain_seq_type(self):
        df = pd.DataFrame({
            "user_id": [1, 2], "item_id": [10, 20],
            "domain_a_seq": [[1, 2, 3], [4, 5]],
        })
        result = analyze(df)
        assert result.sequence_type == "domain_seq"

    def test_nested_seq_type(self):
        df = pd.DataFrame({
            "user_id": [1, 2], "item_id": [10, 20],
            "seq": [[1, 2], [3, 4, 5]],
        })
        result = analyze(df)
        assert result.sequence_type == "nested_seq"

    def test_no_sequences(self):
        df = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20]})
        result = analyze(df)
        assert result.sequence_type == "none"

    def test_modality_single(self):
        df = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20]})
        result = analyze(df)
        assert result.modality == "single"

    def test_modality_multimodal(self):
        df = pd.DataFrame({
            "user_id": [1, 2], "item_id": [10, 20],
            "mm_emb_text": [[0.1, 0.2], [0.3, 0.4]],
        })
        result2 = analyze(df)
        assert result2.modality == "multimodal"
