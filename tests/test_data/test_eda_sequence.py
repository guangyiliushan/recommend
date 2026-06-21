"""Sequence analysis tests."""

import numpy as np
import pandas as pd
import pytest

from recsys.data.eda.stats.sequence import _try_parse_sequence_value, analyze


class TestSequence:
    def test_with_domain_columns(self):
        df = pd.DataFrame(
            {
                "domain_a_seq": [
                    [1, 2, 3],
                    [4, 5],
                    [6, 7, 8, 9],
                ],
                "domain_b_seq": [
                    [10],
                    [11, 12],
                    [13, 14, 15],
                ],
            }
        )
        result = analyze(df, domain_pattern="domain_")
        assert not result.skipped
        assert result.has_sequences
        assert len(result.domain_lengths) == 2
        assert "domain_a_seq" in result.domain_lengths
        # Mean length of domain_a: (3+2+4)/3 = 3.0
        assert abs(result.domain_lengths["domain_a_seq"]["mean"] - 3.0) < 0.1

    def test_no_domain_columns(self):
        df = pd.DataFrame({"col_a": [1, 2, 3]})
        result = analyze(df, domain_pattern="domain_")
        assert result.skipped
        assert not result.has_sequences
        assert "domain_" in (result.skip_reason or "")

    def test_empty_dataframe(self):
        result = analyze(pd.DataFrame())
        assert result.skipped

    def test_seq_repeat_rate(self):
        df = pd.DataFrame(
            {
                "domain_a_seq": [
                    [1, 1, 2],   # 2 unique / 3 items → repeat = 1 - 2/3 = 0.333
                    [1, 1, 1, 1],  # repeat = 1 - 1/4 = 0.75
                ],
            }
        )
        result = analyze(df, domain_pattern="domain_")
        assert "domain_a_seq" in result.seq_repeat_rates
        # Average: (0.333 + 0.75) / 2 = 0.5417
        rate = result.seq_repeat_rates["domain_a_seq"]
        assert abs(rate - 0.5415) < 0.01

    def test_empty_sequences(self):
        df = pd.DataFrame(
            {
                "domain_a_seq": [[], [1, 2], []],
            }
        )
        result = analyze(df, domain_pattern="domain_")
        assert result.domain_lengths["domain_a_seq"]["empty_rate"] == pytest.approx(2 / 3, abs=0.01)


class TestParseSequenceValue:
    def test_list(self):
        assert _try_parse_sequence_value([1, 2, 3]) == [1, 2, 3]

    def test_numpy_array(self):
        result = _try_parse_sequence_value(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_string_representation(self):
        assert _try_parse_sequence_value("[1, 2, 3]") == [1, 2, 3]

    def test_none(self):
        assert _try_parse_sequence_value(None) is None

    def test_nan(self):
        assert _try_parse_sequence_value(float("nan")) is None
