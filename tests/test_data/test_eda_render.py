"""Renderer tests."""

import json
from pathlib import Path

from recsys.data.eda.render import render_to_echarts
from recsys.data.eda.sampler import SampleMetadata
from recsys.data.eda.stats.distribution import DistributionResult
from recsys.data.eda.stats.effectiveness import EffectivenessResult
from recsys.data.eda.stats.missing import MissingResult
from recsys.data.eda.stats.overview import OverviewResult
from recsys.data.eda.stats.sequence import SequenceResult
from recsys.data.eda.stats.user_item import UserItemResult


def _make_meta() -> SampleMetadata:
    return SampleMetadata(
        sample_strategy="stratified_tail_preserving",
        total_rows=1000,
        sample_ratio=0.5,
        strat_rows=250,
        tail_rows=250,
        union_rows=500,
        seed=42,
        total_users=100,
        total_items=200,
    )


def _make_overview() -> OverviewResult:
    return OverviewResult(
        total_rows=500,
        total_columns=10,
        column_groups={
            "core": ["user_id", "item_id", "label_type"],
            "user_feat": ["user_int_1", "user_int_2"],
            "item_feat": ["item_int_1"],
            "domain_seq": ["domain_a_seq"],
        },
        memory_usage_mb=12.5,
        has_label=True,
        has_timestamp=True,
        suspected_multimodal_embeddings=[],
    )


def _make_missing() -> MissingResult:
    return MissingResult(
        column_missing_rates={"col_a": 0.1, "col_b": 0.2},
        overall_missing_rate=0.15,
        co_missing_pairs=[("col_a", "col_b", 0.05)],
        null_rate_by_label={0: {"col_a": 0.05, "col_b": 1.0}, 1: {"col_a": 0.15, "col_b": 0.0}},
        coverage_matrix={"col_a": 0.9, "col_b": 0.8},
        label_null_diff=[("col_b", 0, 1, 1.0), ("col_a", 0, 1, 0.1)],
    )


def _make_distribution() -> DistributionResult:
    return DistributionResult(
        label_distribution={0: 0.5, 1: 0.5},
        feature_cardinality={"col_a": 100, "col_b": 50},
        cardinality_bins={"1-10": 0, "11-100": 1, "101-1K": 1},
        dense_stats={},
    )


def _make_sequence() -> SequenceResult:
    return SequenceResult(
        domain_lengths={
            "domain_a_seq": {"mean": 50.0, "p95": 100.0, "empty_rate": 0.1},
        },
        seq_repeat_rates={"domain_a_seq": 0.3},
        has_sequences=True,
    )


def _make_effectiveness() -> EffectivenessResult:
    return EffectivenessResult(
        feature_auc={"feat_a": 0.75, "feat_b": 0.55},
        skipped_features={"feat_c": "constant value"},
    )


def _make_user_item() -> UserItemResult:
    return UserItemResult(
        user_activity={"mean": 10.0, "p50": 5.0, "total_users": 100},
        item_popularity={"mean": 8.0, "total_items": 200},
        cross_domain_overlap={"domain_a_seq,domain_b_seq": 0.6},
    )


class TestRender:
    def test_chart_file_naming(self, tmp_path: Path):
        out = render_to_echarts(
            overview=_make_overview(),
            missing=_make_missing(),
            distribution=_make_distribution(),
            sequence=_make_sequence(),
            effectiveness=_make_effectiveness(),
            user_item=_make_user_item(),
            metadata=_make_meta(),
            output_dir=tmp_path,
        )
        assert out.chart_count > 0
        # Check expected filenames exist
        for name in out.chart_files:
            filepath = out.chart_files[name]
            assert filepath.exists()
            assert filepath.suffix == ".json"
            assert "echarts" in filepath.name

    def test_eda_metadata_in_charts(self, tmp_path: Path):
        out = render_to_echarts(
            overview=_make_overview(),
            missing=_make_missing(),
            distribution=_make_distribution(),
            sequence=_make_sequence(),
            effectiveness=_make_effectiveness(),
            user_item=_make_user_item(),
            metadata=_make_meta(),
            output_dir=tmp_path,
        )
        for chart_path in out.chart_files.values():
            data = json.loads(chart_path.read_text(encoding="utf-8"))
            assert "_eda_metadata" in data
            meta = data["_eda_metadata"]
            assert meta["sample_strategy"] == "stratified_tail_preserving"
            assert meta["total_rows"] == 1000

    def test_skipped_sequence_produces_no_chart(self, tmp_path: Path):
        seq = SequenceResult(
            domain_lengths={},
            seq_repeat_rates={},
            has_sequences=False,
            skipped=True,
            skip_reason="No domain columns.",
        )
        out = render_to_echarts(
            overview=_make_overview(),
            missing=_make_missing(),
            distribution=_make_distribution(),
            sequence=seq,
            effectiveness=_make_effectiveness(),
            user_item=_make_user_item(),
            metadata=_make_meta(),
            output_dir=tmp_path,
        )
        # sequence_lengths should NOT be in chart files
        assert "sequence_lengths" not in out.chart_files

    def test_label_null_diff_chart_exists(self, tmp_path: Path):
        """label_null_diff chart should be generated when data is present."""
        out = render_to_echarts(
            overview=_make_overview(),
            missing=_make_missing(),
            distribution=_make_distribution(),
            sequence=_make_sequence(),
            effectiveness=_make_effectiveness(),
            user_item=_make_user_item(),
            metadata=_make_meta(),
            output_dir=tmp_path,
        )
        assert "label_null_diff" in out.chart_files
        chart_path = out.chart_files["label_null_diff"]
        assert chart_path.exists()
