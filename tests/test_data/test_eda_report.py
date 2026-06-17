"""Report generation tests."""

from pathlib import Path

from recsys.data.eda.report import generate_markdown_report
from recsys.data.eda.sampler import SampleMetadata
from recsys.data.eda.stats.distribution import DistributionResult
from recsys.data.eda.stats.effectiveness import EffectivenessResult
from recsys.data.eda.stats.missing import MissingResult
from recsys.data.eda.stats.overview import OverviewResult
from recsys.data.eda.stats.sequence import SequenceResult
from recsys.data.eda.stats.user_item import UserItemResult


class TestReport:
    def test_report_contains_expected_sections(self, tmp_path: Path):
        meta = SampleMetadata(
            sample_strategy="stratified_tail_preserving",
            total_rows=1000, sample_ratio=0.5,
            strat_rows=250, tail_rows=250, union_rows=500,
            seed=42, total_users=100, total_items=200,
        )
        overview = OverviewResult(
            total_rows=500, total_columns=10,
            column_groups={"core": ["user_id"], "user_feat": [], "item_feat": [], "domain_seq": []},
            memory_usage_mb=5.0, has_label=True, has_timestamp=True,
            suspected_multimodal_embeddings=[],
        )
        missing = MissingResult(
            column_missing_rates={"col_a": 0.1}, overall_missing_rate=0.1,
            co_missing_pairs=[], null_rate_by_label=None,
            coverage_matrix={"col_a": 0.9},
            label_null_diff=[],
        )
        distribution = DistributionResult(
            label_distribution={0: 0.5, 1: 0.5},
            feature_cardinality={"col_a": 10},
            cardinality_bins={"1-10": 1},
            dense_stats={},
        )
        sequence = SequenceResult(
            domain_lengths={}, seq_repeat_rates={},
            has_sequences=False, skipped=True,
            skip_reason="No domain columns.",
        )
        effectiveness = EffectivenessResult(
            feature_auc={}, skipped_features={}, skipped=True,
            skip_reason="No label column.",
        )
        user_item = UserItemResult(
            user_activity={"mean": 10.0, "total_users": 100},
            item_popularity={"mean": 5.0, "total_items": 200},
            cross_domain_overlap=None,
        )

        output_path = tmp_path / "dataset-eda.md"
        chart_files = {}  # no actual charts needed for report test

        path = generate_markdown_report(
            overview=overview,
            missing=missing,
            distribution=distribution,
            sequence=sequence,
            effectiveness=effectiveness,
            user_item=user_item,
            metadata=meta,
            chart_files=chart_files,
            output_path=output_path,
        )
        assert path.exists()

        content = path.read_text(encoding="utf-8")
        # Check key sections exist
        assert "数据集 EDA 报告" in content
        assert "数据集概况" in content
        assert "列布局概览" in content
        assert "行为类型分布" in content
        assert "特征缺失率" in content
        assert "采样说明" in content

    def test_skipped_sections_show_reason(self, tmp_path: Path):
        meta = SampleMetadata(
            sample_strategy="none", total_rows=10, sample_ratio=1.0,
            strat_rows=10, tail_rows=0, union_rows=10, seed=42,
        )
        overview = OverviewResult(
            total_rows=10, total_columns=3,
            column_groups={"core": ["user_id", "item_id"], "user_feat": [], "item_feat": ["f1"], "domain_seq": []},
            memory_usage_mb=0.1, has_label=False, has_timestamp=False,
            suspected_multimodal_embeddings=[],
        )
        missing = MissingResult(
            column_missing_rates={}, overall_missing_rate=0.0,
            co_missing_pairs=[], null_rate_by_label=None,
            coverage_matrix={},
            label_null_diff=[],
        )
        distribution = DistributionResult(
            label_distribution={}, feature_cardinality={},
            cardinality_bins={}, dense_stats={},
        )
        sequence = SequenceResult(
            domain_lengths={}, seq_repeat_rates={},
            has_sequences=False, skipped=True,
            skip_reason="No domain columns.",
        )
        effectiveness = EffectivenessResult(
            feature_auc={}, skipped_features={},
            skipped=True, skip_reason="No label column.",
        )
        user_item = UserItemResult(
            user_activity={}, item_popularity={},
            cross_domain_overlap=None,
        )

        output_path = tmp_path / "report.md"
        path = generate_markdown_report(
            overview=overview, missing=missing, distribution=distribution,
            sequence=sequence, effectiveness=effectiveness,
            user_item=user_item, metadata=meta,
            chart_files={}, output_path=output_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "分析跳过" in content
        assert "No domain columns." in content
        assert "No label column." in content

    def test_multimodal_section(self, tmp_path: Path):
        """When suspected_multimodal_embeddings is non-empty, the section should appear."""
        meta = SampleMetadata(
            sample_strategy="none", total_rows=10, sample_ratio=1.0,
            strat_rows=10, tail_rows=0, union_rows=10, seed=42,
        )
        overview = OverviewResult(
            total_rows=10, total_columns=5,
            column_groups={"core": ["user_id"], "user_feat": [], "item_feat": ["item_int_feats_83", "item_int_feats_84", "item_int_feats_85"], "domain_seq": []},
            memory_usage_mb=0.1, has_label=True, has_timestamp=False,
            suspected_multimodal_embeddings=["item_int_feats_83", "item_int_feats_84", "item_int_feats_85"],
        )
        missing = MissingResult(
            column_missing_rates={"item_int_feats_83": 0.83, "item_int_feats_84": 0.83, "item_int_feats_85": 0.83},
            overall_missing_rate=0.1, co_missing_pairs=[],
            null_rate_by_label=None, coverage_matrix={}, label_null_diff=[],
        )
        distribution = DistributionResult(
            label_distribution={0: 0.5, 1: 0.5},
            feature_cardinality={"item_int_feats_83": 500, "item_int_feats_84": 300, "item_int_feats_85": 200},
            cardinality_bins={}, dense_stats={},
        )
        output_path = tmp_path / "report.md"
        path = generate_markdown_report(
            overview=overview, missing=missing, distribution=distribution,
            sequence=SequenceResult(domain_lengths={}, seq_repeat_rates={}, has_sequences=False, skipped=True, skip_reason="No domain columns."),
            effectiveness=EffectivenessResult(feature_auc={}, skipped_features={}, skipped=True, skip_reason="No label column."),
            user_item=UserItemResult(user_activity={}, item_popularity={}, cross_domain_overlap=None),
            metadata=meta, chart_files={}, output_path=output_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "多模态嵌入分析" in content
        assert "item_int_feats_83" in content
        assert "item_int_feats_84" in content
        assert "item_int_feats_85" in content

    def test_label_null_diff_section(self, tmp_path: Path):
        """When label_null_diff is non-empty, the section should appear."""
        meta = SampleMetadata(
            sample_strategy="none", total_rows=10, sample_ratio=1.0,
            strat_rows=10, tail_rows=0, union_rows=10, seed=42,
        )
        overview = OverviewResult(
            total_rows=10, total_columns=3,
            column_groups={"core": ["user_id"], "user_feat": [], "item_feat": [], "domain_seq": []},
            memory_usage_mb=0.1, has_label=True, has_timestamp=False,
            suspected_multimodal_embeddings=[],
        )
        missing = MissingResult(
            column_missing_rates={"feat_a": 0.3}, overall_missing_rate=0.1,
            co_missing_pairs=[],
            null_rate_by_label={0: {"feat_a": 0.1}, 1: {"feat_a": 0.9}},
            coverage_matrix={},
            label_null_diff=[("feat_a", 0, 1, 0.8)],
        )
        output_path = tmp_path / "report.md"
        path = generate_markdown_report(
            overview=overview, missing=missing,
            distribution=DistributionResult(label_distribution={}, feature_cardinality={}, cardinality_bins={}, dense_stats={}),
            sequence=SequenceResult(domain_lengths={}, seq_repeat_rates={}, has_sequences=False, skipped=True, skip_reason="No domain columns."),
            effectiveness=EffectivenessResult(feature_auc={}, skipped_features={}, skipped=True, skip_reason="No label column."),
            user_item=UserItemResult(user_activity={}, item_popularity={}, cross_domain_overlap=None),
            metadata=meta, chart_files={}, output_path=output_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "正负样本缺失率对比" in content
        assert "feat_a" in content
        assert "MNAR" in content
