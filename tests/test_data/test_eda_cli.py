"""CLI integration tests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from recsys.data.eda import EDAConfig
from recsys.data.eda.cli import main, run_eda


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a small sample CSV for testing."""
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "item_id": [10, 20, 30, 40, 50],
            "label_type": [0, 1, 1, 0, 1],
            "feat_a": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


class TestRunEda:
    def test_json_only_mode(self, sample_csv: Path, tmp_path: Path):
        df = pd.read_csv(sample_csv)
        json_path = str(tmp_path / "stats.json")
        config = EDAConfig(
            max_rows=100,
            output_dir=str(tmp_path / "charts"),
            report_path=str(tmp_path / "report.md"),
            json_path=json_path,
            json_only=True,
        )
        result = run_eda(config, df)
        assert result["status"] == "ok"
        assert result["mode"] == "json_only"
        assert Path(json_path).exists()

        # Verify JSON structure
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        assert "overview" in data
        assert "missing" in data
        assert "distribution" in data
        assert "sample_metadata" in data

    def test_full_mode(self, sample_csv: Path, tmp_path: Path):
        df = pd.read_csv(sample_csv)
        chart_dir = tmp_path / "charts"
        report_path = tmp_path / "report.md"
        config = EDAConfig(
            max_rows=100,
            output_dir=str(chart_dir),
            report_path=str(report_path),
            json_only=False,
        )
        result = run_eda(config, df)
        assert result["status"] == "ok"
        assert result["mode"] == "full"
        assert result["chart_count"] > 0
        assert report_path.exists()

    def test_empty_dataframe(self):
        config = EDAConfig()
        result = run_eda(config, pd.DataFrame())
        assert result["status"] == "error"

    def test_integration_new_modules(self, sample_csv: Path, tmp_path: Path):
        """Verify sparsity/temporal/rating modules are present in output."""
        df = pd.read_csv(sample_csv)
        json_path = str(tmp_path / "stats.json")
        config = EDAConfig(
            max_rows=100,
            json_path=json_path,
            json_only=True,
        )
        result = run_eda(config, df)
        stats = result["stats"]
        assert "sparsity" in stats
        assert "temporal" in stats
        assert "rating" in stats
        # Sparsity should NOT be skipped (has user_id + item_id)
        assert not stats["sparsity"]["skipped"]
        # Temporal skipped (no timestamp column)
        assert stats["temporal"]["skipped"]
        # Rating skipped (no rating column)
        assert stats["rating"]["skipped"]


class TestCli:
    def test_backward_compatible_passes_rating_and_timestamp_cols(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Test that --rating-col and --timestamp-col are passed through."""
        from recsys.core import registry as registry_module

        class MockDataset:
            def _load_raw(self):
                df = pd.DataFrame(
                    {
                        "user_id": [1, 1, 2, 2],
                        "item_id": [10, 20, 10, 30],
                        "label_type": [1, 0, 1, 1],
                        "event_time": [
                            1700000000,
                            1700086400,
                            1700172800,
                            1700259200,
                        ],
                        "my_rating": [4.0, 5.0, 3.0, 4.5],
                    }
                )
                # _load_raw should return a dict with 'dataset' key
                return {"dataset": df}

        monkeypatch.setattr(
            registry_module.DATASET_REGISTRY,
            "get",
            lambda name: MockDataset if name == "mock_dataset" else None,
        )

        json_path = tmp_path / "mock-summary.json"
        exit_code = main(
            [
                "--dataset",
                "mock_dataset",
                "--json-only",
                "--json-path",
                str(json_path),
                "--output-dir",
                str(tmp_path / "charts"),
                "--report-path",
                str(tmp_path / "report.md"),
                "--timestamp-col",
                "event_time",
                "--rating-col",
                "my_rating",
            ]
        )

        assert exit_code == 0

        summary = json.loads(json_path.read_text(encoding="utf-8"))
        assert not summary["temporal"]["skipped"]
        assert not summary["rating"]["skipped"]

    def test_json_only_flag(self, sample_csv: Path, tmp_path: Path):
        json_path = tmp_path / "out.json"
        argv = [
            "--dataset-path", str(sample_csv),
            "--json-only",
            "--json-path", str(json_path),
        ]
        exit_code = main(argv)
        assert exit_code == 0
        assert json_path.exists()

    def test_missing_input(self):
        """No --dataset or --dataset-path should return error."""
        exit_code = main([])
        assert exit_code == 1

    def test_nonexistent_file(self):
        argv = ["--dataset-path", "nonexistent.parquet"]
        exit_code = main(argv)
        assert exit_code == 1
