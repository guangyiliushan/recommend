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


class TestCli:
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
