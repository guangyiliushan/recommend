"""Tests for TAAC2025 EDA subset loading behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from recsys.data.datasets import taac2025 as taac2025_module


def _build_dataset(version: str, root_dir: Path):
    ds = taac2025_module.TAAC2025Dataset.__new__(taac2025_module.TAAC2025Dataset)
    ds._version = version
    ds._repo_id = f"TAAC2025/TencentGR-{version}"
    ds.root_dir = str(root_dir)
    return ds


def test_list_cached_dataset_configs_filters_incomplete_configs(tmp_path: Path):
    base = tmp_path / "TAAC2025___tencent_gr-1_m"
    complete = base / "mm_emb_81_32" / "0.0.0" / "hash123"
    complete.mkdir(parents=True)
    (complete / "dataset_info.json").write_text("{}", encoding="utf-8")

    incomplete = base / "mm_emb_82_1024" / "0.0.0"
    incomplete.mkdir(parents=True)
    (incomplete / "hash123_builder.lock").write_text("", encoding="utf-8")

    cached = taac2025_module._list_cached_dataset_configs(str(tmp_path), "1M")
    assert cached == {"mm_emb_81_32"}


def test_build_manifest_uses_offline_cached_configs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    base = tmp_path / "TAAC2025___tencent_gr-1_m"
    for name in ("seq", "user_feat", "mm_emb_81_32"):
        config_dir = base / name / "0.0.0" / "hash123"
        config_dir.mkdir(parents=True)
        (config_dir / "dataset_info.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        taac2025_module,
        "_list_dataset_configs",
        lambda repo_id: [
            "seq",
            "user_feat",
            "candidate",
            "mm_emb_81_32",
            "mm_emb_83_3584",
        ],
    )
    monkeypatch.setattr(
        taac2025_module, "_estimate_subset_rows", lambda repo_id, config, cache_dir: 0
    )

    ds = _build_dataset("1M", tmp_path)
    manifest = ds._build_manifest()

    assert manifest.list_subsets() == ["mm_emb_81_32", "seq", "user_feat"]


def test_estimate_subset_rows_uses_cached_dataset_info(tmp_path: Path):
    dataset_info = (
        tmp_path
        / "TAAC2025___tencent_gr-1_m"
        / "mm_emb_81_32"
        / "0.0.0"
        / "hash123"
        / "dataset_info.json"
    )
    dataset_info.parent.mkdir(parents=True)
    dataset_info.write_text(
        '{"splits":{"train":{"num_examples":4742961}}}', encoding="utf-8"
    )

    rows = taac2025_module._estimate_subset_rows(
        "TAAC2025/TencentGR-1M", "mm_emb_81_32", str(tmp_path)
    )

    assert rows == 4742961


def test_load_subset_mm_emb_supports_anonymous_cid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    datasets = pytest.importorskip("datasets")

    class MockDataset:
        def to_pandas(self):
            return pd.DataFrame(
                {
                    "anonymous_cid": ["20000000029", "20000000030"],
                    "emb": [[0.1, 0.2], [0.3, 0.4]],
                }
            )

    monkeypatch.setattr(
        taac2025_module, "_estimate_subset_rows", lambda repo_id, config, cache_dir: 2
    )
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda repo_id, config, split, cache_dir: MockDataset(),
    )

    ds = _build_dataset("1M", tmp_path)
    ds.get_schema_manifest = lambda: SimpleNamespace(
        get_subset=lambda name: taac2025_module.SubsetDescriptor(
            name=name,
            hf_config=name,
            primary_key="item_id",
            join_key="item_id",
            estimated_rows=2,
            recommended_profile="vector",
            is_vector=True,
        )
    )

    store, load_meta = ds._load_subset_mm_emb("mm_emb_81_32", max_rows=10, seed=42)

    assert load_meta is None
    assert store.dim == 2
    assert store.item_ids.tolist() == ["20000000029", "20000000030"]
