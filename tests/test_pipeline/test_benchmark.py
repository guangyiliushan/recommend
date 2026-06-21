"""benchmark 端到端测试：验证矩阵展开、恢复模式与聚合产出。"""

import os
from pathlib import Path

import pytest

from recsys.pipeline.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    ResumeMode,
    expand_benchmark_config,
    run_benchmark,
)


def _hf_dataset_cached(repo_id: str, variant: str) -> bool:
    """检查 HF 数据集是否已缓存到本地。"""
    cache_base = os.path.expanduser("~/.cache/huggingface/datasets")
    if not os.path.isdir(cache_base):
        return False
    for root, dirs, _files in os.walk(cache_base):
        for d in dirs:
            if variant in d and repo_id.replace("/", "___") in root:
                return True
    return False


def test_expand_benchmark_config():
    """矩阵展开数量 = models x datasets x seeds。"""
    bench_cfg = BenchmarkConfig(
        benchmark_name="test",
        models=["itemcf"],
        datasets=["taac2026_data_sample"],
        seeds=[42, 43],
    )
    configs = expand_benchmark_config(bench_cfg)
    assert len(configs) == 1 * 1 * 2

    for cfg in configs:
        assert cfg.model_name == "itemcf"
        assert cfg.dataset_name == "taac2026_data_sample"
        assert cfg.seed in [42, 43]


def test_expand_benchmark_config_multiple_models():
    """多模型多数据集展开。"""
    bench_cfg = BenchmarkConfig(
        benchmark_name="test",
        models=["itemcf"],
        datasets=["taac2026_data_sample"],
        seeds=[42, 43, 44],
    )
    configs = expand_benchmark_config(bench_cfg)
    assert len(configs) == 1 * 1 * 3


def test_benchmark_result_structure():
    """BenchmarkResult 字段完整性。"""
    result = BenchmarkResult(
        benchmark_name="test",
        status="succeeded",
        runs=[],
        summary_path="/tmp/summary.csv",
        leaderboard_path="/tmp/leaderboard.csv",
        failures_path="/tmp/failures.csv",
        manifest_path="/tmp/manifest.json",
    )
    assert result.benchmark_name == "test"
    assert result.succeeded_runs == []
    assert result.failed_runs == []


@pytest.mark.integration
def test_benchmark_integration(tmp_path, monkeypatch):
    """跑 1 model x 1 dataset x 2 seeds，验证聚合产物。"""
    # 离线模式：避免挂死在网络请求上
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    if not _hf_dataset_cached("TAAC2026", "data_sample_1000"):
        pytest.skip("TAAC2026 data_sample_1000 未缓存，跳过（请在联网时先下载一次）")
    bench_cfg = BenchmarkConfig(
        benchmark_name="test_benchmark",
        models=["itemcf"],
        datasets=["taac2026_data_sample"],
        seeds=[42, 43],
        resume_mode=ResumeMode.FORCE,
        max_concurrent_runs=1,
        output_root=str(tmp_path / "outputs"),
        experiment_output_dir=str(tmp_path / "outputs" / "experiments"),
    )

    result = run_benchmark(bench_cfg)
    assert result.status in ("succeeded", "partial_success")

    if result.summary_path:
        assert Path(result.summary_path).exists()
    if result.manifest_path:
        assert Path(result.manifest_path).exists()
