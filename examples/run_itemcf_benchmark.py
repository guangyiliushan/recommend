"""最小 benchmark demo：itemcf x taac2026_data_sample x 3 seeds。

运行方式：
    uv run python examples/run_itemcf_benchmark.py

前置条件：
    - uv sync --extra dev 已执行
    - 网络可用（数据集从 HuggingFace Hub 下载并缓存到 ./data）
"""

from recsys.pipeline.benchmark import BenchmarkConfig, ResumeMode, run_benchmark

bench_cfg = BenchmarkConfig(
    benchmark_name="demo_itemcf_benchmark",
    models=["itemcf"],
    datasets=["taac2026_data_sample"],
    seeds=[42, 43, 44],
    resume_mode=ResumeMode.SUCCESSFUL_SKIP,
    max_concurrent_runs=1,
    output_root="./outputs",
    experiment_output_dir="./outputs/experiments",
)

result = run_benchmark(bench_cfg)
print(f"status: {result.status}")
print(f"summary: {result.summary_path}")
print(f"leaderboard: {result.leaderboard_path}")
print(f"report: {result.report_path}")
