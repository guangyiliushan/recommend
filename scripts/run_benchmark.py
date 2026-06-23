"""Batch benchmark runner — CLI entry point.

Usage:
    # 配置文件模式
    uv run python scripts/run_benchmark.py --config configs/experiment/benchmark_classical.yaml
    uv run python scripts/run_benchmark.py --config configs/experiment/benchmark_all.yaml --max-concurrent 2

    # 命令行指定模型和数据集
    uv run python scripts/run_benchmark.py --models itemcf hyformer --datasets taac2026_data_sample

    # Hydra 模式：从 YAML 加载每轮实验的默认参数
    uv run python scripts/run_benchmark.py --config configs/experiment/benchmark_classical.yaml --hydra
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# 触发数据集注册副作用（导入各 dataset adapter 以执行 @DATASET_REGISTRY.register()）
import recsys.data.dataset_registry  # noqa: F401
from recsys import auto_discover_models
from recsys.core.registry import DATASET_REGISTRY, MODEL_REGISTRY
from recsys.pipeline.benchmark import (
    BenchmarkConfig,
    ResumeMode,
    run_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Run a RecBench batch benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c",
        default=None,
        help="配置文件路径 (e.g. configs/experiment/benchmark_classical.yaml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="模型列表 (覆盖配置文件中定义)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="数据集列表 (覆盖配置文件中定义)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="随机种子列表 (默认: [42])",
    )
    parser.add_argument(
        "--resume-mode",
        default="successful_skip",
        choices=["successful_skip", "failed_only", "unfinished_only", "force"],
        help="恢复模式 (默认: successful_skip)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="最大并发数 (默认: 1, 串行)",
    )
    parser.add_argument(
        "--output-root",
        default="./outputs",
        help="输出根目录 (默认: ./outputs)",
    )
    parser.add_argument(
        "--experiment-output-dir",
        default="./outputs/runs",
        help="单实验输出目录 (默认: ./outputs/runs)",
    )
    parser.add_argument(
        "--hydra",
        action="store_true",
        help="从 configs/config.yaml 加载每轮实验的默认参数（split_mode 等）",
    )

    return parser


def _run_benchmark_hydra_mode(
    bench_cfg: BenchmarkConfig,
    models: list,
    datasets: list,
    args: argparse.Namespace,
) -> None:
    """Hydra 模式：从 config.yaml 加载基准配置，注入每个实验参数。

    覆盖 expand_benchmark_config() 的默认空配置行为，
    为每个 (model, dataset, seed) 组合注入 data/training/evaluation 默认参数。
    """
    from recsys.pipeline.experiment import ExperimentConfig as PipelineExperimentConfig
    from recsys.pipeline.experiment import (
        run_experiment,
    )
    from recsys.utils.config import (
        load_config,
        recbench_to_experiment_config,
    )

    # 加载基准 YAML 配置
    base_recbench = load_config("configs/config.yaml")
    base_exp = recbench_to_experiment_config(base_recbench)

    # 按矩阵展开执行
    all_results: list = []
    for model_name in models:
        for dataset_name in datasets:
            for seed in args.seeds:
                cfg = PipelineExperimentConfig(
                    experiment_name=bench_cfg.benchmark_name,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed=seed,
                    output_dir=bench_cfg.experiment_output_dir,
                    data_config=dict(base_exp.data_config),
                    model_config=dict(base_exp.model_config),
                    training_config=dict(base_exp.training_config),
                    evaluation_config=dict(base_exp.evaluation_config),
                    runtime_config=dict(base_exp.runtime_config),
                )
                all_results.append(run_experiment(cfg))

    # 汇总
    succeeded = sum(
        1 for r in all_results
        if r.status.value in ("succeeded", "skipped")
    )
    failed = sum(
        1 for r in all_results if r.status.value == "failed"
    )
    print(f"\nBenchmark completed (Hydra): {succeeded} succeeded, {failed} failed")
    if failed:
        sys.exit(1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 1. 触发模型自动发现
    print("Discovering models...")
    auto_discover_models()
    available_models = MODEL_REGISTRY.list()
    available_datasets = DATASET_REGISTRY.list()
    print(f"  Available models: {available_models}")
    print(f"  Available datasets: {available_datasets}")

    # 2. 解析配置
    models = args.models
    datasets = args.datasets
    benchmark_name = "cli_benchmark"

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: config file not found: {args.config}")
            sys.exit(1)
        with open(config_path, encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f)

        if cfg_data is None:
            print(f"Error: config file is empty or all commented: {args.config}")
            sys.exit(1)

        if models is None:
            models = [m["name"] if isinstance(m, dict) else m for m in cfg_data.get("models", [])]
        if datasets is None:
            datasets = cfg_data.get("datasets", [])
        benchmark_name = cfg_data.get("experiment", {}).get("name", benchmark_name)

    if not models:
        print("Error: no models specified. Use --models or --config.")
        sys.exit(1)
    if not datasets:
        print("Error: no datasets specified. Use --datasets or --config.")
        sys.exit(1)

    # 3. 校验模型和数据集
    for m in models:
        if m not in available_models:
            print(f"Error: model '{m}' not registered. Available: {available_models}")
            sys.exit(1)
    for d in datasets:
        if d not in available_datasets:
            print(f"Error: dataset '{d}' not registered. Available: {available_datasets}")
            sys.exit(1)

    # 4. 构造 BenchmarkConfig
    resume_mode_map = {
        "successful_skip": ResumeMode.SUCCESSFUL_SKIP,
        "failed_only": ResumeMode.FAILED_ONLY,
        "unfinished_only": ResumeMode.UNFINISHED_ONLY,
        "force": ResumeMode.FORCE,
    }
    bench_cfg = BenchmarkConfig(
        benchmark_name=benchmark_name,
        models=models,
        datasets=datasets,
        seeds=args.seeds,
        resume_mode=resume_mode_map[args.resume_mode],
        max_concurrent_runs=args.max_concurrent,
        output_root=args.output_root,
        experiment_output_dir=args.experiment_output_dir,
    )

    # 5. 运行 Benchmark
    total_runs = len(models) * len(datasets) * len(args.seeds)
    print(f"\nStarting benchmark: {benchmark_name}")
    print(f"  Matrix: {len(models)} models x {len(datasets)} datasets x {len(args.seeds)} seeds = {total_runs} runs")

    if args.hydra:
        _run_benchmark_hydra_mode(bench_cfg, models, datasets, args)
        return

    result = run_benchmark(bench_cfg)

    # 6. 打印结果
    succeeded = len(result.succeeded_runs)
    failed = len(result.failed_runs)
    print(f"\nBenchmark completed: {result.status}")
    print(f"  Runs: {succeeded} succeeded, {failed} failed (total: {len(result.runs)})")
    print("  Artifacts:")
    print(f"    Summary:    {result.summary_path}")
    print(f"    Leaderboard: {result.leaderboard_path}")
    print(f"    Failures:   {result.failures_path}")
    print(f"    Manifest:   {result.manifest_path}")
    if result.report_path:
        print(f"    Report:     {result.report_path}")

    if result.status == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
