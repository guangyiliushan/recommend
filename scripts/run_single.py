"""Single experiment runner — CLI entry point.

Usage:
    uv run python scripts/run_single.py --model itemcf --dataset taac2026_data_sample --seed 42
    uv run python scripts/run_single.py --model dssm --dataset taac2026_data_sample --seed 42 --epochs 10 --batch-size 128
    uv run python scripts/run_single.py --model dssm --dataset taac2026_data_sample --seed 42 --lr 3e-4 --optimizer adamw
"""

from __future__ import annotations

import argparse
import json
import sys

from recsys import auto_discover_models
from recsys.core.registry import DATASET_REGISTRY, MODEL_REGISTRY
from recsys.pipeline.experiment import ExperimentConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Run a single RecBench experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 核心参数
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="模型注册名 (e.g. itemcf, dssm)",
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="数据集注册名 (e.g. taac2026_data_sample)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/experiments",
        help="实验输出目录 (默认: ./outputs/experiments)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="实验名称 (默认: {model}_{dataset})",
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        help="数据根目录 (默认: ./data)",
    )

    # 训练参数
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=10, help="训练轮数 (默认: 10)")
    train_group.add_argument("--batch-size", type=int, default=256, help="批次大小 (默认: 256)")
    train_group.add_argument("--lr", type=float, default=1e-3, help="学习率 (默认: 1e-3)")
    train_group.add_argument("--optimizer", default="adam", help="优化器 (默认: adam)")
    train_group.add_argument("--scheduler", default=None, help="学习率调度器")
    train_group.add_argument("--weight-decay", type=float, default=1e-5, help="权重衰减")
    train_group.add_argument("--warmup-epochs", type=int, default=0, help="Warmup 轮数")

    # 评估参数
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="评估指标列表 (默认: 模型默认指标)",
    )
    eval_group.add_argument(
        "--ranking-k",
        type=int,
        nargs="+",
        default=[10],
        help="Ranking Top-K 列表 (默认: [10])",
    )
    eval_group.add_argument(
        "--no-curves",
        action="store_true",
        help="禁用曲线生成",
    )

    # 运行时参数
    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--device", default="auto", help="设备 (默认: auto)")
    runtime_group.add_argument("--log-level", default="INFO", help="日志级别")

    return parser


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

    # 2. 校验模型和数据集
    if args.model not in available_models:
        print(f"Error: model '{args.model}' not found. Available: {available_models}")
        sys.exit(1)
    if args.dataset not in available_datasets:
        print(f"Error: dataset '{args.dataset}' not found. Available: {available_datasets}")
        sys.exit(1)

    # 3. 构造 ExperimentConfig
    experiment_name = args.name or f"{args.model}_{args.dataset}"
    config = ExperimentConfig(
        experiment_name=experiment_name,
        dataset_name=args.dataset,
        model_name=args.model,
        seed=args.seed,
        output_dir=args.output_dir,
        data_config={"root_dir": args.data_root},
        training_config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
        },
        evaluation_config={
            "metrics": args.metrics or [],
            "ranking_k": args.ranking_k,
            "generate_curves": not args.no_curves,
        },
        runtime_config={
            "device": args.device,
            "seed": args.seed,
            "log_level": args.log_level,
        },
    )

    # 4. 运行实验
    print(f"\nStarting experiment: {experiment_name}")
    print(f"  Model: {args.model}, Dataset: {args.dataset}, Seed: {args.seed}")
    result = run_experiment(config)

    # 5. 打印结果
    print(f"\nExperiment completed: {result.status.value}")
    if result.succeeded:
        print(f"  Summary metrics: {json.dumps(result.summary_metrics, indent=2)}")
        print("  Artifacts:")
        for key, path in result.artifact_paths.items():
            print(f"    {key}: {path}")
    else:
        print(f"  Error: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
