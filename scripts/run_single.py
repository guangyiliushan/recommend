"""Single experiment runner — CLI entry point.

Two modes:
    argparse mode (default):
        uv run python scripts/run_single.py --model itemcf --dataset taac2026_data_sample --seed 42
        uv run python scripts/run_single.py --model hyformer --dataset taac2026_data_sample --seed 42 --epochs 10

    Hydra mode (--hydra):
        uv run python scripts/run_single.py --hydra \
            --hydra-config config --hydra-overrides model=classical/itemcf
        uv run python scripts/run_single.py --hydra \
            --hydra-overrides model.params.similarity=iuf data.split_mode=random
"""

from __future__ import annotations

import argparse
import json
import sys

# 触发数据集注册副作用（导入各 dataset adapter 以执行 @DATASET_REGISTRY.register()）
import recsys.data.dataset_registry  # noqa: F401
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
        default="./outputs/runs",
        help="实验输出目录 (默认: ./outputs/runs)",
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

    # 数据参数
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--min-action-type",
        type=int,
        default=0,
        help="最小行为类型 (0=全部, 1=仅点击) — TAAC2025 隐式反馈推荐建议设为 1",
    )
    data_group.add_argument(
        "--split-mode",
        choices=["temporal", "random"],
        default="temporal",
        help="数据切分方式：temporal（时序默认）/ random（随机）",
    )

    # 运行时参数
    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--device", default="auto", help="设备 (默认: auto)")
    runtime_group.add_argument("--log-level", default="INFO", help="日志级别")

    # Hydra 模式
    hydra_group = parser.add_argument_group("Hydra")
    hydra_group.add_argument(
        "--hydra",
        action="store_true",
        help="启用 Hydra 配置模式（从 configs/ 加载 YAML，支持组合覆盖）",
    )
    hydra_group.add_argument(
        "--hydra-config",
        default="config",
        help="Hydra 配置名（不含 .yaml 后缀，默认: config）",
    )
    hydra_group.add_argument(
        "--hydra-overrides",
        nargs="*",
        default=[],
        help="Hydra 覆盖项 (如: model.params.similarity=iuf data.split_mode=random)",
    )

    return parser


def _run_hydra_mode(
    args: argparse.Namespace,
    available_models: list,
    available_datasets: list,
) -> None:
    """Hydra 配置模式：从 configs/ YAML 加载配置并运行实验。

    Parameters
    ----------
    args : argparse.Namespace
        解析后的命令行参数。
    available_models : list
        已注册的模型名列表。
    available_datasets : list
        已注册的数据集名列表。
    """
    from recsys.utils.config import (
        load_config,
        recbench_to_experiment_config,
    )

    # 加载 YAML 配置并应用 CLI 覆盖
    config_path = f"configs/{args.hydra_config}.yaml"
    print(f"Loading Hydra config: {config_path}")
    recbench_cfg = load_config(
        config_path=config_path,
        overrides=list(args.hydra_overrides) if args.hydra_overrides else [],
    )

    # 校验数据集和模型是否注册
    if recbench_cfg.data.name not in available_datasets:
        print(
            f"Error: dataset '{recbench_cfg.data.name}' not registered. "
            f"Available: {available_datasets}"
        )
        sys.exit(1)
    if recbench_cfg.model.name not in available_models:
        print(
            f"Error: model '{recbench_cfg.model.name}' not registered. "
            f"Available: {available_models}"
        )
        sys.exit(1)

    # 转换为 pipeline 层配置
    exp_config = recbench_to_experiment_config(recbench_cfg)

    print(
        f"\nStarting experiment (Hydra): {recbench_cfg.experiment.name}"
    )
    print(
        f"  Model: {exp_config.model_name}, "
        f"Dataset: {exp_config.dataset_name}, "
        f"Seed: {exp_config.seed}"
    )
    result = run_experiment(exp_config)

    print(f"\nExperiment completed: {result.status.value}")
    if result.succeeded:
        print(f"  Summary metrics: {json.dumps(result.summary_metrics, indent=2)}")
        print("  Artifacts:")
        for key, path in result.artifact_paths.items():
            print(f"    {key}: {path}")
    else:
        print(f"  Error: {result.error}")
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

    # ---- Hydra mode ----
    if args.hydra:
        _run_hydra_mode(args, available_models, available_datasets)
        return

    # ---- argparse mode (original path) ----
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
        data_config={"root_dir": args.data_root, "min_action_type": args.min_action_type, "split_mode": args.split_mode},
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
            "metrics": args.metrics,  # None = use model defaults (not [])
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
