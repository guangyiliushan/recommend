"""Ablation study runner — 消融实验矩阵执行。

对单一超参数构建多组变体配置，逐组调用 ``run_experiment()``，
输出 CSV 对比表。

Usage:
    uv run python scripts/run_ablation.py --model itemcf --dataset taac2026_data_sample
    uv run python scripts/run_ablation.py --model itemcf --vary top_k_neighbors 10 20 50 100
    uv run python scripts/run_ablation.py --model itemcf --vary similarity cosine iuf \\
        --output outputs/ablations/
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# 触发数据集注册副作用（导入各 dataset adapter 以执行 @DATASET_REGISTRY.register()）
import recsys.data.dataset_registry  # noqa: F401
from recsys import auto_discover_models
from recsys.core.registry import DATASET_REGISTRY, MODEL_REGISTRY
from recsys.pipeline.experiment import ExperimentConfig, run_experiment

# 模型支持的消融参数及其默认候选值
_ABLATION_PARAMS: dict[str, dict[str, list[Any]]] = {
    "itemcf": {
        "similarity": ["cosine", "iuf"],
        "top_k_neighbors": [10, 20, 50, 100],
        "recommend_k": [5, 10, 20],
        "normalize": [True, False],
    },
}


def _parse_vary_arg(raw: list[str]) -> tuple[str, list[str]]:
    """解析 --vary 参数：参数名 + 候选值列表。"""
    if len(raw) < 2:
        raise ValueError("--vary 至少需要: <参数名> <值1> <值2> ...")
    return raw[0], raw[1:]


@dataclass
class AblationResult:
    """单次消融实验的结果。"""

    param: str
    value: Any
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "unknown"
    error: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="消融实验 — 对超参数变体执行对比评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", "-m", required=True, help="模型注册名")
    parser.add_argument("--dataset", "-d", required=True, help="数据集注册名")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vary",
        nargs="+",
        default=None,
        help="消融参数: <参数名> <候选值1> <候选值2> ... (不指定则使用模型默认候选)",
    )
    parser.add_argument(
        "--output", "-o",
        default="./outputs/ablations",
        help="输出根目录 (默认: ./outputs/ablations)",
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["ndcg_at_k", "recall_at_k", "hit_rate_at_k", "mrr"],
    )
    args = parser.parse_args()

    # 解析消融参数
    if args.vary:
        param_name, candidates = _parse_vary_arg(args.vary)
    else:
        defaults = _ABLATION_PARAMS.get(args.model, {})
        if not defaults:
            print(
                f"错误: 模型 '{args.model}' 没有默认消融参数，请用 --vary 指定",
                file=sys.stderr,
            )
            sys.exit(1)
        param_name = next(iter(defaults.keys()))
        candidates = [str(v) for v in defaults[param_name]]
        print(f"使用默认消融参数: {param_name} = {candidates}")

    # 发现模型
    auto_discover_models()
    available_models = MODEL_REGISTRY.list()
    available_datasets = DATASET_REGISTRY.list()
    if args.model not in available_models:
        print(f"错误: 模型 '{args.model}' 未注册。可用: {available_models}")
        sys.exit(1)
    if args.dataset not in available_datasets:
        print(f"错误: 数据集 '{args.dataset}' 未注册。可用: {available_datasets}")
        sys.exit(1)

    # 执行消融
    results: list[AblationResult] = []
    total = len(candidates)

    for i, value in enumerate(candidates):
        label = f"[{i + 1}/{total}] {param_name}={value}"
        print(f"{label} ...", end=" ", flush=True)

        # 将 string 候选值转为合适的 Python 类型
        typed_value: Any = value
        if value.lower() in ("true", "false"):
            typed_value = value.lower() == "true"
        elif value.isdigit():
            typed_value = int(value)
        elif value.replace(".", "", 1).isdigit():
            typed_value = float(value)

        cfg = ExperimentConfig(
            experiment_name=f"ablation_{args.model}_{args.dataset}_{param_name}_{value}",
            dataset_name=args.dataset,
            model_name=args.model,
            seed=args.seed,
            output_dir=str(Path(args.output) / "experiments"),
            data_config={"root_dir": args.data_root},
            model_config={"params": {param_name: typed_value}},
            evaluation_config={
                "metrics": args.metrics,
                "ranking_k": [10],
                "generate_curves": False,
            },
        )

        try:
            result = run_experiment(cfg)
            r = AblationResult(
                param=param_name,
                value=value,
                metrics=dict(result.summary_metrics),
                status=result.status.value,
            )
        except Exception as exc:
            r = AblationResult(
                param=param_name,
                value=value,
                status="failed",
                error=str(exc),
            )

        status_icon = "OK" if r.status == "succeeded" else f"ERR: {r.error}"
        print(status_icon)
        results.append(r)

    # 输出 CSV 对比表
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "param": r.param,
            "value": str(r.value),
            "status": r.status,
            **{k: v for k, v in r.metrics.items()},
            "error": r.error or "",
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"ablation_{args.model}_{args.dataset}_{param_name}.csv"
    df.to_csv(csv_path, index=False)

    succeeded = sum(1 for r in results if r.status == "succeeded")
    failed = sum(1 for r in results if r.status == "failed")
    print(f"\n消融完成: {succeeded} succeeded, {failed} failed")
    print(f"结果已保存: {csv_path}")
    print(f"\n{df[['param', 'value', 'status', *list(results[-1].metrics.keys())]].to_string(index=False)}")


if __name__ == "__main__":
    main()
