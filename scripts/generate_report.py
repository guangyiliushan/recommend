"""Report generation — 从 benchmark 结果生成 Markdown 性能对比报告。

从 ``summary.csv`` 加载实验数据，按模型分组计算耗时/指标的均值与标准差，
输出 ``performance_comparison.md``。

Usage:
    uv run python scripts/generate_report.py --input outputs/benchmarks/benchmark_classical/
    uv run python scripts/generate_report.py --input outputs/benchmarks/benchmark_all/
    uv run python scripts/generate_report.py --input outputs/experiments/ --output reports/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def generate(input_dir: str, output_dir: str | None = None) -> str:
    """从 summary.csv 生成对比报告。

    Parameters
    ----------
    input_dir : str
        包含 summary.csv 的 benchmark 输出目录。
    output_dir : str or None
        报告输出目录（默认与 input_dir 相同）。

    Returns
    -------
    str
        生成的 markdown 报告路径。
    """
    bench_path = Path(input_dir)
    summary_csv = bench_path / "summary.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"未找到 summary.csv: {summary_csv}")

    output_path = Path(output_dir or input_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载数据
    summary = pd.read_csv(summary_csv)

    # 按模型分组计算统计
    group_cols = ["duration_seconds"]
    if "primary_metric_value" in summary.columns:
        group_cols.append("primary_metric_value")

    aggs: dict = {}
    for col in group_cols:
        if col in summary.columns and summary[col].notna().any():
            aggs[col] = ["mean", "std", "min", "max"]

    if not aggs:
        print("Warning: summary.csv 中无可聚合的数值列", file=sys.stderr)
        return ""

    grouped = summary.groupby("model_name").agg(aggs)

    # 基线耗时（最快的 run）
    baseline_duration = (
        summary["duration_seconds"].min()
        if "duration_seconds" in summary.columns and len(summary) > 0
        else 1.0
    )

    # 构建报告
    lines: list[str] = [
        "# 性能对比分析",
        "",
        f"**数据源**: `{input_dir}`",
        f"**生成时间**: {pd.Timestamp.now().isoformat()}",
        f"**总运行数**: {len(summary)}",
        "",
        "## 速度对比",
        "",
        "| 模型 | 平均耗时 (s) | 标准差 | 最小耗时 | 最大耗时 | Speedup |",
        "|------|-------------|--------|---------|---------|---------|",
    ]

    has_duration = ("duration_seconds", "mean") in grouped.columns
    if has_duration:
        for model_name, row in grouped.iterrows():
            avg = row[("duration_seconds", "mean")]
            std = row[("duration_seconds", "std")]
            rmin = row[("duration_seconds", "min")]
            rmax = row[("duration_seconds", "max")]
            speedup = baseline_duration / avg if avg > 0 else 0
            lines.append(
                f"| {model_name} | {avg:.4f} | {std:.4f} "
                f"| {rmin:.4f} | {rmax:.4f} | {speedup:.2f}x |"
            )
        lines.append("")

    # 准确性（如果 primary_metric_value 存在）
    has_metric = ("primary_metric_value", "mean") in grouped.columns
    if has_metric:
        lines.extend([
            "## 准确性对比",
            "",
            "| 模型 | 平均指标值 | 标准差 |",
            "|------|-----------|--------|",
        ])
        for model_name, row in grouped.iterrows():
            avg_metric = row[("primary_metric_value", "mean")]
            std_metric = row[("primary_metric_value", "std")]
            lines.append(f"| {model_name} | {avg_metric:.6f} | {std_metric:.6f} |")
        lines.append("")

    # 内存统计
    if "peak_memory_mb" in summary.columns and summary["peak_memory_mb"].notna().any():
        memory_grouped = summary.groupby("model_name")["peak_memory_mb"].agg(["mean", "max"])
        lines.extend([
            "## 内存使用",
            "",
            "| 模型 | 平均峰值内存 (MB) | 最大峰值内存 (MB) |",
            "|------|-------------------|-------------------|",
        ])
        for model_name, row in memory_grouped.iterrows():
            lines.append(f"| {model_name} | {row['mean']:.2f} | {row['max']:.2f} |")
        lines.append("")

    report_content = "\n".join(lines)
    report_path = output_path / "performance_comparison.md"
    report_path.write_text(report_content, encoding="utf-8")

    print(report_content)
    return str(report_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 benchmark summary.csv 生成 Markdown 对比报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="包含 summary.csv 的目录路径",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="报告输出目录 (默认与 --input 相同)",
    )
    args = parser.parse_args()

    try:
        report_path = generate(args.input, args.output)
        if report_path:
            print(f"\n报告已生成: {report_path}")
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
