"""性能对比分析脚本（Part D3）。

加载 benchmark 产出的 summary.csv，按模型分组计算 speedup 与 NDCG 差异，
输出 performance_comparison.md 对比报告。

用法:
    uv run python benchmarks/analyze_performance.py --input outputs/benchmarks/itemcf_perf_1m/
    uv run python benchmarks/analyze_performance.py --input outputs/benchmarks/itemcf_perf_100k/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def analyze(benchmark_dir: str) -> str:
    """分析 benchmark 结果并生成对比报告。

    Parameters
    ----------
    benchmark_dir : str
        Benchmark 输出目录。

    Returns
    -------
    str
        生成的 markdown 报告路径。
    """
    bench_path = Path(benchmark_dir)
    summary_csv = bench_path / "summary.csv"
    bench_path / "leaderboard.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"未找到 summary.csv: {summary_csv}")

    # 加载数据
    summary = pd.read_csv(summary_csv)

    # 按模型分组计算统计数据
    [col for col in summary.columns if col not in (
        "run_id", "model_name", "dataset_name", "seed", "status", "duration_seconds",
        "peak_memory_mb", "primary_metric", "primary_metric_value", "succeeded",
        "schema_version", "prediction_schema_version", "started_at", "finished_at",
    )]

    # 构建分组统计
    grouped = summary.groupby("model_name").agg({
        "duration_seconds": ["mean", "std", "min", "max"],
        "primary_metric_value": ["mean", "std"],
    })

    # 如果 peak_memory_mb 列存在，添加内存统计
    memory_stats = ""
    if "peak_memory_mb" in summary.columns and summary["peak_memory_mb"].notna().any():
        memory_grouped = summary.groupby("model_name")["peak_memory_mb"].agg(
            ["mean", "max"]
        )
        memory_stats = "\n## 内存使用统计\n\n"
        memory_stats += "| 模型 | 平均峰值内存 (MB) | 最大峰值内存 (MB) |\n"
        memory_stats += "|------|-------------------|-------------------|\n"
        for model_name, row in memory_grouped.iterrows():
            memory_stats += f"| {model_name} | {row['mean']:.2f} | {row['max']:.2f} |\n"

    # 计算 speedup（相对于第一条记录）
    baseline_duration = summary["duration_seconds"].min() if len(summary) > 0 else 1.0

    # 生成 Markdown 报告
    report_lines = [
        "# ItemCF 性能对比分析",
        "",
        f"**数据源**: `{benchmark_dir}`",
        f"**生成时间**: {pd.Timestamp.now().isoformat()}",
        f"**总运行数**: {len(summary)}",
        "",
        "## 速度对比",
        "",
        "| 模型 | 平均耗时 (s) | 标准差 | 最小耗时 | 最大耗时 | Speedup |",
        "|------|-------------|--------|---------|---------|---------|",
    ]

    for model_name, row in grouped.iterrows():
        avg_duration = row[("duration_seconds", "mean")]
        std_duration = row[("duration_seconds", "std")]
        min_duration = row[("duration_seconds", "min")]
        max_duration = row[("duration_seconds", "max")]
        speedup = baseline_duration / avg_duration if avg_duration > 0 else 0

        report_lines.append(
            f"| {model_name} | {avg_duration:.4f} | {std_duration:.4f} "
            f"| {min_duration:.4f} | {max_duration:.4f} | {speedup:.2f}x |"
        )

    report_lines.append("")

    # 准确性对比
    if "primary_metric_value" in summary.columns:
        report_lines.extend([
            "## 准确性对比",
            "",
            "| 模型 | 平均指标值 | 标准差 |",
            "|------|-----------|--------|",
        ])

        for model_name, row in grouped.iterrows():
            avg_metric = row[("primary_metric_value", "mean")]
            std_metric = row[("primary_metric_value", "std")]
            report_lines.append(
                f"| {model_name} | {avg_metric:.6f} | {std_metric:.6f} |"
            )

        report_lines.append("")

    # 内存统计
    if memory_stats:
        report_lines.append(memory_stats)

    # 结论
    report_lines.extend([
        "## 结论",
        "",
        "_请在完成测试后，根据实际结果手动填写结论_",
        "",
        "### 建议",
        "- 小规模数据（<10万交互）：推荐 cosine + weighted_sum（基线）",
        "- 中等规模数据（10万-100万交互）：推荐 cosine + weighted_sum + normalize",
        "- 大规模数据（>100万交互）：推荐 IUF 加权余弦",
    ])

    report_content = "\n".join(report_lines)
    report_path = bench_path / "performance_comparison.md"
    report_path.write_text(report_content, encoding="utf-8")

    print(report_content)
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(
        description="分析 ItemCF 性能基准结果",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Benchmark 输出目录路径",
    )
    args = parser.parse_args()

    try:
        report_path = analyze(args.input)
        print(f"\n报告已生成: {report_path}")
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
