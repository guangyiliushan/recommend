"""Report generation — 报告生成引导脚本。

当前 Benchmark 的 Reporter 已可在 benchmark 运行时自动生成聚合报告，
无需单独调用本脚本。

Reporter 自动生成的产物包括：
  - summary.csv: 逐 run 摘要表
  - leaderboard.csv: 排序视图
  - failures.csv: 失败排查表
  - trend.csv: 趋势表
  - stability.csv: 稳定性统计
  - report.html: 人工浏览摘要页

如需单独从已有结果重新生成报告，请使用 Python API:
  from recsys.pipeline.reporter import Reporter, ReporterConfig
  reporter = Reporter(ReporterConfig(...))
  reporter.generate(results)

本脚本将在后续版本中支持更丰富的报告格式（LaTeX、Markdown 等）。
"""

from __future__ import annotations


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
