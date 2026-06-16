"""Results aggregation and report generation — 结构化结果聚合器。

v2 新增：
- 趋势分析 (trend.csv)：同一模型×数据集的指标随 seed 变化趋势
- 稳定性视图 (stability.csv)：跨 seed 的 mean / std / cv 分析
- 交互式 HTML（内联 JS 排序，无外部依赖）
- Leaderboard 按 task_type 分组

职责：
- 消费 ExperimentResult 列表，生成 CSV/JSON/HTML 聚合产物
- 对比表 (summary.csv)、排行榜 (leaderboard.csv)、失败列表 (failures.csv)
- 趋势表 (trend.csv)、稳定性表 (stability.csv)
- 基准索引 (manifest.json)、交互式 HTML 摘要页 (report.html)

边界：
- 只消费 ExperimentResult，不接触 model / dataset / trainer 对象
- CSV/JSON first，HTML second，图表 later（v3）
- 可离线重建：所有输入来自结构化数据，不依赖运行时状态

功能分层（与 development.md 对齐）：
- v1: summary.csv, leaderboard.csv, failures.csv, manifest.json, report.html
- v2: trend.csv, stability.csv, 交互式 HTML 排序
- v3: ROC/PR 曲线引用、显著性检验、LaTeX table export
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypeVar

from recsys.pipeline.experiment import (
    ExperimentResult,
    ExperimentStatus,
)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ReporterConfig:
    """Reporter 配置。"""

    benchmark_name: str
    output_dir: str
    primary_metric: str = ""
    task_type: str = ""
    generate_html: bool = True
    generate_leaderboard: bool = True
    generate_failures: bool = True
    # v2
    generate_trend: bool = True
    generate_stability: bool = True


@dataclass
class SummaryRow:
    """summary.csv 中的一行。

    字段语义与 benchmarking.md / artifacts.md 一致。
    """

    run_id: str
    dataset: str
    model: str
    seed: int
    status: str
    primary_metric: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class LeaderboardRow:
    """leaderboard.csv 中的一行。

    按 model + dataset + primary_metric 聚合，
    只包含 status=succeeded 的 run。
    """

    model: str
    dataset: str
    primary_metric_name: str
    mean: float
    std: float
    rank: int
    num_runs: int


@dataclass
class FailureRow:
    """failures.csv 中的一行。

    只包含 status=failed 的 run。
    """

    run_id: str
    dataset: str
    model: str
    seed: int
    phase: str
    error_code: str
    error_message: str


# v2: 趋势与稳定性数据结构

@dataclass
class TrendRow:
    """trend.csv 中的一行。

    展示同一 model × dataset 组合下，各 seed 的指标变化趋势。
    """

    model: str
    dataset: str
    seed: int
    task_type: str
    primary_metric_name: str
    primary_metric_value: Optional[float] = None
    duration_seconds: Optional[float] = None
    num_users: Optional[int] = None
    num_items: Optional[int] = None


@dataclass
class StabilityRow:
    """stability.csv 中的一行。

    跨 seed 的指标稳定性分析（mean / std / cv）。
    """

    model: str
    dataset: str
    task_type: str
    primary_metric_name: str
    mean: float
    std: float
    cv: float  # coefficient of variation = std / |mean|
    min_val: float
    max_val: float
    num_seeds: int


# ============================================================================
# 通用 CSV / JSON 写入
# ============================================================================

T = TypeVar("T")


def _write_csv(
    path: Path,
    rows: Sequence[Any],
    fieldnames: List[str],
    extra_metric_keys: Optional[List[str]] = None,
) -> str:
    """通用 CSV 写入，支持可选指标列扩展。"""
    all_fields = list(fieldnames)
    if extra_metric_keys:
        all_fields = all_fields + extra_metric_keys

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(all_fields)
        for row in rows:
            d = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            metrics = d.pop("metrics", {}) if isinstance(d, dict) else {}
            base = [d.get(f, "") for f in fieldnames if f != "metrics"]
            for key in (extra_metric_keys or []):
                base.append(metrics.get(key, ""))
            writer.writerow(base)

    return str(path.resolve())


def _write_json(path: Path, data: Any) -> str:
    """通用 JSON 写入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return str(path.resolve())


# ============================================================================
# 行提取与聚合
# ============================================================================

def extract_summary_row(result: ExperimentResult) -> SummaryRow:
    """从 ExperimentResult 提取一行 summary。"""
    primary_metric = result.metadata.get("primary_metric")
    primary_value: Optional[float] = None
    if primary_metric and primary_metric in result.summary_metrics:
        primary_value = result.summary_metrics[primary_metric]

    return SummaryRow(
        run_id=result.run_id,
        dataset=result.metadata.get("dataset_name", ""),
        model=result.metadata.get("model_name", ""),
        seed=result.metadata.get("seed", 0),
        status=result.status.value,
        primary_metric=primary_value,
        metrics=result.summary_metrics,
    )


def extract_failure_row(result: ExperimentResult) -> Optional[FailureRow]:
    """从失败的 ExperimentResult 提取一行 failure。

    只对 status=failed 的 run 生效。
    """
    if result.status != ExperimentStatus.FAILED:
        return None
    if result.error is None:
        return FailureRow(
            run_id=result.run_id,
            dataset=result.metadata.get("dataset_name", ""),
            model=result.metadata.get("model_name", ""),
            seed=result.metadata.get("seed", 0),
            phase="unknown",
            error_code="UNKNOWN_ERROR",
            error_message="no structured error available",
        )

    return FailureRow(
        run_id=result.run_id,
        dataset=result.metadata.get("dataset_name", ""),
        model=result.metadata.get("model_name", ""),
        seed=result.metadata.get("seed", 0),
        phase=result.error.phase.value if result.error.phase else "unknown",
        error_code=result.error.code,
        error_message=result.error.message,
    )


def aggregate_leaderboard(
    summary_rows: List[SummaryRow],
    primary_metric: Optional[str] = None,
) -> List[LeaderboardRow]:
    """按 model + dataset 聚合 leaderboard。

    只包含 status=succeeded 的 run。
    """
    # 过滤成功的 run
    successful = [r for r in summary_rows if r.status == "succeeded"]

    # 确定主指标
    if primary_metric is None and successful:
        # 取第一个 run 的 metrics keys 作为参考
        first_metrics = successful[0].metrics
        if first_metrics:
            primary_metric = next(iter(first_metrics.keys()))

    if primary_metric is None:
        return []

    # 按 model + dataset 分组
    groups: Dict[tuple, List[float]] = {}
    for row in successful:
        key = (row.model, row.dataset)
        val = row.metrics.get(primary_metric)
        if val is not None:
            groups.setdefault(key, []).append(val)

    # 计算 mean / std
    import statistics
    rows: List[LeaderboardRow] = []
    for (model, dataset), values in groups.items():
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(LeaderboardRow(
            model=model,
            dataset=dataset,
            primary_metric_name=primary_metric,
            mean=mean,
            std=std,
            rank=0,
            num_runs=len(values),
        ))

    # 排序并设 rank（mean 越高越好，降序）
    rows.sort(key=lambda r: r.mean, reverse=True)
    for i, row in enumerate(rows):
        row.rank = i + 1

    return rows


# v2: 趋势与稳定性分析

def extract_trend_rows(
    results: List[ExperimentResult],
    primary_metric: Optional[str] = None,
) -> List[TrendRow]:
    """从 ExperimentResult 列表提取趋势行。

    仅保留 status=succeeded 的 run。
    """
    rows: List[TrendRow] = []
    for r in results:
        if r.status != ExperimentStatus.SUCCEEDED:
            continue
        meta = r.metadata
        pm = primary_metric or meta.get("primary_metric")
        pv: Optional[float] = None
        if pm and pm in r.summary_metrics:
            pv = r.summary_metrics[pm]

        rows.append(TrendRow(
            model=meta.get("model_name", ""),
            dataset=meta.get("dataset_name", ""),
            seed=meta.get("seed", 0),
            task_type=meta.get("task_type", ""),
            primary_metric_name=pm or "",
            primary_metric_value=pv,
            duration_seconds=meta.get("duration_seconds"),
            num_users=meta.get("num_users"),
            num_items=meta.get("num_items"),
        ))

    return rows


def analyze_stability(
    trend_rows: List[TrendRow],
    primary_metric: Optional[str] = None,
) -> List[StabilityRow]:
    """从趋势行中计算跨 seed 稳定性指标。

    按 model + dataset + task_type 分组。
    至少需要 2 个 seed 才有意义。
    """
    # 按 model + dataset + task_type 分组
    groups: Dict[tuple, List[TrendRow]] = {}
    for row in trend_rows:
        key = (row.model, row.dataset, row.task_type)
        groups.setdefault(key, []).append(row)

    rows: List[StabilityRow] = []
    for (model, dataset, task_type), group in groups.items():
        vals = [r.primary_metric_value for r in group if r.primary_metric_value is not None]
        if len(vals) < 2:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals)
        cv = abs(std / mean) if mean != 0 else float("inf")
        rows.append(StabilityRow(
            model=model,
            dataset=dataset,
            task_type=task_type,
            primary_metric_name=group[0].primary_metric_name or (primary_metric or ""),
            mean=mean,
            std=std,
            cv=cv,
            min_val=min(vals),
            max_val=max(vals),
            num_seeds=len(vals),
        ))

    # 按 cv 升序（越稳定越靠前）
    rows.sort(key=lambda r: r.cv)
    return rows


# ============================================================================
# Reporter 主类
# ============================================================================

class Reporter:
    """结构化结果聚合器。

    消费 ExperimentResult 列表，生成 CSV/JSON/HTML 聚合产物。
    不接触任何运行时对象（model、dataset、trainer）。
    """

    def __init__(self, config: ReporterConfig) -> None:
        self._config = config
        self._output_dir = Path(config.output_dir)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def generate(self, results: List[ExperimentResult]) -> Dict[str, str]:
        """生成所有聚合产物。

        Returns
        -------
        Dict[str, str]
            产物路径字典：summary_csv, leaderboard_csv, failures_csv,
            manifest_json, report_html。
        """
        paths: Dict[str, str] = {}

        # summary.csv
        paths["summary_csv"] = self.generate_summary_csv(results)

        # leaderboard.csv
        if self._config.generate_leaderboard:
            lb_path = self.generate_leaderboard_csv(results)
            if lb_path:
                paths["leaderboard_csv"] = lb_path

        # failures.csv
        if self._config.generate_failures:
            paths["failures_csv"] = self.generate_failures_csv(results)

        # report.html
        if self._config.generate_html:
            paths["report_html"] = self.generate_html_report(
                paths.get("summary_csv", ""),
                paths.get("leaderboard_csv"),
                paths.get("failures_csv", ""),
            )

        # v2: trend.csv
        if self._config.generate_trend:
            paths["trend_csv"] = self.generate_trend_csv(results)

        # v2: stability.csv
        if self._config.generate_stability:
            stability_path = self.generate_stability_csv(results)
            if stability_path:
                paths["stability_csv"] = stability_path

        return paths

    # ------------------------------------------------------------------
    # 各产物生成方法
    # ------------------------------------------------------------------

    def generate_summary_csv(
        self,
        results: List[ExperimentResult],
    ) -> str:
        """生成 summary.csv。

        每行一个 run，包含 run_id/dataset/model/seed/status/primary_metric
        以及所有浮动指标列。
        """
        rows = [extract_summary_row(r) for r in results]

        # 收集所有指标键名
        metric_keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row.metrics:
                if key not in seen:
                    seen.add(key)
                    metric_keys.append(key)

        path = self._output_dir / "summary.csv"
        fieldnames = [
            "run_id", "dataset", "model", "seed", "status", "primary_metric",
        ]
        return _write_csv(path, rows, fieldnames, metric_keys)

    def generate_leaderboard_csv(
        self,
        results: List[ExperimentResult],
    ) -> Optional[str]:
        """生成 leaderboard.csv。

        按 model + dataset 聚合，只包含成功 run。
        若没有成功 run 或无法确定主指标，返回 None。
        """
        summary_rows = [extract_summary_row(r) for r in results]
        primary_metric = self._config.primary_metric
        leaderboard_rows = aggregate_leaderboard(summary_rows, primary_metric)

        if not leaderboard_rows:
            return None

        path = self._output_dir / "leaderboard.csv"
        fieldnames = [
            "model", "dataset", "primary_metric_name", "mean", "std", "rank", "num_runs",
        ]
        return _write_csv(path, leaderboard_rows, fieldnames)

    def generate_failures_csv(
        self,
        results: List[ExperimentResult],
    ) -> str:
        """生成 failures.csv。

        只包含 status=failed 的 run。
        """
        rows: List[FailureRow] = []
        for r in results:
            frow = extract_failure_row(r)
            if frow is not None:
                rows.append(frow)

        path = self._output_dir / "failures.csv"
        fieldnames = [
            "run_id", "dataset", "model", "seed",
            "phase", "error_code", "error_message",
        ]
        return _write_csv(path, rows, fieldnames)

    # v2: 趋势与稳定性报告

    def generate_trend_csv(
        self,
        results: List[ExperimentResult],
    ) -> str:
        """v2: 生成 trend.csv。

        同一 model × dataset 组合下，各 seed 的指标变化趋势。
        包含 duration_seconds / num_users / num_items 等辅助列。
        """
        trend_rows = extract_trend_rows(results, self._config.primary_metric or None)
        path = self._output_dir / "trend.csv"
        fieldnames = [
            "model", "dataset", "seed", "task_type",
            "primary_metric_name", "primary_metric_value",
            "duration_seconds", "num_users", "num_items",
        ]
        return _write_csv(path, trend_rows, fieldnames)

    def generate_stability_csv(
        self,
        results: List[ExperimentResult],
    ) -> Optional[str]:
        """v2: 生成 stability.csv。

        跨 seed 的指标稳定性分析（mean / std / cv）。
        若 seed 数不足 2 则返回 None。
        """
        trend_rows = extract_trend_rows(results, self._config.primary_metric or None)
        stability_rows = analyze_stability(trend_rows, self._config.primary_metric or None)

        if not stability_rows:
            return None

        path = self._output_dir / "stability.csv"
        fieldnames = [
            "model", "dataset", "task_type", "primary_metric_name",
            "mean", "std", "cv", "min_val", "max_val", "num_seeds",
        ]
        return _write_csv(path, stability_rows, fieldnames)

    def generate_manifest(
        self,
        results: List[ExperimentResult],
        benchmark_cfg: Any,  # BenchmarkConfig，避免循环导入
    ) -> str:
        """生成 manifest.json。

        一般由 benchmark.py 调用，也可独立使用。
        """
        manifest = {
            "benchmark_name": self._config.benchmark_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runs": [r.run_id for r in results],
            "models": getattr(benchmark_cfg, "models", []),
            "datasets": getattr(benchmark_cfg, "datasets", []),
            "seeds": getattr(benchmark_cfg, "seeds", []),
        }
        path = self._output_dir / "manifest.json"
        return _write_json(path, manifest)

    def generate_html_report(
        self,
        summary_path: str,
        leaderboard_path: Optional[str],
        failures_path: str,
    ) -> str:
        """生成最小 HTML 摘要页。

        v1 为纯静态 HTML，内联 summary / leaderboard / failures 数据。
        v2 可升级为带排序、搜索的交互式表格。
        """
        # 读取 summary 数据
        summary_data = _read_csv_as_dicts(summary_path) if summary_path else []
        leaderboard_data = (
            _read_csv_as_dicts(leaderboard_path) if leaderboard_path else []
        )
        failures_data = (
            _read_csv_as_dicts(failures_path) if failures_path else []
        )

        succeeded = sum(
            1 for r in summary_data
            if r.get("status") in ("succeeded", "skipped")
        )
        failed = sum(1 for r in summary_data if r.get("status") == "failed")

        html = _build_html(
            benchmark_name=self._config.benchmark_name,
            total=len(summary_data),
            succeeded=succeeded,
            failed=failed,
            summary_rows=summary_data,
            leaderboard_rows=leaderboard_data,
            failure_rows=failures_data,
        )

        path = self._output_dir / "report.html"
        path.write_text(html, encoding="utf-8")
        return str(path.resolve())


# ============================================================================
# 内部 HTML 构建
# ============================================================================

def _read_csv_as_dicts(path: str) -> List[Dict[str, str]]:
    """读取 CSV 文件为字典列表。"""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _build_html_table(
    rows: List[Dict[str, str]],
    caption: str,
    table_id: str = "",
) -> str:
    """v2: 构建可排序 HTML 表格。"""
    if not rows:
        return f"<p>No data for {caption}.</p>"

    columns = list(rows[0].keys())
    header = "".join(
        f'<th onclick="sortTable(\'{table_id}\', {i})" style="cursor:pointer">&#9650;&#9660; {c}</th>'
        for i, c in enumerate(columns)
    )
    body_rows = ""
    for row in rows:
        cells = "".join(
            f"<td>{row.get(c, '')}</td>" for c in columns
        )
        body_rows += f"<tr>{cells}</tr>"

    return f"""
    <h3>{caption}</h3>
    <table id="{table_id}">
        <thead><tr>{header}</tr></thead>
        <tbody>{body_rows}</tbody>
    </table>
    """


def _build_html(
    benchmark_name: str,
    total: int,
    succeeded: int,
    failed: int,
    summary_rows: List[Dict[str, str]],
    leaderboard_rows: List[Dict[str, str]],
    failure_rows: List[Dict[str, str]],
) -> str:
    """构建完整 HTML 报告。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    summary_section = _build_html_table(summary_rows, "Summary", "summary-table")
    leaderboard_section = (
        _build_html_table(leaderboard_rows, "Leaderboard", "leaderboard-table")
        if leaderboard_rows
        else "<p>No leaderboard data.</p>"
    )
    failures_section = (
        _build_html_table(failure_rows, "Failures", "failures-table")
        if failure_rows
        else "<p>No failures.</p>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{benchmark_name} — Benchmark Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #212529; }}
        h1 {{ border-bottom: 2px solid #dee2e6; padding-bottom: 8px; }}
        h3 {{ margin-top: 24px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; background: white;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #dee2e6; font-size: 14px; }}
        th {{ background: #e9ecef; font-weight: 600; user-select: none; }}
        th:hover {{ background: #dee2e6; }}
        tr:hover {{ background: #f1f3f5; }}
        .summary-cards {{ display: flex; gap: 16px; margin: 16px 0; }}
        .card {{ background: white; border-radius: 8px; padding: 16px 24px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 120px; text-align: center; }}
        .card .value {{ font-size: 28px; font-weight: 700; }}
        .card .label {{ font-size: 12px; color: #6c757d; text-transform: uppercase; }}
        .success {{ color: #198754; }}
        .failure {{ color: #dc3545; }}
        .timestamp {{ color: #6c757d; font-size: 14px; }}
    </style>
    <script>
    function sortTable(tableId, colIdx) {{
        var table = document.getElementById(tableId);
        var tbody = table.tBodies[0];
        var rows = Array.from(tbody.rows);
        var ascending = table.getAttribute("data-sort-col") !== String(colIdx);
        table.setAttribute("data-sort-col", ascending ? colIdx : -1);
        rows.sort(function(a, b) {{
            var av = a.cells[colIdx].textContent.trim();
            var bv = b.cells[colIdx].textContent.trim();
            var an = parseFloat(av), bn = parseFloat(bv);
            if (!isNaN(an) && !isNaN(bn)) {{ av = an; bv = bn; }}
            if (av < bv) return ascending ? -1 : 1;
            if (av > bv) return ascending ? 1 : -1;
            return 0;
        }});
        rows.forEach(function(r) {{ tbody.appendChild(r); }});
    }}
    </script>
</head>
<body>
    <h1>{benchmark_name} — Benchmark Report</h1>
    <p class="timestamp">Generated: {now}</p>

    <div class="summary-cards">
        <div class="card">
            <div class="value">{total}</div>
            <div class="label">Total Runs</div>
        </div>
        <div class="card">
            <div class="value success">{succeeded}</div>
            <div class="label">Succeeded</div>
        </div>
        <div class="card">
            <div class="value failure">{failed}</div>
            <div class="label">Failed</div>
        </div>
    </div>

    {leaderboard_section}
    {summary_section}
    {failures_section}
</body>
</html>"""
