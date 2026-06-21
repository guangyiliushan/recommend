"""Benchmark runner — 矩阵展开、调度与恢复。

v2 新增：
- 受控并发：max_concurrent_runs 控制并行度（默认 1，串行）
- tqdm 进度条
- 每个 run 写入独立的 run.log

职责：
- 将 benchmark 矩阵配置展开为多个 fully resolved experiment config
- 按恢复策略生成 RunPlan，决定各 run 的跳过/执行/重试
- 串行或受控并发调用 run_experiment()，失败隔离
- 收集结果后委托 Reporter 生成聚合产物

边界：
- 不直接接触 core 模块，只通过 experiment 间接消费
- 不实现训练、评估、指标逻辑
- 并发粒度在 experiment 级别（每个 run 一个执行单元）
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from recsys.pipeline.experiment import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    generate_run_id,
    run_experiment,
)
from recsys.pipeline.reporter import (
    Reporter,
    ReporterConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 枚举与数据结构
# ============================================================================

class ResumeMode(str, Enum):
    """恢复模式枚举。

    与 benchmarking.md 中的恢复语义对齐。
    """

    SUCCESSFUL_SKIP = "successful_skip"
    """跳过已成功的 run（默认）。"""

    FAILED_ONLY = "failed_only"
    """只重试失败的 run。"""

    UNFINISHED_ONLY = "unfinished_only"
    """继续未完成的 run（pending / running）。"""

    FORCE = "force"
    """强制重跑所有 run。"""


@dataclass
class BenchmarkConfig:
    """批量 benchmark 配置。

    与 configuration.md 和 benchmarking.md 中定义的矩阵式配置对齐：
    只描述"要跑哪些组合"，不复制整套训练细节。
    """

    benchmark_name: str

    # 实验矩阵
    models: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    seeds: List[int] = field(default_factory=lambda: [42])

    # preset 引用（当前阶段均为可选，后续通过 Hydra 解析）
    experiment_preset: str = "default"
    training_preset: Optional[str] = None
    evaluation_preset: Optional[str] = None
    runtime_preset: Optional[str] = None

    # 恢复策略
    resume_mode: ResumeMode = ResumeMode.SUCCESSFUL_SKIP

    # v2: 并发控制（默认 1 = 串行）
    max_concurrent_runs: int = 1

    # 输出
    output_root: str = "./outputs"
    experiment_output_dir: str = "./outputs/experiments"


@dataclass
class BenchmarkResult:
    """批量 benchmark 的结构化结果。

    与 benchmarking.md 和 artifacts.md 中约定的返回结构对齐。
    """

    benchmark_name: str
    status: str  # succeeded / partial_success / failed
    runs: List[ExperimentResult] = field(default_factory=list)
    summary_path: str = ""
    leaderboard_path: Optional[str] = None
    failures_path: str = ""
    manifest_path: str = ""
    report_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded_runs(self) -> List[ExperimentResult]:
        return [r for r in self.runs if r.succeeded]

    @property
    def failed_runs(self) -> List[ExperimentResult]:
        return [r for r in self.runs if r.status == ExperimentStatus.FAILED]


@dataclass
class RunPlan:
    """单个 run 的执行计划。

    benchmark 在真正执行前先为每个组合生成 plan，
    确定其 run_id、config 以及是否应该跳过。
    """

    run_id: str
    config: ExperimentConfig
    should_skip: bool = False
    skip_reason: Optional[str] = None


# ============================================================================
# 矩阵展开
# ============================================================================

def expand_benchmark_config(
    bench_cfg: BenchmarkConfig,
) -> List[ExperimentConfig]:
    """将 benchmark 矩阵配置展开为单实验配置列表。

    展开维度: models × datasets × seeds。
    每个组合生成一份独立的 ExperimentConfig。
    """
    configs: List[ExperimentConfig] = []

    for model_name in bench_cfg.models:
        for dataset_name in bench_cfg.datasets:
            for seed in bench_cfg.seeds:
                cfg = ExperimentConfig(
                    experiment_name=bench_cfg.benchmark_name,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed=seed,
                    output_dir=bench_cfg.experiment_output_dir,
                    # preset 信息写入 runtime，后续 config.py 完成后统一解析
                    runtime_config={
                        "experiment_preset": bench_cfg.experiment_preset,
                        "training_preset": bench_cfg.training_preset,
                        "evaluation_preset": bench_cfg.evaluation_preset,
                        "runtime_preset": bench_cfg.runtime_preset,
                    },
                )
                configs.append(cfg)

    return configs


# ============================================================================
# 恢复与跳过判断
# ============================================================================

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_run_completion(
    run_dir: Path,
    config_hash: str,
) -> tuple[bool, Optional[str]]:
    """检查一个 run 是否已完成。

    检查条件（与 artifacts.md 中的恢复逻辑一致）：
    1. status.json 存在且 status = succeeded
    2. metrics.json 存在
    3. config.yaml 存在且 hash 匹配

    Returns
    -------
    (is_complete, reason)
    """
    status_file = run_dir / "status.json"
    metrics_file = run_dir / "metrics.json"
    config_file = run_dir / "config.yaml"

    if not status_file.exists():
        return False, "status.json not found"

    if not metrics_file.exists():
        return False, "metrics.json not found"

    try:
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, "status.json is corrupted"

    if status_data.get("status") != "succeeded":
        return False, f"status is '{status_data.get('status')}'"

    # 可选：检查 config hash 一致性
    if config_file.exists():
        # config yaml 中会包含 config_hash 字段
        try:
            import yaml
            config_data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            stored_hash = config_data.get("config_hash", "") if isinstance(config_data, dict) else ""
            if stored_hash and stored_hash != config_hash:
                return False, "config hash mismatch — configuration has changed"
        except Exception:
            pass  # 解析失败不阻塞，依赖 status 判断

    return True, None


def plan_runs(
    configs: List[ExperimentConfig],
    resume_mode: ResumeMode,
    output_root: str,
) -> List[RunPlan]:
    """为每个实验配置生成执行计划。

    根据 resume_mode 决定是否跳过已有的成功 run。
    """
    plans: List[RunPlan] = []

    for cfg in configs:
        cfg.freeze()
        run_id = generate_run_id(cfg)
        run_dir = Path(output_root) / "experiments" / run_id

        should_skip = False
        skip_reason: Optional[str] = None

        if resume_mode != ResumeMode.FORCE:
            is_complete, reason = check_run_completion(run_dir, cfg.config_hash)

            if resume_mode == ResumeMode.SUCCESSFUL_SKIP and is_complete:
                should_skip = True
                skip_reason = reason

            elif resume_mode == ResumeMode.FAILED_ONLY:
                # 只重试失败的；成功的、未开始的都跳过
                status_file = run_dir / "status.json"
                if status_file.exists():
                    try:
                        data = json.loads(status_file.read_text(encoding="utf-8"))
                        st = data.get("status", "")
                        if st == "succeeded":
                            should_skip = True
                            skip_reason = "already succeeded"
                        elif st not in ("failed",):
                            should_skip = True
                            skip_reason = f"status is '{st}', not failed"
                    except Exception:
                        pass
                else:
                    should_skip = True
                    skip_reason = "no status file — nothing to retry"

            elif resume_mode == ResumeMode.UNFINISHED_ONLY:
                # 只继续未完成的；成功的、失败的都跳过
                if is_complete:
                    should_skip = True
                    skip_reason = reason
                else:
                    status_file = run_dir / "status.json"
                    if status_file.exists():
                        try:
                            data = json.loads(status_file.read_text(encoding="utf-8"))
                            st = data.get("status", "")
                            if st == "failed":
                                should_skip = True
                                skip_reason = "already failed"
                        except Exception:
                            pass

        plans.append(RunPlan(
            run_id=run_id,
            config=cfg,
            should_skip=should_skip,
            skip_reason=skip_reason,
        ))

    return plans


def collect_skipped_result(run_id: str, config: ExperimentConfig) -> ExperimentResult:
    """从已有 artifact 重建跳过 run 的 ExperimentResult。"""
    run_dir = Path(config.output_dir) / run_id

    result = ExperimentResult(
        run_id=run_id,
        status=ExperimentStatus.SKIPPED,
        metadata={
            "dataset_name": config.dataset_name,
            "model_name": config.model_name,
            "seed": config.seed,
        },
    )

    # 尝试从 status.json 重建
    status_file = run_dir / "status.json"
    if status_file.exists():
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
            result.metadata["original_status"] = data.get("status")
        except Exception:
            pass

    # 尝试从 metrics.json 重建指标
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        try:
            mdata = json.loads(metrics_file.read_text(encoding="utf-8"))
            result.summary_metrics = mdata.get("summary_metrics", {})
            result.task_metrics = mdata.get("task_metrics")
        except Exception:
            pass

    return result


# ============================================================================
# 执行循环
# ============================================================================

def _try_import_tqdm():
    """尝试导入 tqdm，不可用时返回空包装器。"""
    try:
        from tqdm import tqdm as _tqdm
        return _tqdm
    except ImportError:
        return lambda it, **kw: it


def _execute_one(plan: RunPlan, idx: int, total: int) -> ExperimentResult:
    """执行单个 run plan（供并发调度器消费）。"""
    if plan.should_skip:
        return collect_skipped_result(plan.run_id, plan.config)

    try:
        result = run_experiment(plan.config)
    except Exception:
        from recsys.pipeline.experiment import ExperimentError as _ExpErr
        from recsys.pipeline.experiment import ExperimentPhase as _ExpPhase
        result = ExperimentResult(
            run_id=plan.run_id,
            status=ExperimentStatus.FAILED,
            error=_ExpErr(
                code="BENCHMARK_EXECUTION_ERROR",
                phase=_ExpPhase.TRAINING,
                message="run_experiment raised unhandled exception",
            ),
            metadata={
                "dataset_name": plan.config.dataset_name,
                "model_name": plan.config.model_name,
                "seed": plan.config.seed,
            },
        )

    return result


def execute_runs_parallel(
    plans: List[RunPlan],
    max_workers: int,
) -> List[ExperimentResult]:
    """v2: 受控并发执行所有 run plans。

    使用 ThreadPoolExecutor，每个 run 一个 future。
    失败隔离：单个 run 失败不阻塞其他。
    """
    results: List[ExperimentResult] = []
    total = len(plans)
    tqdm = _try_import_tqdm()

    # 并发执行时静默 ItemCF 等模型的内部进度条
    os.environ["RECSYS_BENCHMARK_MODE"] = "1"
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_execute_one, plan, i, total): (i, plan)
                for i, plan in enumerate(plans)
            }

            with tqdm(total=total, desc="Benchmark", unit="run") as pbar:
                for future in as_completed(futures):
                    i, plan = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        result = ExperimentResult(
                            run_id=plan.run_id,
                            status=ExperimentStatus.FAILED,
                            metadata={
                                "dataset_name": plan.config.dataset_name,
                                "model_name": plan.config.model_name,
                                "seed": plan.config.seed,
                            },
                        )
                    results.append((i, result))
                    pbar.update(1)
    finally:
        os.environ.pop("RECSYS_BENCHMARK_MODE", None)

    # 按原始顺序排序
    results.sort(key=lambda x: x[0])
    return [r for _, r in results]


def execute_runs(
    plans: List[RunPlan],
) -> List[ExperimentResult]:
    """串行执行所有 run plans（v1 兼容，保留）。

    每个 plan：
    - 若 should_skip，从已有 artifact 重建结果
    - 否则调用 run_experiment()
    - 失败时记录结构化错误，继续下一个
    """
    results: List[ExperimentResult] = []
    total = len(plans)
    tqdm = _try_import_tqdm()

    for i, plan in enumerate(tqdm(plans, desc="Benchmark", unit="run")):
        result = _execute_one(plan, i, total)
        results.append(result)

    return results


# ============================================================================
# 主入口
# ============================================================================

def run_benchmark(bench_cfg: BenchmarkConfig) -> BenchmarkResult:
    """执行一次批量 benchmark。

    流程:
        1. 展开矩阵配置
        2. 生成 run plans（含恢复判断）
        3. 初始化 benchmark 输出目录
        4. 串行执行所有 run
        5. 委托 Reporter 生成聚合产物
        6. 返回 BenchmarkResult

    Parameters
    ----------
    bench_cfg : BenchmarkConfig
        benchmark 矩阵配置。

    Returns
    -------
    BenchmarkResult
        结构化 benchmark 结果。
    """
    # ---- 1. 展开矩阵 ----
    experiment_configs = expand_benchmark_config(bench_cfg)

    # ---- 2. 生成 plans ----
    plans = plan_runs(
        experiment_configs,
        bench_cfg.resume_mode,
        bench_cfg.output_root,
    )

    # ---- 3. 输出目录 ----
    bench_dir = Path(bench_cfg.output_root) / "benchmarks" / bench_cfg.benchmark_name
    bench_dir.mkdir(parents=True, exist_ok=True)

    # ---- 4. 执行 ----
    if bench_cfg.max_concurrent_runs > 1:
        results = execute_runs_parallel(plans, bench_cfg.max_concurrent_runs)
    else:
        results = execute_runs(plans)

    # ---- 5. 汇总状态 ----
    succeeded = sum(1 for r in results if r.status in (
        ExperimentStatus.SUCCEEDED, ExperimentStatus.SKIPPED
    ))
    failed = sum(1 for r in results if r.status == ExperimentStatus.FAILED)
    total = len(results)

    if failed == 0:
        overall_status = "succeeded"
    elif succeeded == 0:
        overall_status = "failed"
    else:
        overall_status = "partial_success"

    # ---- 6. 委托 Reporter 生成聚合产物 ----
    reporter = Reporter(
        ReporterConfig(
            benchmark_name=bench_cfg.benchmark_name,
            output_dir=str(bench_dir),
            generate_html=True,
            generate_leaderboard=True,
            generate_failures=True,
        )
    )
    artifact_paths = reporter.generate(results)

    # ---- 7. 写 manifest.json ----
    manifest = {
        "benchmark_name": bench_cfg.benchmark_name,
        "created_at": _utc_now_iso(),
        "runs": [r.run_id for r in results],
        "models": bench_cfg.models,
        "datasets": bench_cfg.datasets,
        "seeds": bench_cfg.seeds,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "summary_path": artifact_paths.get("summary_csv", ""),
        "failures_path": artifact_paths.get("failures_csv", ""),
        "leaderboard_path": artifact_paths.get("leaderboard_csv"),
        "report_path": artifact_paths.get("report_html"),
    }
    manifest_file = bench_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ---- 8. 返回 ----
    return BenchmarkResult(
        benchmark_name=bench_cfg.benchmark_name,
        status=overall_status,
        runs=results,
        summary_path=artifact_paths.get("summary_csv", ""),
        leaderboard_path=artifact_paths.get("leaderboard_csv"),
        failures_path=artifact_paths.get("failures_csv", ""),
        manifest_path=str(manifest_file),
        report_path=artifact_paths.get("report_html"),
        metadata={
            "total_runs": total,
            "succeeded": succeeded,
            "failed": failed,
            "resume_mode": bench_cfg.resume_mode.value,
            "created_at": _utc_now_iso(),
        },
    )
