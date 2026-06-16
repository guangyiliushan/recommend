"""Single experiment orchestration — 原子执行单元。

职责：
- 接收一份 fully resolved experiment config
- 编排 config → data → model → execution → prediction → evaluation → artifact 全流程
- 返回一份结构化 ExperimentResult

与 core 的边界：
- 通过 MODEL_REGISTRY / DATASET_REGISTRY 发现组件
- 通过 BaseDataset.load()/get_split()/get_dataloader() 消费数据
- 通过 BaseRecommender.supports()/predict()/fit() 消费模型
- 通过 PredictionBundle → Evaluator 完成评估
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

from recsys.core.base_dataset import BaseDataset
from recsys.core.base_model import BaseRecommender, Capability
from recsys.core.prediction_bundle import PredictionBundle
from recsys.core.registry import DATASET_REGISTRY, MODEL_REGISTRY
from recsys.evaluation.evaluator import (
    EvaluationConfig,
    EvaluationResult,
    evaluate,
)


# ============================================================================
# 阶段与状态枚举
# ============================================================================

class ExperimentPhase(str, Enum):
    """实验阶段枚举。"""

    CONFIG = "config"
    BOOTSTRAP = "bootstrap"
    DATA = "data"
    MODEL = "model"
    TRAINING = "training"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"
    ARTIFACT = "artifact"


class ExperimentStatus(str, Enum):
    """实验状态枚举。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ExperimentError:
    """结构化实验错误。

    与 api-contracts.md 中的错误模型对齐。
    """

    code: str
    phase: ExperimentPhase
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    hint: Optional[str] = None
    traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。"""
        return {
            "code": self.code,
            "phase": self.phase.value,
            "message": self.message,
            "details": self.details,
            "hint": self.hint,
            "traceback": self.traceback,
        }


@dataclass
class ExperimentResult:
    """单次实验的结构化结果。

    与 api-contracts.md 和 artifacts.md 中对 run_experiment()
    返回值的约定对齐。
    """

    run_id: str
    status: ExperimentStatus
    summary_metrics: Dict[str, float] = field(default_factory=dict)
    task_metrics: Optional[Dict[str, Dict[str, float]]] = None
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    error: Optional[ExperimentError] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status == ExperimentStatus.SUCCEEDED


@dataclass
class ExperimentConfig:
    """归一化后的单实验配置。

    由 benchmark 展开或直接构造，是 experiment 的唯一定义输入。
    在当前阶段 config.py 未完成时，这里承担最小的归一化职责。
    """

    experiment_name: str
    dataset_name: str
    model_name: str
    seed: int

    output_dir: str = "./outputs/experiments"

    # 各子组件配置
    data_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    runtime_config: Dict[str, Any] = field(default_factory=dict)

    # 配置快照 hash（冻结时计算）
    config_hash: str = ""

    def freeze(self) -> "ExperimentConfig":
        """冻结配置：计算 hash 并返回 self。"""
        if not self.config_hash:
            self.config_hash = compute_config_hash(self)
        return self


# ============================================================================
# 辅助函数 —— run_id / hash
# ============================================================================

def generate_run_id(config: ExperimentConfig) -> str:
    """生成稳定、可恢复的 run_id。

    格式: {experiment}__{dataset}__{model}__seed{seed}__{short_hash}
    """
    if not config.config_hash:
        config.freeze()
    short_hash = config.config_hash[:8]
    return (
        f"{config.experiment_name}"
        f"__{config.dataset_name}"
        f"__{config.model_name}"
        f"__seed{config.seed}"
        f"__{short_hash}"
    )


def compute_config_hash(config: ExperimentConfig) -> str:
    """计算配置快照的 SHA-256 短 hash。

    只用稳定字段参与 hash 计算。
    """
    stable = {
        "experiment_name": config.experiment_name,
        "dataset_name": config.dataset_name,
        "model_name": config.model_name,
        "seed": config.seed,
        "data_config": config.data_config,
        "model_config": config.model_config,
        "training_config": config.training_config,
        "evaluation_config": config.evaluation_config,
        "runtime_config": config.runtime_config,
    }
    raw = json.dumps(stable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ============================================================================
# 辅助函数 —— status / artifact 文件写入
# ============================================================================

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status_file(
    run_dir: Path,
    status: ExperimentStatus,
    *,
    run_id: str = "",
    dataset: str = "",
    model: str = "",
    seed: int = 0,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    primary_metric: Optional[str] = None,
    primary_metric_value: Optional[float] = None,
    error: Optional[ExperimentError] = None,
    resume_supported: bool = False,
) -> None:
    """将实验状态写入 status.json。

    字段语义与 artifacts.md 中的 status.json 契约一致。
    """
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "status": status.value,
        "started_at": started_at,
        "finished_at": finished_at or _utc_now_iso(),
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "primary_metric": primary_metric,
        "primary_metric_value": primary_metric_value,
        "resume_supported": resume_supported,
    }
    if error is not None:
        payload["error"] = error.to_dict()

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_config_snapshot(run_dir: Path, config: ExperimentConfig) -> None:
    """将 fully resolved 配置快照写入 config.yaml。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_metrics_file(run_dir: Path, eval_result: EvaluationResult) -> None:
    """将评估结果写入 metrics.json。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(eval_result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ============================================================================
# 辅助函数 —— registry / runtime bootstrap
# ============================================================================

def bootstrap_registries() -> None:
    """显式触发模型与数据集注册副作用。

    model_registry.py 当前为骨架，仅触发已有的导入链。
    dataset_registry 已有完整的显式注册。
    """
    # 数据集注册 —— 已稳定
    import recsys.data.dataset_registry  # noqa: F401

    # 模型注册 —— 触发现有已注册模型（ItemCF 等）的导入副作用
    # 用户新增模型后，需在此处或 model_registry 中补注册导入
    import recsys.models.model_registry  # noqa: F401


def setup_runtime(runtime_config: Dict[str, Any]) -> None:
    """根据 runtime_config 设置 seed、device、日志级别等。"""
    import logging
    import random

    import numpy as np
    import torch

    seed: int = runtime_config.get("seed", 42)
    deterministic: bool = runtime_config.get("deterministic", False)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log_level = runtime_config.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ============================================================================
# 适配层 —— 从 dataset split 提取经典模型需要的交互数据
# ============================================================================

def extract_interactions_from_split(split_dataset: Any) -> List[Tuple[int, int]]:
    """从 dataset split 中提取 (user_id, item_id) 列表。

    适用于 ItemCF、MF 等需要用户-物品交互对的经典方法。
    遍历 split 的每个样本，收集 user_id 和 item_id。
    """
    pairs: List[Tuple[int, int]] = []
    for sample in split_dataset:
        uid = sample.get("user_id")
        iid = sample.get("item_id")
        if uid is not None and iid is not None:
            # 处理 tensor 或标量
            uid_val = uid.item() if hasattr(uid, "item") else int(uid)
            iid_val = iid.item() if hasattr(iid, "item") else int(iid)
            pairs.append((uid_val, iid_val))
    return pairs


def extract_user_item_mapping_from_split(
    split_dataset: Any,
) -> Dict[int, Set[int]]:
    """从 dataset split 中提取 user_id -> set(item_ids) 映射。

    适用于 ItemCF.predict() 的用户历史触发和 ground truth 构造。
    """
    mapping: Dict[int, Set[int]] = {}
    for sample in split_dataset:
        uid = sample.get("user_id")
        iid = sample.get("item_id")
        if uid is not None and iid is not None:
            uid_val = uid.item() if hasattr(uid, "item") else int(uid)
            iid_val = iid.item() if hasattr(iid, "item") else int(iid)
            mapping.setdefault(uid_val, set()).add(iid_val)
    return mapping


# ============================================================================
# 能力路由
# ============================================================================

def route_execution(
    model: BaseRecommender,
    dataset: BaseDataset,
    config: ExperimentConfig,
) -> PredictionBundle:
    """根据模型能力选择执行路径。

    - TRAINABLE → _execute_trainable_path
    - 否则     → _execute_nontrainable_path
    """
    if model.supports(Capability.TRAINABLE):
        return _execute_trainable_path(model, dataset, config)
    return _execute_nontrainable_path(model, dataset, config)


def _execute_trainable_path(
    model: BaseRecommender,
    dataset: BaseDataset,
    config: ExperimentConfig,
) -> PredictionBundle:
    """可训练模型路径（预留接口）。

    未来流程：
        1. 包装为 LightningRecommender
        2. 构造 Trainer(callbacks / loggers)
        3. fit(train_loader, val_loader)
        4. 产出 PredictionBundle
    """
    raise NotImplementedError(
        "trainable 路径尚未实现。"
        "请等待 trainer.py (LightningRecommender + TrainerFactory) 完成后接入。"
    )


def _execute_nontrainable_path(
    model: BaseRecommender,
    dataset: BaseDataset,
    config: ExperimentConfig,
) -> PredictionBundle:
    """非训练式模型路径。

    当前支持 interaction-group 适配：
    从 train split 提取交互对 → model.fit() →
    从 test split 提取映射 → model.predict() → PredictionBundle
    """
    train_split = dataset.get_split("train")
    test_split = dataset.get_split("test")

    # 1. fit
    train_pairs = extract_interactions_from_split(train_split)
    if not train_pairs:
        raise RuntimeError(
            "无法从 train split 提取 (user_id, item_id) 交互对。"
            "请确认数据集 schema 包含 user_id 和 item_id 字段。"
        )
    model.fit(train_pairs)

    # 2. predict
    train_mapping = extract_user_item_mapping_from_split(train_split)
    test_mapping = extract_user_item_mapping_from_split(test_split)

    return model.predict(
        user_train_items=train_mapping,
        user_test_items=test_mapping,
    )


# ============================================================================
# 主入口
# ============================================================================

def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """执行一次完整实验 —— 项目的原子执行单元。

    流程:
        1. 配置冻结
        2. registry bootstrap
        3. 输出目录初始化
        4. 状态文件初始化 (pending → running)
        5. dataset 构建
        6. model 构建
        7. 能力路由执行 → PredictionBundle
        8. evaluator 调用
        9. artifact 落盘
        10. 状态收尾
        11. 返回 ExperimentResult

    Parameters
    ----------
    config : ExperimentConfig
        fully resolved 实验配置。

    Returns
    -------
    ExperimentResult
        结构化实验结果。
    """
    # ---- 0. 配置冻结 ----
    config.freeze()
    run_id = generate_run_id(config)
    run_dir = Path(config.output_dir) / run_id
    started_at = _utc_now_iso()

    result = ExperimentResult(
        run_id=run_id,
        status=ExperimentStatus.RUNNING,
        metadata={
            "dataset_name": config.dataset_name,
            "model_name": config.model_name,
            "seed": config.seed,
            "started_at": started_at,
        },
    )

    # ---- 1. status = pending ----
    write_status_file(
        run_dir,
        ExperimentStatus.PENDING,
        run_id=run_id,
        dataset=config.dataset_name,
        model=config.model_name,
        seed=config.seed,
        started_at=started_at,
    )

    # 日志目录
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 阶段执行包装器
    def _run_phase(
        phase: ExperimentPhase,
        fn: Callable[[], Any],
    ) -> Any:
        try:
            return fn()
        except Exception as exc:
            err = ExperimentError(
                code=_phase_to_error_code(phase),
                phase=phase,
                message=str(exc),
                hint=None,
                traceback=traceback.format_exc(),
            )
            # 写失败状态
            write_status_file(
                run_dir,
                ExperimentStatus.FAILED,
                run_id=run_id,
                dataset=config.dataset_name,
                model=config.model_name,
                seed=config.seed,
                started_at=started_at,
                error=err,
            )
            # 记日志
            (logs_dir / "stderr.log").write_text(
                err.traceback or "", encoding="utf-8"
            )
            result.status = ExperimentStatus.FAILED
            result.error = err
            raise  # 重新抛出，由最外层统一捕获

    try:
        # ---- 2. registry bootstrap ----
        _run_phase(ExperimentPhase.BOOTSTRAP, bootstrap_registries)

        # ---- 3. runtime 初始化 ----
        setup_runtime({**config.runtime_config, "seed": config.seed})

        # ---- 4. 写 config 快照 ----
        write_config_snapshot(run_dir, config)

        # ---- 5. status = running ----
        write_status_file(
            run_dir,
            ExperimentStatus.RUNNING,
            run_id=run_id,
            dataset=config.dataset_name,
            model=config.model_name,
            seed=config.seed,
            started_at=started_at,
        )

        # ---- 6. dataset 构建 ----
        def _build_dataset() -> BaseDataset:
            ds_cls = DATASET_REGISTRY.get(config.dataset_name)
            ds = ds_cls(
                root_dir=config.data_config.get("root_dir", "./data"),
                split_ratios=tuple(
                    config.data_config.get("split_ratios", (0.8, 0.1, 0.1))
                ),
                max_seq_len=config.data_config.get("max_seq_len", 50),
                min_seq_len=config.data_config.get("min_seq_len", 2),
                neg_sample_count=config.data_config.get("neg_sample_count", 4),
                **config.data_config.get("extra", {}),
            )
            ds.load()
            return ds

        dataset = _run_phase(ExperimentPhase.DATA, _build_dataset)

        # ---- 7. model 构建 ----
        def _build_model() -> BaseRecommender:
            model_cls = MODEL_REGISTRY.get(config.model_name)
            model = model_cls(
                config=config.model_config.get("params", {}),
                schema_metadata={
                    "num_users": dataset.num_users,
                    "num_items": dataset.num_items,
                    "feature_cols": dataset.feature_cols,
                    **config.model_config.get("schema_extra", {}),
                },
            )
            return model

        model = _run_phase(ExperimentPhase.MODEL, _build_model)

        # ---- 8. training / prediction ----
        bundle = _run_phase(
            ExperimentPhase.TRAINING,
            lambda: route_execution(model, dataset, config),
        )

        # ---- 9. evaluation ----
        def _run_evaluation() -> EvaluationResult:
            eval_cfg = EvaluationConfig(
                metrics=config.evaluation_config.get("metrics"),
                primary_metric=config.evaluation_config.get("primary_metric"),
                threshold=config.evaluation_config.get("threshold", 0.5),
                threshold_strategy=config.evaluation_config.get(
                    "threshold_strategy", "fixed"
                ),
                ranking_k=config.evaluation_config.get(
                    "ranking_k", [5, 10, 20]
                ),
                generate_curves=config.evaluation_config.get(
                    "generate_curves", True
                ),
                curve_types=config.evaluation_config.get(
                    "curve_types", ["roc", "pr"]
                ),
                average=config.evaluation_config.get("average", "binary"),
                pos_label=config.evaluation_config.get("pos_label", 1),
                sample_weight_enabled=config.evaluation_config.get(
                    "sample_weight_enabled", False
                ),
                statistical_test=config.evaluation_config.get("statistical_test"),
                per_group_metrics=config.evaluation_config.get(
                    "per_group_metrics", True
                ),
                save_predictions=config.evaluation_config.get(
                    "save_predictions", False
                ),
                save_curves=config.evaluation_config.get("save_curves", True),
                report_aliases=config.evaluation_config.get("report_aliases", {}),
            )
            return evaluate(bundle, eval_cfg)

        eval_result = _run_phase(ExperimentPhase.EVALUATION, _run_evaluation)

        # ---- 10. artifact 落盘 ----
        def _write_artifacts() -> None:
            write_metrics_file(run_dir, eval_result)
            # 预测明细后续通过 predictions.parquet 支持

        _run_phase(ExperimentPhase.ARTIFACT, _write_artifacts)

        # ---- 11. 成功收尾 ----
        primary_metric = config.evaluation_config.get(
            "primary_metric",
            eval_result.metadata.get("primary_metric"),
        )
        primary_value: Optional[float] = None
        if primary_metric and primary_metric in eval_result.summary_metrics:
            primary_value = eval_result.summary_metrics[primary_metric]

        write_status_file(
            run_dir,
            ExperimentStatus.SUCCEEDED,
            run_id=run_id,
            dataset=config.dataset_name,
            model=config.model_name,
            seed=config.seed,
            started_at=started_at,
            primary_metric=primary_metric,
            primary_metric_value=primary_value,
        )

        result.status = ExperimentStatus.SUCCEEDED
        result.summary_metrics = eval_result.summary_metrics
        result.task_metrics = eval_result.task_metrics
        result.artifact_paths = {
            "config": str(run_dir / "config.yaml"),
            "metrics": str(run_dir / "metrics.json"),
            "status": str(run_dir / "status.json"),
            "run_dir": str(run_dir),
        }
        result.warnings = eval_result.warnings
        result.metadata.update({
            "finished_at": _utc_now_iso(),
            "primary_metric": primary_metric,
            "primary_metric_value": primary_value,
            "num_samples": bundle.num_samples,
        })

        return result

    except Exception:
        # 异常已在 _run_phase 中记录到 status.json 并填充 result.error
        return result


# ============================================================================
# 内部 —— 错误码映射
# ============================================================================

_PHASE_ERROR_CODES: Dict[ExperimentPhase, str] = {
    ExperimentPhase.CONFIG: "CONFIG_VALIDATION_ERROR",
    ExperimentPhase.BOOTSTRAP: "REGISTRY_ITEM_NOT_FOUND",
    ExperimentPhase.DATA: "DATASET_NOT_LOADED",
    ExperimentPhase.MODEL: "MODEL_CONTRACT_ERROR",
    ExperimentPhase.TRAINING: "MODEL_CONTRACT_ERROR",
    ExperimentPhase.PREDICTION: "EVALUATION_CONTRACT_ERROR",
    ExperimentPhase.EVALUATION: "EVALUATION_CONTRACT_ERROR",
    ExperimentPhase.ARTIFACT: "ARTIFACT_WRITE_ERROR",
}


def _phase_to_error_code(phase: ExperimentPhase) -> str:
    """阶段到统一错误码的映射。"""
    return _PHASE_ERROR_CODES.get(phase, "UNKNOWN_ERROR")
