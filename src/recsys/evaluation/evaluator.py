"""Evaluator — 评估层编排器。

职责定位：
- 评估层编排器，不做底层公式，不画图，不绑定训练框架
- 校验 PredictionBundle → 根据 task_type/problem_type 分流 → 调用 metrics/ranking → 组装结构化输出

主入口：
- evaluate(bundle, config) -> EvaluationResult
- evaluate_pointwise(bundle, config)
- evaluate_ranking(bundle, config)
- evaluate_multitask(bundle, config)

输出结构：
- summary_metrics / task_metrics / group_metrics / curve_artifacts / metadata / warnings / errors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from recsys.core.prediction_bundle import PredictionBundle
from recsys.evaluation.metrics import (
    ClassificationMetricsResult,
    evaluate_binary_classification,
    evaluate_multiclass_classification,
)
from recsys.evaluation.ranking import (
    RankingMetricsResult,
    evaluate_ranking_from_bundle,
)

# ---------------------------------------------------------------------------
# 配置结构
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """评估配置。

    与 configuration.md 中要求的 evaluation 配置段对齐。

    Attributes
    ----------
    metrics : List[str]
        要计算的指标列表。
    primary_metric : str
        主指标（benchmark 使用）。
    threshold : float
        固定分类阈值。
    threshold_strategy : str
        阈值策略：fixed / bundle / tuned_on_val / per_task。
    ranking_k : List[int]
        ranking 评估的 K 值列表。
    generate_curves : bool
        是否生成曲线数据。
    curve_types : List[str]
        要生成的曲线类型，如 ["roc", "pr"]。
    average : str
        多分类平均方式：binary / macro / micro / weighted。
    pos_label : int
        正类标签值。
    sample_weight_enabled : bool
        是否启用样本权重。
    statistical_test : Optional[str]
        统计检验方法。
    per_group_metrics : bool
        是否计算分组指标。
    save_predictions : bool
        是否保存预测结果。
    save_curves : bool
        是否落盘曲线数据。
    report_aliases : Dict[str, str]
        报表别名映射。
    """

    # 指标配置
    metrics: Optional[List[str]] = None  # None = 使用默认指标
    primary_metric: Optional[str] = None

    # 阈值配置
    threshold: float = 0.5
    threshold_strategy: str = "fixed"  # fixed / bundle / tuned_on_val / per_task

    # ranking 配置
    ranking_k: List[int] = field(default_factory=lambda: [5, 10, 20])

    # 曲线配置
    generate_curves: bool = True
    curve_types: List[str] = field(default_factory=lambda: ["roc", "pr"])

    # 多分类配置
    average: str = "binary"  # binary / macro / micro / weighted
    pos_label: int = 1

    # 权重配置
    sample_weight_enabled: bool = False

    # 统计检验
    statistical_test: Optional[str] = None

    # 输出控制
    per_group_metrics: bool = True
    save_predictions: bool = False
    save_curves: bool = True

    # 别名映射
    report_aliases: Dict[str, str] = field(default_factory=dict)

    def get_primary_metric(self, task_type: str, problem_type: str) -> str:
        """根据任务类型获取主指标。

        Parameters
        ----------
        task_type : str
            任务类型。
        problem_type : str
            问题类型。

        Returns
        -------
        str
            主指标名称。
        """
        if self.primary_metric:
            return self.primary_metric

        # 默认主指标选择逻辑
        if task_type == "ranking":
            k = self.ranking_k[0] if self.ranking_k else 10
            return f"ndcg@{k}"

        if task_type == "pointwise" or task_type == "multitask":
            _problem_metrics = {
                "binary": "pr_auc",
                "multiclass": "accuracy",
                "multilabel": "f1_macro",
            }
            return _problem_metrics.get(problem_type, "roc_auc")

        return "roc_auc"

    def get_threshold(self, bundle: PredictionBundle) -> float:
        """根据策略获取阈值。

        Parameters
        ----------
        bundle : PredictionBundle
            预测产物。

        Returns
        -------
        float
            决策阈值。
        """
        if self.threshold_strategy == "bundle":
            if bundle.thresholds:
                return bundle.thresholds[0]
            return self.threshold

        if self.threshold_strategy == "per_task":
            # 多任务场景下由调用方传入
            return self.threshold

        # fixed / tuned_on_val
        return self.threshold


# ---------------------------------------------------------------------------
# 输出结构
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """评估结果。

    与 evaluation.md 和 artifacts.md 中要求的输出契约对齐。

    Attributes
    ----------
    summary_metrics : Dict[str, float]
        主结果指标（benchmark 消费）。
    task_metrics : Dict[str, Dict[str, float]]
        分任务/分头指标。
    group_metrics : Dict[str, Any]
        分段诊断结果。
    curve_artifacts : Dict[str, Any]
        曲线数据（内存或路径引用）。
    metadata : Dict[str, Any]
        元信息。
    warnings : List[str]
        警告信息。
    errors : List[str]
        错误信息。
    """

    summary_metrics: Dict[str, float] = field(default_factory=dict)
    task_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    group_metrics: Dict[str, Any] = field(default_factory=dict)
    curve_artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。"""
        # 将 curve_artifacts 中的 numpy 数组转为列表
        serializable_curves = {}
        for key, value in self.curve_artifacts.items():
            if isinstance(value, dict):
                serializable_curves[key] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in value.items()
                }
            elif isinstance(value, np.ndarray):
                serializable_curves[key] = value.tolist()
            else:
                serializable_curves[key] = value

        return {
            "summary_metrics": self.summary_metrics,
            "task_metrics": self.task_metrics,
            "group_metrics": self.group_metrics,
            "curve_artifacts": serializable_curves,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def evaluate(
    bundle: PredictionBundle,
    config: Optional[EvaluationConfig] = None,
) -> EvaluationResult:
    """评估主入口。

    根据 PredictionBundle 的 task_type / problem_type 自动选择评估路径。

    Parameters
    ----------
    bundle : PredictionBundle
        预测产物。
    config : EvaluationConfig, optional
        评估配置。

    Returns
    -------
    EvaluationResult
        结构化评估结果。
    """
    if config is None:
        config = EvaluationConfig()

    # 校验 bundle
    errors = bundle.validate()
    if errors:
        result = EvaluationResult(errors=errors)
        result.warnings.append("bundle_validation_failed")
        return result

    # 按 task_type 分流
    task_type = bundle.task_type
    problem_type = bundle.problem_type

    if task_type == "pointwise":
        return evaluate_pointwise(bundle, config)

    elif task_type == "ranking":
        return evaluate_ranking(bundle, config)

    elif task_type == "multitask":
        return evaluate_multitask(bundle, config)

    else:
        return EvaluationResult(
            errors=[f"unsupported task_type: {task_type}"],
            metadata={"task_type": task_type, "problem_type": problem_type},
        )


def evaluate_pointwise(
    bundle: PredictionBundle,
    config: Optional[EvaluationConfig] = None,
) -> EvaluationResult:
    """评估 pointwise 任务。

    Parameters
    ----------
    bundle : PredictionBundle
        预测产物（task_type="pointwise"）。
    config : EvaluationConfig, optional
        评估配置。

    Returns
    -------
    EvaluationResult
        结构化评估结果。
    """
    if config is None:
        config = EvaluationConfig()

    result = EvaluationResult()
    warnings: List[str] = []
    errors: List[str] = []

    # 获取阈值
    threshold = config.get_threshold(bundle)
    if config.threshold_strategy == "fixed":
        # 记录阈值来源
        result.metadata["threshold_source"] = "config"
        result.metadata["threshold_strategy"] = "fixed"
    elif config.threshold_strategy == "bundle":
        result.metadata["threshold_source"] = "bundle"
        result.metadata["threshold_strategy"] = "bundle"

    problem_type = bundle.problem_type

    # 获取样本权重
    sample_weight = None
    if config.sample_weight_enabled and bundle.sample_weight:
        sample_weight = np.asarray(bundle.sample_weight)

    try:
        if problem_type == "binary":
            metrics_result: ClassificationMetricsResult = evaluate_binary_classification(
                y_true=np.asarray(bundle.y_true),
                y_score=np.asarray(bundle.y_score),
                y_pred=np.asarray(bundle.y_pred) if bundle.y_pred else None,
                threshold=threshold,
                pos_label=bundle.pos_label if bundle.pos_label else 1,
                sample_weight=sample_weight,
                metrics=config.metrics,
                generate_curves=config.generate_curves,
            )

            # 汇总指标
            result.summary_metrics = metrics_result.metrics

            # 曲线数据
            if config.generate_curves and metrics_result.roc_curve and "roc" in config.curve_types:
                result.curve_artifacts["roc_curve"] = metrics_result.roc_curve
            if config.generate_curves and metrics_result.pr_curve and "pr" in config.curve_types:
                result.curve_artifacts["pr_curve"] = metrics_result.pr_curve

            # 混淆矩阵
            if metrics_result.confusion_matrix:
                result.metadata["confusion_matrix"] = {
                    "tp": metrics_result.confusion_matrix.tp,
                    "fp": metrics_result.confusion_matrix.fp,
                    "tn": metrics_result.confusion_matrix.tn,
                    "fn": metrics_result.confusion_matrix.fn,
                    "support": metrics_result.confusion_matrix.support,
                    "prevalence": metrics_result.confusion_matrix.prevalence,
                }

            warnings.extend(metrics_result.warnings)

        elif problem_type == "multiclass":
            metrics_result = evaluate_multiclass_classification(
                y_true=np.asarray(bundle.y_true),
                y_score=np.asarray(bundle.y_score),
                y_pred=np.asarray(bundle.y_pred) if bundle.y_pred else None,
                labels=bundle.class_names,
                average=config.average,  # type: ignore
                sample_weight=sample_weight,
                metrics=config.metrics,
            )

            result.summary_metrics = metrics_result.metrics
            warnings.extend(metrics_result.warnings)

        elif problem_type == "multilabel":
            # 多标签评估：逐标签计算
            y_true = np.asarray(bundle.y_true)
            y_score = np.asarray(bundle.y_score)

            if y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            if y_score.ndim == 1:
                y_score = y_score.reshape(-1, 1)

            num_labels = y_true.shape[1]
            label_metrics: Dict[str, Dict[str, float]] = {}
            label_names = bundle.class_names or [f"label_{i}" for i in range(num_labels)]

            for i in range(num_labels):
                try:
                    label_result = evaluate_binary_classification(
                        y_true=y_true[:, i],
                        y_score=y_score[:, i],
                        threshold=threshold,
                        pos_label=1,
                        sample_weight=sample_weight,
                        metrics=config.metrics,
                        generate_curves=False,
                    )
                    label_metrics[label_names[i]] = label_result.metrics
                except Exception as e:
                    warnings.append(f"label_{i}_evaluation_failed: {str(e)}")

            # 跨标签聚合
            if label_metrics:
                result.task_metrics = label_metrics

                # 计算 macro 平均
                all_keys = set()
                for m in label_metrics.values():
                    all_keys.update(m.keys())

                for key in all_keys:
                    values = [m[key] for m in label_metrics.values() if key in m]
                    if values:
                        result.summary_metrics[f"{key}_macro"] = float(np.mean(values))

        else:
            errors.append(f"unsupported problem_type for pointwise: {problem_type}")

    except Exception as e:
        errors.append(f"pointwise_evaluation_failed: {str(e)}")

    # 元信息
    result.metadata.update({
        "task_type": "pointwise",
        "problem_type": problem_type,
        "threshold": threshold,
        "pos_label": config.pos_label,
        "num_samples": bundle.num_samples,
        "primary_metric": config.get_primary_metric("pointwise", problem_type),
        "schema_version": "1.0.0",
    })

    # 注入 bundle metadata
    if bundle.metadata:
        result.metadata["model_name"] = bundle.metadata.get("model_name", "unknown")
        result.metadata["dataset_name"] = bundle.metadata.get("dataset_name", "unknown")

    result.warnings = warnings
    result.errors = errors

    return result


def evaluate_ranking(
    bundle: PredictionBundle,
    config: Optional[EvaluationConfig] = None,
) -> EvaluationResult:
    """评估 ranking 任务。

    Parameters
    ----------
    bundle : PredictionBundle
        预测产物（task_type="ranking"）。
    config : EvaluationConfig, optional
        评估配置。

    Returns
    -------
    EvaluationResult
        结构化评估结果。
    """
    if config is None:
        config = EvaluationConfig()

    result = EvaluationResult()
    warnings: List[str] = []
    errors: List[str] = []

    try:
        ranking_result: RankingMetricsResult = evaluate_ranking_from_bundle(
            bundle,
            k_list=config.ranking_k,
            metrics=config.metrics,
        )

        # 汇总指标
        result.summary_metrics = ranking_result.summary_metrics

        # 分组指标
        if config.per_group_metrics and ranking_result.group_metrics:
            result.group_metrics = {
                "group_results": [
                    {
                        "group_id": str(gr.group_id),
                        "metrics": gr.metrics,
                        "num_candidates": gr.num_candidates,
                        "num_relevant": gr.num_relevant,
                    }
                    for gr in ranking_result.group_metrics
                ]
            }

        # 元信息
        result.metadata = {
            "task_type": "ranking",
            "problem_type": bundle.problem_type,
            "k_list": config.ranking_k,
            "num_groups": ranking_result.metadata.get("num_groups", 0),
            "num_valid_groups": ranking_result.metadata.get("num_valid_groups", 0),
            "num_skipped_groups": ranking_result.metadata.get("num_skipped_groups", 0),
            "primary_metric": config.get_primary_metric("ranking", bundle.problem_type),
            "schema_version": "1.0.0",
        }

        warnings.extend(ranking_result.warnings)

    except Exception as e:
        errors.append(f"ranking_evaluation_failed: {str(e)}")

    # 注入 bundle metadata
    if bundle.metadata:
        result.metadata["model_name"] = bundle.metadata.get("model_name", "unknown")
        result.metadata["dataset_name"] = bundle.metadata.get("dataset_name", "unknown")

    result.warnings = warnings
    result.errors = errors

    return result


def evaluate_multitask(
    bundle: PredictionBundle,
    config: Optional[EvaluationConfig] = None,
    primary_task: Optional[str] = None,
) -> EvaluationResult:
    """评估 multitask 任务。

    每个任务头独立调用 pointwise 评估。
    不做"直接平均所有任务头"的默认总分。

    Parameters
    ----------
    bundle : PredictionBundle
        预测产物（task_type="multitask"）。
    config : EvaluationConfig, optional
        评估配置。
    primary_task : str, optional
        主任务名称。只对 primary_task 生成 benchmark 主指标。

    Returns
    -------
    EvaluationResult
        结构化评估结果。
    """
    if config is None:
        config = EvaluationConfig()

    result = EvaluationResult()
    warnings: List[str] = []
    errors: List[str] = []

    if bundle.task_outputs is None or bundle.task_labels is None:
        errors.append("multitask bundle 缺少 task_outputs 或 task_labels")
        result.errors = errors
        return result

    # 收集所有任务名
    task_names = set(bundle.task_outputs.keys()) | set(bundle.task_labels.keys())

    if primary_task is None:
        # 尝试从 metadata 获取，或使用第一个任务
        primary_task = bundle.metadata.get("primary_task", sorted(task_names)[0] if task_names else None)

    # 默认为 primary_task 的第一个
    if primary_task and primary_task not in task_names:
        warnings.append(f"primary_task '{primary_task}' not in available tasks: {task_names}")
        primary_task = sorted(task_names)[0] if task_names else None

    threshold_strategy = config.threshold_strategy

    for task_name in sorted(task_names):
        task_scores = bundle.task_outputs.get(task_name)
        task_labels = bundle.task_labels.get(task_name)
        task_mask = bundle.task_masks.get(task_name) if bundle.task_masks else None

        if task_scores is None or task_labels is None:
            warnings.append(f"task '{task_name}': missing scores or labels, skipped")
            continue

        # 应用 mask
        if task_mask is not None:
            mask = np.asarray(task_mask, dtype=bool)
            task_scores = [s for i, s in enumerate(task_scores) if mask[i]]
            task_labels = [label for i, label in enumerate(task_labels) if mask[i]]

        if len(task_scores) == 0 or len(task_labels) == 0:
            warnings.append(f"task '{task_name}': empty after mask, skipped")
            continue

        # 获取该任务的 pos_label（从 task_outputs 元信息或默认值）
        task_pos_label = bundle.pos_label or 1

        # 获取该任务的阈值
        if threshold_strategy == "per_task":
            task_threshold = config.threshold
        else:
            task_threshold = config.get_threshold(bundle)

        # 获取该任务的样本权重
        task_sample_weight = None
        if config.sample_weight_enabled and bundle.sample_weight:
            sample_weight_arr = np.asarray(bundle.sample_weight)
            if task_mask is not None:
                sample_weight_arr = sample_weight_arr[np.asarray(task_mask, dtype=bool)]
            task_sample_weight = sample_weight_arr

        try:
            # 判断该任务的 problem_type
            task_problem_type = bundle.metadata.get(
                f"task_{task_name}_problem_type", bundle.problem_type
            )

            if task_problem_type in ("binary", "implicit_ranking"):
                task_result: ClassificationMetricsResult = evaluate_binary_classification(
                    y_true=np.asarray(task_labels),
                    y_score=np.asarray(task_scores),
                    threshold=task_threshold,
                    pos_label=task_pos_label,
                    sample_weight=task_sample_weight,
                    metrics=config.metrics,
                    generate_curves=(task_name == primary_task and config.generate_curves),
                )

                task_metrics = task_result.metrics

                # 如果是主任务，填充曲线和 summary
                if task_name == primary_task:
                    result.summary_metrics = task_metrics

                    if config.generate_curves and task_result.roc_curve:
                        result.curve_artifacts["roc_curve"] = task_result.roc_curve
                    if config.generate_curves and task_result.pr_curve:
                        result.curve_artifacts["pr_curve"] = task_result.pr_curve

                    if task_result.confusion_matrix:
                        result.metadata["confusion_matrix"] = {
                            "tp": task_result.confusion_matrix.tp,
                            "fp": task_result.confusion_matrix.fp,
                            "tn": task_result.confusion_matrix.tn,
                            "fn": task_result.confusion_matrix.fn,
                        }

                warnings.extend(task_result.warnings)

            elif task_problem_type == "multiclass":
                task_result = evaluate_multiclass_classification(
                    y_true=np.asarray(task_labels),
                    y_score=np.asarray(task_scores),
                    average=config.average,  # type: ignore
                    sample_weight=task_sample_weight,
                    metrics=config.metrics,
                )
                task_metrics = task_result.metrics

                if task_name == primary_task:
                    result.summary_metrics = task_metrics

            else:
                warnings.append(f"task '{task_name}': unsupported problem_type '{task_problem_type}', skipped")
                continue

            # 将各任务指标带上前缀放入 task_metrics
            result.task_metrics[task_name] = task_metrics

        except Exception as e:
            errors.append(f"task '{task_name}' evaluation failed: {str(e)}")

    # 元信息
    result.metadata.update({
        "task_type": "multitask",
        "problem_type": bundle.problem_type,
        "threshold_strategy": threshold_strategy,
        "num_tasks": len(task_names),
        "evaluated_tasks": list(result.task_metrics.keys()),
        "primary_task": primary_task,
        "primary_metric": config.get_primary_metric("multitask", bundle.problem_type) if primary_task else None,
        "schema_version": "1.0.0",
    })

    # 注入 bundle metadata
    if bundle.metadata:
        result.metadata["model_name"] = bundle.metadata.get("model_name", "unknown")
        result.metadata["dataset_name"] = bundle.metadata.get("dataset_name", "unknown")

    result.warnings = warnings
    result.errors = errors

    return result
