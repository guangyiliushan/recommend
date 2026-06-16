"""Classification metrics for pointwise and multitask evaluation.

职责定位：
- 只负责 pointwise 和 multitask 里的单任务头分类数值指标
- 不处理排序，不落盘，不画图
- 包装成熟库实现，维护契约与边界

三层实现：
- 第一层：原子统计层（混淆矩阵与基础计数）
- 第二层：派生指标层（accuracy, precision, recall, f1, specificity, npv 等）
- 第三层：阈值无关层（roc_auc, pr_auc, log_loss, brier_score, 曲线点）

指标命名规范：
- 内部统一用规范英文键名（accuracy, precision, recall, f1, specificity, npv, roc_auc, pr_auc）
- 展示层再映射中文或别名（ACC, PPV, Sensitivity, TNR, F1-score）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

# 尝试导入 sklearn，如果不可用则提供降级实现
try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# 指标别名映射
# ---------------------------------------------------------------------------

METRIC_ALIASES: Dict[str, str] = {
    # accuracy 别名
    "acc": "accuracy",
    "ACC": "accuracy",
    
    # precision 别名
    "ppv": "precision",
    "PPV": "precision",
    "精确度": "precision",
    
    # recall 别名
    "sensitivity": "recall",
    "Sensitivity": "recall",
    "敏感性": "recall",
    "召回率": "recall",
    "tpr": "recall",
    "TPR": "recall",
    
    # specificity 别名
    "tnr": "specificity",
    "TNR": "specificity",
    "特异性": "specificity",
    
    # f1 别名
    "f1_score": "f1",
    "F1": "f1",
    "F1-score": "f1",
    
    # roc_auc 别名
    "ROC-AUC": "roc_auc",
    "ROC AUC": "roc_auc",
    "auroc": "roc_auc",
    
    # pr_auc 别名
    "PR-AUC": "pr_auc",
    "PR AUC": "pr_auc",
    "auprc": "pr_auc",
    
    # npv 别名
    "NPV": "npv",
    
    # log_loss 别名
    "logloss": "log_loss",
    "cross_entropy": "log_loss",
    
    # brier_score 别名
    "brier": "brier_score",
}

# 规范指标键列表
CANONICAL_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "npv",
    "fpr",
    "fnr",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "average_precision",
    "log_loss",
    "brier_score",
]


def normalize_metric_name(name: str) -> str:
    """将指标别名映射为规范名称。
    
    Parameters
    ----------
    name : str
        指标名称或别名。
    
    Returns
    -------
    str
        规范指标名称。
    """
    return METRIC_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# 数据结构定义
# ---------------------------------------------------------------------------

@dataclass
class ConfusionMatrixResult:
    """混淆矩阵结果。
    
    Attributes
    ----------
    tp : int
        True Positive 数量。
    fp : int
        False Positive 数量。
    tn : int
        True Negative 数量。
    fn : int
        False Negative 数量。
    support : int
        总样本数。
    support_positive : int
        正类样本数。
    support_negative : int
        负类样本数。
    prevalence : float
        正类比例。
    """
    
    tp: int
    fp: int
    tn: int
    fn: int
    support: int
    support_positive: int
    support_negative: int
    prevalence: float
    
    @classmethod
    def from_labels(
        cls,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        pos_label: Any = 1,
    ) -> "ConfusionMatrixResult":
        """从标签计算混淆矩阵。
        
        Parameters
        ----------
        y_true : ArrayLike
            真实标签。
        y_pred : ArrayLike
            预测标签。
        pos_label : Any, optional
            正类标签值，默认为 1。
        
        Returns
        -------
        ConfusionMatrixResult
            混淆矩阵结果。
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # 二值化
        y_true_bin = (y_true == pos_label).astype(int)
        y_pred_bin = (y_pred == pos_label).astype(int)
        
        tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
        tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
        fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
        
        support = tp + fp + tn + fn
        support_positive = tp + fn
        support_negative = tn + fp
        prevalence = support_positive / support if support > 0 else 0.0
        
        return cls(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            support=support,
            support_positive=support_positive,
            support_negative=support_negative,
            prevalence=prevalence,
        )


@dataclass
class ClassificationMetricsResult:
    """分类指标结果。
    
    Attributes
    ----------
    metrics : Dict[str, float]
        指标值字典。
    confusion_matrix : Optional[ConfusionMatrixResult]
        混淆矩阵结果（仅二分类）。
    roc_curve : Optional[Dict[str, ArrayLike]]
        ROC 曲线点（fpr, tpr, thresholds）。
    pr_curve : Optional[Dict[str, ArrayLike]]
        PR 曲线点（precision, recall, thresholds）。
    metadata : Dict[str, Any]
        元信息。
    warnings : List[str]
        警告信息列表。
    """
    
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[ConfusionMatrixResult] = None
    roc_curve: Optional[Dict[str, ArrayLike]] = None
    pr_curve: Optional[Dict[str, ArrayLike]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 第一层：原子统计层
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: Any = 1,
) -> ConfusionMatrixResult:
    """计算混淆矩阵。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_pred : ArrayLike
        预测标签。
    pos_label : Any, optional
        正类标签值，默认为 1。
    
    Returns
    -------
    ConfusionMatrixResult
        混淆矩阵结果。
    """
    return ConfusionMatrixResult.from_labels(y_true, y_pred, pos_label)


def compute_confusion_matrix_from_scores(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float = 0.5,
    pos_label: Any = 1,
) -> ConfusionMatrixResult:
    """从分数计算混淆矩阵。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_score : ArrayLike
        预测分数。
    threshold : float, optional
        分类阈值，默认为 0.5。
    pos_label : Any, optional
        正类标签值，默认为 1。
    
    Returns
    -------
    ConfusionMatrixResult
        混淆矩阵结果。
    """
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    
    # 如果 pos_label 不是 1，需要映射
    if pos_label != 1:
        y_true = np.asarray(y_true)
        y_true_bin = (y_true == pos_label).astype(int)
        return ConfusionMatrixResult.from_labels(y_true_bin, y_pred, pos_label=1)
    
    return ConfusionMatrixResult.from_labels(y_true, y_pred, pos_label=pos_label)


# ---------------------------------------------------------------------------
# 第二层：派生指标层
# ---------------------------------------------------------------------------

def compute_accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    normalize: bool = True,
) -> float:
    """计算准确率。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_pred : ArrayLike
        预测标签。
    sample_weight : ArrayLike, optional
        样本权重。
    normalize : bool, optional
        是否返回比例，默认为 True。
    
    Returns
    -------
    float
        准确率。
    """
    if SKLEARN_AVAILABLE:
        return float(accuracy_score(y_true, y_pred, sample_weight=sample_weight, normalize=normalize))
    
    # 降级实现
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    correct = (y_true == y_pred)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if normalize:
            return float(np.sum(correct * sample_weight) / np.sum(sample_weight))
        return float(np.sum(correct * sample_weight))
    
    if normalize:
        return float(np.mean(correct))
    return float(np.sum(correct))


def compute_precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: Any = 1,
    average: Literal["binary", "macro", "micro", "weighted"] = "binary",
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Union[str, float] = 0.0,
) -> float:
    """计算精确率（PPV）。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_pred : ArrayLike
        预测标签。
    pos_label : Any, optional
        正类标签值，默认为 1。
    average : str, optional
        平均方式，默认为 "binary"。
    sample_weight : ArrayLike, optional
        样本权重。
    zero_division : Union[str, float], optional
        除零时的返回值，默认为 0.0。
    
    Returns
    -------
    float
        精确率。
    """
    if SKLEARN_AVAILABLE:
        return float(precision_score(
            y_true, y_pred,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        ))
    
    # 降级实现（仅支持 binary）
    cm = compute_confusion_matrix(y_true, y_pred, pos_label=pos_label)
    if cm.tp + cm.fp == 0:
        return float(zero_division) if isinstance(zero_division, (int, float)) else 0.0
    return cm.tp / (cm.tp + cm.fp)


def compute_recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: Any = 1,
    average: Literal["binary", "macro", "micro", "weighted"] = "binary",
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Union[str, float] = 0.0,
) -> float:
    """计算召回率（敏感性、TPR）。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_pred : ArrayLike
        预测标签。
    pos_label : Any, optional
        正类标签值，默认为 1。
    average : str, optional
        平均方式，默认为 "binary"。
    sample_weight : ArrayLike, optional
        样本权重。
    zero_division : Union[str, float], optional
        除零时的返回值，默认为 0.0。
    
    Returns
    -------
    float
        召回率。
    """
    if SKLEARN_AVAILABLE:
        return float(recall_score(
            y_true, y_pred,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        ))
    
    # 降级实现（仅支持 binary）
    cm = compute_confusion_matrix(y_true, y_pred, pos_label=pos_label)
    if cm.tp + cm.fn == 0:
        return float(zero_division) if isinstance(zero_division, (int, float)) else 0.0
    return cm.tp / (cm.tp + cm.fn)


def compute_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: Any = 1,
    average: Literal["binary", "macro", "micro", "weighted"] = "binary",
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Union[str, float] = 0.0,
) -> float:
    """计算 F1 分数。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_pred : ArrayLike
        预测标签。
    pos_label : Any, optional
        正类标签值，默认为 1。
    average : str, optional
        平均方式，默认为 "binary"。
    sample_weight : ArrayLike, optional
        样本权重。
    zero_division : Union[str, float], optional
        除零时的返回值，默认为 0.0。
    
    Returns
    -------
    float
        F1 分数。
    """
    if SKLEARN_AVAILABLE:
        return float(f1_score(
            y_true, y_pred,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        ))
    
    # 降级实现（仅支持 binary）
    precision = compute_precision(y_true, y_pred, pos_label, average, sample_weight, zero_division)
    recall = compute_recall(y_true, y_pred, pos_label, average, sample_weight, zero_division)
    
    if precision + recall == 0:
        return float(zero_division) if isinstance(zero_division, (int, float)) else 0.0
    return 2 * precision * recall / (precision + recall)


def compute_specificity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: Any = 1,
) -> float:
    """计算特异性（TNR）。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_pred : ArrayLike
        预测标签。
    pos_label : Any, optional
        正类标签值，默认为 1。
    
    Returns
    -------
    float
        特异性。
    """
    cm = compute_confusion_matrix(y_true, y_pred, pos_label=pos_label)
    if cm.tn + cm.fp == 0:
        return 0.0
    return cm.tn / (cm.tn + cm.fp)


def compute_npv(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: Any = 1,
) -> float:
    """计算阴性预测值（NPV）。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_pred : ArrayLike
        预测标签。
    pos_label : Any, optional
        正类标签值，默认为 1。
    
    Returns
    -------
    float
        NPV。
    """
    cm = compute_confusion_matrix(y_true, y_pred, pos_label=pos_label)
    if cm.tn + cm.fn == 0:
        return 0.0
    return cm.tn / (cm.tn + cm.fn)


def compute_fpr(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: Any = 1,
) -> float:
    """计算假阳性率（FPR）。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_pred : ArrayLike
        预测标签。
    pos_label : Any, optional
        正类标签值，默认为 1。
    
    Returns
    -------
    float
        FPR。
    """
    specificity = compute_specificity(y_true, y_pred, pos_label)
    return 1.0 - specificity


def compute_fnr(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: Any = 1,
) -> float:
    """计算假阴性率（FNR）。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_pred : ArrayLike
        预测标签。
    pos_label : Any, optional
        正类标签值，默认为 1。
    
    Returns
    -------
    float
        FNR。
    """
    recall = compute_recall(y_true, y_pred, pos_label)
    return 1.0 - recall


def compute_balanced_accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pos_label: Any = 1,
) -> float:
    """计算平衡准确率。
    
    平衡准确率 = (sensitivity + specificity) / 2
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_pred : ArrayLike
        预测标签。
    pos_label : Any, optional
        正类标签值，默认为 1。
    
    Returns
    -------
    float
        平衡准确率。
    """
    sensitivity = compute_recall(y_true, y_pred, pos_label)
    specificity = compute_specificity(y_true, y_pred, pos_label)
    return (sensitivity + specificity) / 2.0


# ---------------------------------------------------------------------------
# 第三层：阈值无关层
# ---------------------------------------------------------------------------

def compute_roc_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    average: Optional[str] = None,
    multi_class: str = "ovr",
) -> float:
    """计算 ROC-AUC。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_score : ArrayLike
        预测分数（概率或 logit）。
    sample_weight : ArrayLike, optional
        样本权重。
    average : str, optional
        多分类平均方式。
    multi_class : str, optional
        多分类策略，默认为 "ovr"（one-vs-rest）。
    
    Returns
    -------
    float
        ROC-AUC 值。
    
    Raises
    ------
    ValueError
        如果 sklearn 不可用。
    """
    if not SKLEARN_AVAILABLE:
        raise ValueError("sklearn is required for roc_auc computation")
    
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # 判断是二分类还是多分类
    if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
        # 二分类
        return float(roc_auc_score(y_true, y_score.ravel(), sample_weight=sample_weight))
    else:
        # 多分类
        return float(roc_auc_score(
            y_true, y_score,
            average=average,
            multi_class=multi_class,
            sample_weight=sample_weight,
        ))


def compute_pr_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    pos_label: Any = 1,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """计算 PR-AUC（使用 average_precision）。
    
    注意：此函数返回的是 average_precision，即 PR 曲线下的面积。
    如果需要梯形积分面积，请使用 compute_pr_auc_trapezoid。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_score : ArrayLike
        预测分数（概率或 logit）。
    pos_label : Any, optional
        正类标签值，默认为 1。
    sample_weight : ArrayLike, optional
        样本权重。
    
    Returns
    -------
    float
        PR-AUC 值（average_precision）。
    
    Raises
    ------
    ValueError
        如果 sklearn 不可用。
    """
    if not SKLEARN_AVAILABLE:
        raise ValueError("sklearn is required for pr_auc computation")
    
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score).ravel()
    
    # 二值化标签
    y_true_bin = (y_true == pos_label).astype(int)
    
    return float(average_precision_score(y_true_bin, y_score, sample_weight=sample_weight))


def compute_average_precision(
    y_true: ArrayLike,
    y_score: ArrayLike,
    pos_label: Any = 1,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """计算平均精度（Average Precision）。
    
    这是 PR-AUC 的推荐实现。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_score : ArrayLike
        预测分数（概率或 logit）。
    pos_label : Any, optional
        正类标签值，默认为 1。
    sample_weight : ArrayLike, optional
        样本权重。
    
    Returns
    -------
    float
        平均精度值。
    """
    return compute_pr_auc(y_true, y_score, pos_label, sample_weight)


def compute_log_loss(
    y_true: ArrayLike,
    y_score: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    normalize: bool = True,
    labels: Optional[ArrayLike] = None,
) -> float:
    """计算对数损失（交叉熵）。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_score : ArrayLike
        预测概率。
    sample_weight : ArrayLike, optional
        样本权重。
    normalize : bool, optional
        是否返回平均值，默认为 True。
    labels : ArrayLike, optional
        标签列表。
    
    Returns
    -------
    float
        对数损失值。
    
    Raises
    ------
    ValueError
        如果 sklearn 不可用。
    """
    if not SKLEARN_AVAILABLE:
        raise ValueError("sklearn is required for log_loss computation")
    
    return float(log_loss(
        y_true, y_score,
        sample_weight=sample_weight,
        normalize=normalize,
        labels=labels,
    ))


def compute_brier_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    pos_label: Any = 1,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """计算 Brier 分数。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_score : ArrayLike
        预测概率。
    pos_label : Any, optional
        正类标签值，默认为 1。
    sample_weight : ArrayLike, optional
        样本权重。
    
    Returns
    -------
    float
        Brier 分数。
    
    Raises
    ------
    ValueError
        如果 sklearn 不可用。
    """
    if not SKLEARN_AVAILABLE:
        raise ValueError("sklearn is required for brier_score computation")
    
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score).ravel()
    
    # 二值化标签
    y_true_bin = (y_true == pos_label).astype(int)
    
    return float(brier_score_loss(y_true_bin, y_score, sample_weight=sample_weight))


def compute_roc_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    pos_label: Any = 1,
    sample_weight: Optional[ArrayLike] = None,
    drop_intermediate: bool = True,
) -> Dict[str, np.ndarray]:
    """计算 ROC 曲线点。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_score : ArrayLike
        预测分数。
    pos_label : Any, optional
        正类标签值，默认为 1。
    sample_weight : ArrayLike, optional
        样本权重。
    drop_intermediate : bool, optional
        是否丢弃次优阈值，默认为 True。
    
    Returns
    -------
    Dict[str, np.ndarray]
        包含 fpr, tpr, thresholds 的字典。
    
    Raises
    ------
    ValueError
        如果 sklearn 不可用。
    """
    if not SKLEARN_AVAILABLE:
        raise ValueError("sklearn is required for roc_curve computation")
    
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score).ravel()
    
    # 二值化标签
    y_true_bin = (y_true == pos_label).astype(int)
    
    fpr, tpr, thresholds = roc_curve(
        y_true_bin, y_score,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate,
    )
    
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def compute_pr_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    pos_label: Any = 1,
    sample_weight: Optional[ArrayLike] = None,
) -> Dict[str, np.ndarray]:
    """计算 PR 曲线点。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_score : ArrayLike
        预测分数。
    pos_label : Any, optional
        正类标签值，默认为 1。
    sample_weight : ArrayLike, optional
        样本权重。
    
    Returns
    -------
    Dict[str, np.ndarray]
        包含 precision, recall, thresholds 的字典。
    
    Raises
    ------
    ValueError
        如果 sklearn 不可用。
    """
    if not SKLEARN_AVAILABLE:
        raise ValueError("sklearn is required for pr_curve computation")
    
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score).ravel()
    
    # 二值化标签
    y_true_bin = (y_true == pos_label).astype(int)
    
    precision, recall, thresholds = precision_recall_curve(
        y_true_bin, y_score,
        sample_weight=sample_weight,
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
# 综合评估函数
# ---------------------------------------------------------------------------

def evaluate_binary_classification(
    y_true: ArrayLike,
    y_score: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    threshold: float = 0.5,
    pos_label: Any = 1,
    sample_weight: Optional[ArrayLike] = None,
    metrics: Optional[List[str]] = None,
    generate_curves: bool = True,
) -> ClassificationMetricsResult:
    """评估二分类任务。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_score : ArrayLike
        预测分数（概率或 logit）。
    y_pred : ArrayLike, optional
        预测标签。如果未提供，则使用 threshold 从 y_score 生成。
    threshold : float, optional
        分类阈值，默认为 0.5。
    pos_label : Any, optional
        正类标签值，默认为 1。
    sample_weight : ArrayLike, optional
        样本权重。
    metrics : List[str], optional
        要计算的指标列表。如果未提供，则计算所有指标。
    generate_curves : bool, optional
        是否生成曲线数据，默认为 True。
    
    Returns
    -------
    ClassificationMetricsResult
        分类指标结果。
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score).ravel()
    
    # 生成预测标签
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred)
    
    # 默认指标列表
    if metrics is None:
        metrics = [
            "accuracy", "precision", "recall", "f1",
            "specificity", "npv", "fpr", "fnr",
            "balanced_accuracy",
            "roc_auc", "pr_auc", "average_precision",
            "log_loss", "brier_score",
        ]
    
    result = ClassificationMetricsResult()
    warnings: List[str] = []
    
    # 计算混淆矩阵
    cm = compute_confusion_matrix(y_true, y_pred, pos_label)
    result.confusion_matrix = cm
    
    # 检查类别不平衡
    if cm.prevalence < 0.1 or cm.prevalence > 0.9:
        warnings.append(f"class_imbalance_detected: prevalence={cm.prevalence:.4f}")
    
    # 计算派生指标
    for metric in metrics:
        metric = normalize_metric_name(metric)
        
        try:
            if metric == "accuracy":
                result.metrics["accuracy"] = compute_accuracy(y_true, y_pred, sample_weight)
            elif metric == "precision":
                result.metrics["precision"] = compute_precision(y_true, y_pred, pos_label, "binary", sample_weight)
            elif metric == "recall":
                result.metrics["recall"] = compute_recall(y_true, y_pred, pos_label, "binary", sample_weight)
            elif metric == "f1":
                result.metrics["f1"] = compute_f1(y_true, y_pred, pos_label, "binary", sample_weight)
            elif metric == "specificity":
                result.metrics["specificity"] = compute_specificity(y_true, y_pred, pos_label)
            elif metric == "npv":
                result.metrics["npv"] = compute_npv(y_true, y_pred, pos_label)
            elif metric == "fpr":
                result.metrics["fpr"] = compute_fpr(y_true, y_pred, pos_label)
            elif metric == "fnr":
                result.metrics["fnr"] = compute_fnr(y_true, y_pred, pos_label)
            elif metric == "balanced_accuracy":
                result.metrics["balanced_accuracy"] = compute_balanced_accuracy(y_true, y_pred, pos_label)
            elif metric == "roc_auc":
                result.metrics["roc_auc"] = compute_roc_auc(y_true, y_score, sample_weight)
            elif metric in ("pr_auc", "average_precision"):
                result.metrics["average_precision"] = compute_average_precision(y_true, y_score, pos_label, sample_weight)
                result.metrics["pr_auc"] = result.metrics["average_precision"]
            elif metric == "log_loss":
                # 确保 y_score 是概率
                probs = np.clip(y_score, 1e-10, 1 - 1e-10)
                result.metrics["log_loss"] = compute_log_loss(y_true, probs, sample_weight)
            elif metric == "brier_score":
                result.metrics["brier_score"] = compute_brier_score(y_true, y_score, pos_label, sample_weight)
        except Exception as e:
            warnings.append(f"failed_to_compute_{metric}: {str(e)}")
    
    # 生成曲线
    if generate_curves:
        try:
            result.roc_curve = compute_roc_curve(y_true, y_score, pos_label, sample_weight)
        except Exception as e:
            warnings.append(f"failed_to_compute_roc_curve: {str(e)}")
        
        try:
            result.pr_curve = compute_pr_curve(y_true, y_score, pos_label, sample_weight)
        except Exception as e:
            warnings.append(f"failed_to_compute_pr_curve: {str(e)}")
    
    # 元信息
    result.metadata = {
        "task_type": "pointwise",
        "problem_type": "binary",
        "threshold": threshold,
        "pos_label": pos_label,
        "num_samples": len(y_true),
        "num_positive": cm.support_positive,
        "num_negative": cm.support_negative,
        "prevalence": cm.prevalence,
    }
    
    result.warnings = warnings
    
    return result


def evaluate_multiclass_classification(
    y_true: ArrayLike,
    y_score: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    labels: Optional[ArrayLike] = None,
    average: Literal["macro", "micro", "weighted"] = "macro",
    sample_weight: Optional[ArrayLike] = None,
    metrics: Optional[List[str]] = None,
) -> ClassificationMetricsResult:
    """评估多分类任务。
    
    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_score : ArrayLike
        预测概率矩阵（n_samples, n_classes）。
    y_pred : ArrayLike, optional
        预测标签。如果未提供，则从 y_score 取 argmax。
    labels : ArrayLike, optional
        类别标签列表。
    average : str, optional
        平均方式，默认为 "macro"。
    sample_weight : ArrayLike, optional
        样本权重。
    metrics : List[str], optional
        要计算的指标列表。
    
    Returns
    -------
    ClassificationMetricsResult
        分类指标结果。
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # 生成预测标签
    if y_pred is None:
        y_pred = np.argmax(y_score, axis=1)
    else:
        y_pred = np.asarray(y_pred)
    
    # 默认指标列表
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]
    
    result = ClassificationMetricsResult()
    warnings: List[str] = []
    
    # 计算指标
    for metric in metrics:
        metric = normalize_metric_name(metric)
        
        try:
            if metric == "accuracy":
                result.metrics["accuracy"] = compute_accuracy(y_true, y_pred, sample_weight)
            elif metric == "precision":
                result.metrics[f"precision_{average}"] = compute_precision(
                    y_true, y_pred, average=average, sample_weight=sample_weight
                )
            elif metric == "recall":
                result.metrics[f"recall_{average}"] = compute_recall(
                    y_true, y_pred, average=average, sample_weight=sample_weight
                )
            elif metric == "f1":
                result.metrics[f"f1_{average}"] = compute_f1(
                    y_true, y_pred, average=average, sample_weight=sample_weight
                )
            elif metric == "roc_auc":
                result.metrics[f"roc_auc_{average}"] = compute_roc_auc(
                    y_true, y_score, average=average, sample_weight=sample_weight
                )
            elif metric == "log_loss":
                result.metrics["log_loss"] = compute_log_loss(
                    y_true, y_score, sample_weight=sample_weight, labels=labels
                )
        except Exception as e:
            warnings.append(f"failed_to_compute_{metric}: {str(e)}")
    
    # 元信息
    n_classes = y_score.shape[1] if y_score.ndim > 1 else len(np.unique(y_true))
    result.metadata = {
        "task_type": "pointwise",
        "problem_type": "multiclass",
        "average": average,
        "num_samples": len(y_true),
        "num_classes": n_classes,
    }
    
    result.warnings = warnings
    
    return result
