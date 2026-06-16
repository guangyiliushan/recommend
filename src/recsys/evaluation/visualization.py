"""Visualization utilities for evaluation results.

职责定位：
- 只消费已经算好的曲线点和结构化结果
- 输出图表或原始曲线文件
- 不负责重新计算数值指标、推断阈值、推断任务类型

推荐输入：
- curve_artifacts: 曲线数据
- summary_metrics: 汇总指标
- group_metrics: 分组指标
- metadata: 元信息

推荐输出：
- roc_curve.json: ROC 曲线数据
- pr_curve.json: PR 曲线数据
- metric_leaderboard.csv/png/html: 指标排行榜
- ndcg_at_k.json: NDCG@K 数据
- threshold_sweep.json: 阈值扫描结果

设计原则：
- 原始曲线点优先于图片
- 图片只是展示层附加产物
- 支持多模型叠加，但必须保证同一任务、同一数据集、同一 split
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# 数据结构定义
# ---------------------------------------------------------------------------

@dataclass
class CurveData:
    """曲线数据容器。

    Attributes
    ----------
    x : np.ndarray
        X 轴数据。
    y : np.ndarray
        Y 轴数据。
    label : str
        曲线标签。
    metadata : Dict[str, Any]
        附加元信息。
    """

    x: np.ndarray
    y: np.ndarray
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。"""
        return {
            "x": self.x.tolist() if isinstance(self.x, np.ndarray) else list(self.x),
            "y": self.y.tolist() if isinstance(self.y, np.ndarray) else list(self.y),
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurveData":
        """从字典创建。"""
        return cls(
            x=np.array(data["x"]),
            y=np.array(data["y"]),
            label=data.get("label", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VisualizationOutput:
    """可视化输出结果。

    Attributes
    ----------
    curve_files : Dict[str, Path]
        曲线文件路径映射。
    image_files : Dict[str, Path]
        图片文件路径映射。
    data : Dict[str, Any]
        内存中的数据结构。
    warnings : List[str]
        警告信息。
    """

    curve_files: Dict[str, Path] = field(default_factory=dict)
    image_files: Dict[str, Path] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 曲线数据导出
# ---------------------------------------------------------------------------

def export_roc_curve(
    fpr: ArrayLike,
    tpr: ArrayLike,
    thresholds: Optional[ArrayLike] = None,
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "model",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """导出 ROC 曲线数据。

    Parameters
    ----------
    fpr : ArrayLike
        假阳性率。
    tpr : ArrayLike
        真阳性率。
    thresholds : ArrayLike, optional
        阈值列表。
    output_path : Union[str, Path], optional
        输出文件路径。如果未提供，只返回数据结构。
    model_name : str, optional
        模型名称，默认为 "model"。
    metadata : Dict[str, Any], optional
        附加元信息。

    Returns
    -------
    Dict[str, Any]
        ROC 曲线数据结构。
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    data = {
        "curve_type": "roc",
        "model_name": model_name,
        "x_axis": "fpr",
        "y_axis": "tpr",
        "x_label": "False Positive Rate",
        "y_label": "True Positive Rate",
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "metadata": metadata or {},
    }

    if thresholds is not None:
        thresholds = np.asarray(thresholds)
        data["thresholds"] = thresholds.tolist()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return data


def export_pr_curve(
    precision: ArrayLike,
    recall: ArrayLike,
    thresholds: Optional[ArrayLike] = None,
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "model",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """导出 PR 曲线数据。

    Parameters
    ----------
    precision : ArrayLike
        精确率。
    recall : ArrayLike
        召回率。
    thresholds : ArrayLike, optional
        阈值列表。
    output_path : Union[str, Path], optional
        输出文件路径。
    model_name : str, optional
        模型名称。
    metadata : Dict[str, Any], optional
        附加元信息。

    Returns
    -------
    Dict[str, Any]
        PR 曲线数据结构。
    """
    precision = np.asarray(precision)
    recall = np.asarray(recall)

    data = {
        "curve_type": "pr",
        "model_name": model_name,
        "x_axis": "recall",
        "y_axis": "precision",
        "x_label": "Recall",
        "y_label": "Precision",
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "metadata": metadata or {},
    }

    if thresholds is not None:
        thresholds = np.asarray(thresholds)
        data["thresholds"] = thresholds.tolist()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return data


def export_threshold_sweep(
    thresholds: ArrayLike,
    metrics: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "model",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """导出阈值扫描结果。

    Parameters
    ----------
    thresholds : ArrayLike
        阈值列表。
    metrics : Dict[str, List[float]]
        各阈值下的指标值。
    output_path : Union[str, Path], optional
        输出文件路径。
    model_name : str, optional
        模型名称。
    metadata : Dict[str, Any], optional
        附加元信息。

    Returns
    -------
    Dict[str, Any]
        阈值扫描数据结构。
    """
    thresholds = np.asarray(thresholds)

    data = {
        "sweep_type": "threshold",
        "model_name": model_name,
        "thresholds": thresholds.tolist(),
        "metrics": metrics,
        "metadata": metadata or {},
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return data


def export_ranking_metrics_at_k(
    k_list: List[int],
    metrics: Dict[str, float],
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "model",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """导出排序指标 @K 结果。

    Parameters
    ----------
    k_list : List[int]
        K 值列表。
    metrics : Dict[str, float]
        指标值（如 ndcg@5, ndcg@10 等）。
    output_path : Union[str, Path], optional
        输出文件路径。
    model_name : str, optional
        模型名称。
    metadata : Dict[str, Any], optional
        附加元信息。

    Returns
    -------
    Dict[str, Any]
        排序指标数据结构。
    """
    # 按 K 值组织数据
    metrics_by_k: Dict[int, Dict[str, float]] = {}
    for k in k_list:
        metrics_by_k[k] = {}

    for key, value in metrics.items():
        # 解析指标名，如 "ndcg@10" -> ("ndcg", 10)
        if "@" in key:
            parts = key.split("@")
            metric_name = parts[0]
            k = int(parts[1])
            if k in metrics_by_k:
                metrics_by_k[k][metric_name] = value

    data = {
        "metric_type": "ranking_at_k",
        "model_name": model_name,
        "k_list": k_list,
        "metrics": metrics,
        "metrics_by_k": {str(k): v for k, v in metrics_by_k.items()},
        "metadata": metadata or {},
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return data


# ---------------------------------------------------------------------------
# 图表生成
# ---------------------------------------------------------------------------

def plot_roc_curve(
    fpr: ArrayLike,
    tpr: ArrayLike,
    auc: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "ROC Curve",
    model_name: str = "Model",
    figsize: tuple = (8, 6),
) -> Optional[Any]:
    """绘制 ROC 曲线。

    Parameters
    ----------
    fpr : ArrayLike
        假阳性率。
    tpr : ArrayLike
        真阳性率。
    auc : float, optional
        AUC 值。
    output_path : Union[str, Path], optional
        输出图片路径。
    title : str, optional
        图表标题。
    model_name : str, optional
        模型名称。
    figsize : tuple, optional
        图表尺寸。

    Returns
    -------
    Optional[Any]
        matplotlib Figure 对象（如果 matplotlib 可用）。
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制对角线
    ax.plot([0, 1], [0, 1], "k--", label="Random", alpha=0.5)

    # 绘制 ROC 曲线
    label = f"{model_name}"
    if auc is not None:
        label += f" (AUC = {auc:.4f})"
    ax.plot(fpr, tpr, label=label, linewidth=2)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pr_curve(
    precision: ArrayLike,
    recall: ArrayLike,
    ap: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Precision-Recall Curve",
    model_name: str = "Model",
    figsize: tuple = (8, 6),
) -> Optional[Any]:
    """绘制 PR 曲线。

    Parameters
    ----------
    precision : ArrayLike
        精确率。
    recall : ArrayLike
        召回率。
    ap : float, optional
        Average Precision 值。
    output_path : Union[str, Path], optional
        输出图片路径。
    title : str, optional
        图表标题。
    model_name : str, optional
        模型名称。
    figsize : tuple, optional
        图表尺寸。

    Returns
    -------
    Optional[Any]
        matplotlib Figure 对象（如果 matplotlib 可用）。
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    precision = np.asarray(precision)
    recall = np.asarray(recall)

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制 PR 曲线
    label = f"{model_name}"
    if ap is not None:
        label += f" (AP = {ap:.4f})"
    ax.plot(recall, precision, label=label, linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metrics_at_k(
    k_list: List[int],
    metrics: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Metrics at K",
    figsize: tuple = (10, 6),
) -> Optional[Any]:
    """绘制排序指标 @K 曲线。

    Parameters
    ----------
    k_list : List[int]
        K 值列表。
    metrics : Dict[str, List[float]]
        各 K 值下的指标值。
    output_path : Union[str, Path], optional
        输出图片路径。
    title : str, optional
        图表标题。
    figsize : tuple, optional
        图表尺寸。

    Returns
    -------
    Optional[Any]
        matplotlib Figure 对象（如果 matplotlib 可用）。
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    for metric_name, values in metrics.items():
        if len(values) == len(k_list):
            ax.plot(k_list, values, marker="o", label=metric_name, linewidth=2)

    ax.set_xlabel("K")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_list)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metric_leaderboard(
    metrics_table: Dict[str, Dict[str, float]],
    primary_metric: str = "roc_auc",
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Model Leaderboard",
    figsize: tuple = (12, 8),
) -> Optional[Any]:
    """绘制模型排行榜。

    Parameters
    ----------
    metrics_table : Dict[str, Dict[str, float]]
        模型名到指标字典的映射。
    primary_metric : str, optional
        主排序指标，默认为 "roc_auc"。
    output_path : Union[str, Path], optional
        输出图片路径。
    title : str, optional
        图表标题。
    figsize : tuple, optional
        图表尺寸。

    Returns
    -------
    Optional[Any]
        matplotlib Figure 对象（如果 matplotlib 可用）。
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    # 按主指标排序
    sorted_models = sorted(
        metrics_table.items(),
        key=lambda x: x[1].get(primary_metric, 0),
        reverse=True,
    )

    model_names = [m[0] for m in sorted_models]
    primary_values = [m[1].get(primary_metric, 0) for m in sorted_models]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(model_names))
    ax.barh(y_pos, primary_values, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.invert_yaxis()  # 最好的在顶部
    ax.set_xlabel(primary_metric)
    ax.set_title(title)

    # 添加数值标签
    for i, v in enumerate(primary_values):
        ax.text(v + 0.01, i, f"{v:.4f}", va="center")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 综合可视化函数
# ---------------------------------------------------------------------------

def visualize_classification_results(
    summary_metrics: Dict[str, float],
    roc_curve: Optional[Dict[str, ArrayLike]] = None,
    pr_curve: Optional[Dict[str, ArrayLike]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    model_name: str = "model",
    generate_images: bool = True,
) -> VisualizationOutput:
    """可视化分类任务结果。

    Parameters
    ----------
    summary_metrics : Dict[str, float]
        汇总指标。
    roc_curve : Dict[str, ArrayLike], optional
        ROC 曲线数据（fpr, tpr, thresholds）。
    pr_curve : Dict[str, ArrayLike], optional
        PR 曲线数据（precision, recall, thresholds）。
    output_dir : Union[str, Path], optional
        输出目录。
    model_name : str, optional
        模型名称。
    generate_images : bool, optional
        是否生成图片，默认为 True。

    Returns
    -------
    VisualizationOutput
        可视化输出结果。
    """
    result = VisualizationOutput()
    warnings: List[str] = []

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 导出 ROC 曲线
    if roc_curve is not None:
        fpr = roc_curve.get("fpr")
        tpr = roc_curve.get("tpr")
        thresholds = roc_curve.get("thresholds")

        if fpr is not None and tpr is not None:
            # 导出 JSON
            if output_dir is not None:
                roc_path = output_dir / "roc_curve.json"
                result.data["roc_curve"] = export_roc_curve(
                    fpr, tpr, thresholds, roc_path, model_name,
                    metadata={"auc": summary_metrics.get("roc_auc")},
                )
                result.curve_files["roc_curve"] = roc_path
            else:
                result.data["roc_curve"] = export_roc_curve(
                    fpr, tpr, thresholds, model_name=model_name,
                    metadata={"auc": summary_metrics.get("roc_auc")},
                )

            # 生成图片
            if generate_images and MATPLOTLIB_AVAILABLE and output_dir is not None:
                img_path = output_dir / "roc_curve.png"
                plot_roc_curve(
                    fpr, tpr,
                    auc=summary_metrics.get("roc_auc"),
                    output_path=img_path,
                    model_name=model_name,
                )
                result.image_files["roc_curve"] = img_path

    # 导出 PR 曲线
    if pr_curve is not None:
        precision = pr_curve.get("precision")
        recall = pr_curve.get("recall")
        thresholds = pr_curve.get("thresholds")

        if precision is not None and recall is not None:
            # 导出 JSON
            if output_dir is not None:
                pr_path = output_dir / "pr_curve.json"
                result.data["pr_curve"] = export_pr_curve(
                    precision, recall, thresholds, pr_path, model_name,
                    metadata={"ap": summary_metrics.get("average_precision")},
                )
                result.curve_files["pr_curve"] = pr_path
            else:
                result.data["pr_curve"] = export_pr_curve(
                    precision, recall, thresholds, model_name=model_name,
                    metadata={"ap": summary_metrics.get("average_precision")},
                )

            # 生成图片
            if generate_images and MATPLOTLIB_AVAILABLE and output_dir is not None:
                img_path = output_dir / "pr_curve.png"
                plot_pr_curve(
                    precision, recall,
                    ap=summary_metrics.get("average_precision"),
                    output_path=img_path,
                    model_name=model_name,
                )
                result.image_files["pr_curve"] = img_path

    # 导出汇总指标
    result.data["summary_metrics"] = summary_metrics

    if output_dir is not None:
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2, ensure_ascii=False)
        result.curve_files["metrics"] = metrics_path

    result.warnings = warnings
    return result


def visualize_ranking_results(
    summary_metrics: Dict[str, float],
    k_list: List[int],
    output_dir: Optional[Union[str, Path]] = None,
    model_name: str = "model",
    generate_images: bool = True,
) -> VisualizationOutput:
    """可视化排序任务结果。

    Parameters
    ----------
    summary_metrics : Dict[str, float]
        汇总指标（如 ndcg@5, ndcg@10 等）。
    k_list : List[int]
        K 值列表。
    output_dir : Union[str, Path], optional
        输出目录。
    model_name : str, optional
        模型名称。
    generate_images : bool, optional
        是否生成图片，默认为 True。

    Returns
    -------
    VisualizationOutput
        可视化输出结果。
    """
    result = VisualizationOutput()
    warnings: List[str] = []

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 导出排序指标 @K
    if output_dir is not None:
        ranking_path = output_dir / "ranking_metrics.json"
        result.data["ranking_metrics"] = export_ranking_metrics_at_k(
            k_list, summary_metrics, ranking_path, model_name,
        )
        result.curve_files["ranking_metrics"] = ranking_path
    else:
        result.data["ranking_metrics"] = export_ranking_metrics_at_k(
            k_list, summary_metrics, model_name=model_name,
        )

    # 生成图片
    if generate_images and MATPLOTLIB_AVAILABLE and output_dir is not None:
        # 组织各指标在不同 K 下的值
        metrics_by_name: Dict[str, List[float]] = {}
        for metric_prefix in ["ndcg", "recall", "precision", "hit_rate"]:
            values = []
            for k in k_list:
                key = f"{metric_prefix}@{k}"
                values.append(summary_metrics.get(key, 0.0))
            if any(v > 0 for v in values):
                metrics_by_name[metric_prefix] = values

        if metrics_by_name:
            img_path = output_dir / "metrics_at_k.png"
            plot_metrics_at_k(k_list, metrics_by_name, img_path)
            result.image_files["metrics_at_k"] = img_path

    result.warnings = warnings
    return result


def export_metrics_csv(
    metrics_table: Dict[str, Dict[str, float]],
    output_path: Union[str, Path],
) -> None:
    """导出指标表格为 CSV。

    Parameters
    ----------
    metrics_table : Dict[str, Dict[str, float]]
        模型名到指标字典的映射。
    output_path : Union[str, Path]
        输出文件路径。
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 收集所有指标名
    all_metrics = set()
    for model_metrics in metrics_table.values():
        all_metrics.update(model_metrics.keys())
    all_metrics = sorted(all_metrics)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 写表头
        writer.writerow(["model"] + all_metrics)
        # 写数据
        for model_name, model_metrics in metrics_table.items():
            row = [model_name] + [model_metrics.get(m, "") for m in all_metrics]
            writer.writerow(row)
