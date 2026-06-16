"""Ranking metrics for group-aware evaluation.

职责定位：
- 只负责 ranking 任务的组内排序指标
- 不计算分类混淆矩阵指标
- 逐组计算，再跨组聚合

支持的指标：
- ndcg@k: Normalized Discounted Cumulative Gain
- mrr: Mean Reciprocal Rank
- hit_rate@k: 命中率
- recall@k: 召回率
- precision@k: 精确率
- map: Mean Average Precision

设计原则：
- 任何 ranking 指标都先"逐组计算"，再"跨组聚合"
- 禁止把所有样本摊平后直接计算全局指标
- 对空组、无正样本组、候选不足 K 的组有明确处理策略
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# 指标别名映射
# ---------------------------------------------------------------------------

RANKING_METRIC_ALIASES: Dict[str, str] = {
    "ndcg": "ndcg_at_k",
    "NDCG": "ndcg_at_k",
    "MRR": "mrr",
    "hit": "hit_rate_at_k",
    "hitrate": "hit_rate_at_k",
    "HR": "hit_rate_at_k",
    "MAP": "map",
    "average_precision": "map",
}

# 规范指标键列表
CANONICAL_RANKING_METRICS = [
    "ndcg_at_k",
    "mrr",
    "hit_rate_at_k",
    "recall_at_k",
    "precision_at_k",
    "map",
]


def normalize_ranking_metric_name(name: str) -> str:
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
    return RANKING_METRIC_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# 数据结构定义
# ---------------------------------------------------------------------------

@dataclass
class GroupRankingResult:
    """单组排序结果。

    Attributes
    ----------
    group_id : Any
        分组标识。
    metrics : Dict[str, float]
        该组的各指标值。
    num_candidates : int
        候选数量。
    num_relevant : int
        相关项数量。
    warnings : List[str]
        警告信息。
    """

    group_id: Any
    metrics: Dict[str, float] = field(default_factory=dict)
    num_candidates: int = 0
    num_relevant: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class RankingMetricsResult:
    """排序指标汇总结果。

    Attributes
    ----------
    summary_metrics : Dict[str, float]
        汇总指标（跨组平均）。
    group_metrics : List[GroupRankingResult]
        各组详细结果。
    k_list : List[int]
        评估的 K 值列表。
    metadata : Dict[str, Any]
        元信息。
    warnings : List[str]
        全局警告信息。
    """

    summary_metrics: Dict[str, float] = field(default_factory=dict)
    group_metrics: List[GroupRankingResult] = field(default_factory=list)
    k_list: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _get_relevance_set(
    y_true: Union[Set[Any], List[Any], ArrayLike],
    candidate_ids: Optional[List[Any]] = None,
) -> Set[Any]:
    """获取相关项集合。

    Parameters
    ----------
    y_true : Union[Set[Any], List[Any], ArrayLike]
        真实相关项。可以是 item id 集合，或与 candidate_ids 对齐的 relevance 向量。
    candidate_ids : List[Any], optional
        候选项 ID 列表。当 y_true 是 relevance 向量时需要。

    Returns
    -------
    Set[Any]
        相关项 ID 集合。
    """
    if isinstance(y_true, set):
        return y_true

    y_true_arr = np.asarray(y_true)

    # 如果是布尔/数值向量，需要与 candidate_ids 对齐
    if y_true_arr.dtype in (np.bool_, np.integer, np.floating):
        if candidate_ids is None:
            raise ValueError("当 y_true 是 relevance 向量时，必须提供 candidate_ids")

        # 将 relevance > 0 的项视为相关
        relevant_indices = np.where(y_true_arr > 0)[0]
        return {candidate_ids[i] for i in relevant_indices}

    # 否则视为 item id 列表
    return set(y_true)


def _get_relevance_dict(
    y_true: Union[Dict[Any, float], List[Any], ArrayLike],
    candidate_ids: Optional[List[Any]] = None,
) -> Dict[Any, float]:
    """获取相关项及其相关性分数字典。

    Parameters
    ----------
    y_true : Union[Dict[Any, float], List[Any], ArrayLike]
        真实相关项及其相关性分数。
    candidate_ids : List[Any], optional
        候选项 ID 列表。

    Returns
    -------
    Dict[Any, float]
        相关项 ID 到相关性分数的映射。
    """
    if isinstance(y_true, dict):
        return y_true

    y_true_arr = np.asarray(y_true)

    if y_true_arr.dtype in (np.bool_, np.integer, np.floating):
        if candidate_ids is None:
            raise ValueError("当 y_true 是 relevance 向量时，必须提供 candidate_ids")

        return {
            candidate_ids[i]: float(y_true_arr[i])
            for i in range(len(y_true_arr))
            if y_true_arr[i] > 0
        }

    # 如果是 item id 列表，默认相关性为 1
    return {item: 1.0 for item in y_true}


def _dcg_at_k(
    relevance_scores: List[float],
    k: int,
) -> float:
    """计算 DCG@k。

    DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log_2(i + 1)

    Parameters
    ----------
    relevance_scores : List[float]
        按预测排序的相关性分数列表。
    k : int
        截断位置。

    Returns
    -------
    float
        DCG@k 值。
    """
    relevance_scores = relevance_scores[:k]
    if not relevance_scores:
        return 0.0

    gains = np.power(2.0, np.array(relevance_scores)) - 1.0
    discounts = np.log2(np.arange(1, len(relevance_scores) + 1) + 1)

    return float(np.sum(gains / discounts))


def _ideal_dcg_at_k(
    relevance_scores: List[float],
    k: int,
) -> float:
    """计算理想 DCG@k（IDCG@k）。

    Parameters
    ----------
    relevance_scores : List[float]
        所有可能的相关性分数。
    k : int
        截断位置。

    Returns
    -------
    float
        IDCG@k 值。
    """
    # 按相关性分数降序排列
    sorted_scores = sorted(relevance_scores, reverse=True)
    return _dcg_at_k(sorted_scores, k)


# ---------------------------------------------------------------------------
# 核心指标函数
# ---------------------------------------------------------------------------

def compute_ndcg_at_k(
    y_true: Union[Set[Any], Dict[Any, float], List[Any], ArrayLike],
    y_score: ArrayLike,
    candidate_ids: List[Any],
    k: int,
) -> float:
    """计算单组的 NDCG@k。

    Parameters
    ----------
    y_true : Union[Set[Any], Dict[Any, float], List[Any], ArrayLike]
        真实相关项。可以是 item id 集合、相关性字典或 relevance 向量。
    y_score : ArrayLike
        预测分数，与 candidate_ids 对齐。
    candidate_ids : List[Any]
        候选项 ID 列表。
    k : int
        截断位置。

    Returns
    -------
    float
        NDCG@k 值。
    """
    y_score = np.asarray(y_score)

    # 获取相关性字典
    if isinstance(y_true, dict):
        rel_dict = y_true
    elif isinstance(y_true, set):
        rel_dict = {item: 1.0 for item in y_true}
    else:
        rel_dict = _get_relevance_dict(y_true, candidate_ids)

    if not rel_dict:
        return 0.0

    # 按预测分数排序
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_candidate_ids = [candidate_ids[i] for i in sorted_indices]

    # 获取排序后的相关性分数
    sorted_relevance = [rel_dict.get(cid, 0.0) for cid in sorted_candidate_ids]

    # 计算 DCG 和 IDCG
    dcg = _dcg_at_k(sorted_relevance, k)
    idcg = _ideal_dcg_at_k(list(rel_dict.values()), k)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_mrr(
    y_true: Union[Set[Any], List[Any], ArrayLike],
    y_score: ArrayLike,
    candidate_ids: List[Any],
) -> float:
    """计算单组的 MRR（Mean Reciprocal Rank）。

    Parameters
    ----------
    y_true : Union[Set[Any], List[Any], ArrayLike]
        真实相关项集合。
    y_score : ArrayLike
        预测分数，与 candidate_ids 对齐。
    candidate_ids : List[Any]
        候选项 ID 列表。

    Returns
    -------
    float
        MRR 值（第一个相关项的倒数排名）。
    """
    y_score = np.asarray(y_score)

    # 获取相关项集合
    relevant_set = _get_relevance_set(y_true, candidate_ids)

    if not relevant_set:
        return 0.0

    # 按预测分数排序
    sorted_indices = np.argsort(y_score)[::-1]

    # 找到第一个相关项的排名
    for rank, idx in enumerate(sorted_indices, start=1):
        if candidate_ids[idx] in relevant_set:
            return 1.0 / rank

    return 0.0


def compute_hit_rate_at_k(
    y_true: Union[Set[Any], List[Any], ArrayLike],
    y_score: ArrayLike,
    candidate_ids: List[Any],
    k: int,
) -> float:
    """计算单组的 Hit Rate@k。

    Hit Rate@k = 1 if any relevant item in top-k, else 0

    Parameters
    ----------
    y_true : Union[Set[Any], List[Any], ArrayLike]
        真实相关项集合。
    y_score : ArrayLike
        预测分数，与 candidate_ids 对齐。
    candidate_ids : List[Any]
        候选项 ID 列表。
    k : int
        截断位置。

    Returns
    -------
    float
        Hit Rate@k 值（0 或 1）。
    """
    y_score = np.asarray(y_score)

    # 获取相关项集合
    relevant_set = _get_relevance_set(y_true, candidate_ids)

    if not relevant_set:
        return 0.0

    # 按预测分数排序，取 top-k
    sorted_indices = np.argsort(y_score)[::-1][:k]
    top_k_items = {candidate_ids[i] for i in sorted_indices}

    # 检查是否有交集
    return 1.0 if top_k_items & relevant_set else 0.0


def compute_recall_at_k(
    y_true: Union[Set[Any], List[Any], ArrayLike],
    y_score: ArrayLike,
    candidate_ids: List[Any],
    k: int,
) -> float:
    """计算单组的 Recall@k。

    Recall@k = |relevant items in top-k| / |total relevant items|

    Parameters
    ----------
    y_true : Union[Set[Any], List[Any], ArrayLike]
        真实相关项集合。
    y_score : ArrayLike
        预测分数，与 candidate_ids 对齐。
    candidate_ids : List[Any]
        候选项 ID 列表。
    k : int
        截断位置。

    Returns
    -------
    float
        Recall@k 值。
    """
    y_score = np.asarray(y_score)

    # 获取相关项集合
    relevant_set = _get_relevance_set(y_true, candidate_ids)

    if not relevant_set:
        return 0.0

    # 按预测分数排序，取 top-k
    sorted_indices = np.argsort(y_score)[::-1][:k]
    top_k_items = {candidate_ids[i] for i in sorted_indices}

    # 计算召回率
    hits = len(top_k_items & relevant_set)
    return hits / len(relevant_set)


def compute_precision_at_k(
    y_true: Union[Set[Any], List[Any], ArrayLike],
    y_score: ArrayLike,
    candidate_ids: List[Any],
    k: int,
) -> float:
    """计算单组的 Precision@k。

    Precision@k = |relevant items in top-k| / k

    Parameters
    ----------
    y_true : Union[Set[Any], List[Any], ArrayLike]
        真实相关项集合。
    y_score : ArrayLike
        预测分数，与 candidate_ids 对齐。
    candidate_ids : List[Any]
        候选项 ID 列表。
    k : int
        截断位置。

    Returns
    -------
    float
        Precision@k 值。
    """
    y_score = np.asarray(y_score)

    # 获取相关项集合
    relevant_set = _get_relevance_set(y_true, candidate_ids)

    if not relevant_set:
        return 0.0

    # 按预测分数排序，取 top-k
    sorted_indices = np.argsort(y_score)[::-1][:k]
    top_k_items = {candidate_ids[i] for i in sorted_indices}

    # 计算精确率
    hits = len(top_k_items & relevant_set)
    return hits / k


def compute_ap(
    y_true: Union[Set[Any], List[Any], ArrayLike],
    y_score: ArrayLike,
    candidate_ids: List[Any],
) -> float:
    """计算单组的 AP（Average Precision）。

    AP = sum_{k=1}^{n} (P(k) * rel(k)) / |relevant items|

    Parameters
    ----------
    y_true : Union[Set[Any], List[Any], ArrayLike]
        真实相关项集合。
    y_score : ArrayLike
        预测分数，与 candidate_ids 对齐。
    candidate_ids : List[Any]
        候选项 ID 列表。

    Returns
    -------
    float
        AP 值。
    """
    y_score = np.asarray(y_score)

    # 获取相关项集合
    relevant_set = _get_relevance_set(y_true, candidate_ids)

    if not relevant_set:
        return 0.0

    # 按预测分数排序
    sorted_indices = np.argsort(y_score)[::-1]

    # 计算 AP
    num_relevant = 0
    precision_sum = 0.0

    for rank, idx in enumerate(sorted_indices, start=1):
        if candidate_ids[idx] in relevant_set:
            num_relevant += 1
            precision_sum += num_relevant / rank

    if num_relevant == 0:
        return 0.0

    return precision_sum / len(relevant_set)


# ---------------------------------------------------------------------------
# 分组评估函数
# ---------------------------------------------------------------------------

def evaluate_single_group(
    y_true: Union[Set[Any], Dict[Any, float], List[Any], ArrayLike],
    y_score: ArrayLike,
    candidate_ids: List[Any],
    k_list: List[int],
    metrics: Optional[List[str]] = None,
    group_id: Any = None,
) -> GroupRankingResult:
    """评估单个分组的排序质量。

    Parameters
    ----------
    y_true : Union[Set[Any], Dict[Any, float], List[Any], ArrayLike]
        真实相关项。
    y_score : ArrayLike
        预测分数，与 candidate_ids 对齐。
    candidate_ids : List[Any]
        候选项 ID 列表。
    k_list : List[int]
        评估的 K 值列表。
    metrics : List[str], optional
        要计算的指标列表。默认计算所有指标。
    group_id : Any, optional
        分组标识。

    Returns
    -------
    GroupRankingResult
        单组评估结果。
    """
    y_score = np.asarray(y_score)

    # 默认指标
    if metrics is None:
        metrics = ["ndcg_at_k", "mrr", "hit_rate_at_k", "recall_at_k", "precision_at_k", "map"]

    result = GroupRankingResult(group_id=group_id)
    warnings: List[str] = []

    # 获取相关项集合
    if isinstance(y_true, dict):
        relevant_set = set(y_true.keys())
        rel_dict = y_true
    elif isinstance(y_true, set):
        relevant_set = y_true
        rel_dict = {item: 1.0 for item in y_true}
    else:
        try:
            relevant_set = _get_relevance_set(y_true, candidate_ids)
            rel_dict = _get_relevance_dict(y_true, candidate_ids)
        except ValueError:
            warnings.append("invalid_y_true_format")
            result.warnings = warnings
            return result

    result.num_candidates = len(candidate_ids)
    result.num_relevant = len(relevant_set)

    # 边界检查
    if len(y_score) != len(candidate_ids):
        warnings.append(f"y_score length mismatch: {len(y_score)} vs {len(candidate_ids)}")
        result.warnings = warnings
        return result

    if not relevant_set:
        warnings.append("no_relevant_items")
        result.warnings = warnings
        return result

    # 计算指标
    for metric in metrics:
        metric = normalize_ranking_metric_name(metric)

        try:
            if metric == "ndcg_at_k":
                for k in k_list:
                    if k <= len(candidate_ids):
                        result.metrics[f"ndcg@{k}"] = compute_ndcg_at_k(
                            rel_dict, y_score, candidate_ids, k
                        )
                    else:
                        warnings.append(f"k={k} exceeds candidate count {len(candidate_ids)}")

            elif metric == "mrr":
                result.metrics["mrr"] = compute_mrr(relevant_set, y_score, candidate_ids)

            elif metric == "hit_rate_at_k":
                for k in k_list:
                    if k <= len(candidate_ids):
                        result.metrics[f"hit_rate@{k}"] = compute_hit_rate_at_k(
                            relevant_set, y_score, candidate_ids, k
                        )

            elif metric == "recall_at_k":
                for k in k_list:
                    if k <= len(candidate_ids):
                        result.metrics[f"recall@{k}"] = compute_recall_at_k(
                            relevant_set, y_score, candidate_ids, k
                        )

            elif metric == "precision_at_k":
                for k in k_list:
                    if k <= len(candidate_ids):
                        result.metrics[f"precision@{k}"] = compute_precision_at_k(
                            relevant_set, y_score, candidate_ids, k
                        )

            elif metric == "map":
                result.metrics["map"] = compute_ap(relevant_set, y_score, candidate_ids)

        except Exception as e:
            warnings.append(f"failed_to_compute_{metric}: {str(e)}")

    result.warnings = warnings
    return result


def evaluate_ranking(
    group_ids: List[Any],
    y_true_list: List[Union[Set[Any], Dict[Any, float], List[Any], ArrayLike]],
    y_score_list: List[ArrayLike],
    candidate_ids_list: List[List[Any]],
    k_list: List[int],
    metrics: Optional[List[str]] = None,
) -> RankingMetricsResult:
    """评估排序任务。

    Parameters
    ----------
    group_ids : List[Any]
        分组标识列表。
    y_true_list : List[Union[Set[Any], Dict[Any, float], List[Any], ArrayLike]]
        各组的真实相关项列表。
    y_score_list : List[ArrayLike]
        各组的预测分数列表。
    candidate_ids_list : List[List[Any]]
        各组的候选项 ID 列表。
    k_list : List[int]
        评估的 K 值列表。
    metrics : List[str], optional
        要计算的指标列表。

    Returns
    -------
    RankingMetricsResult
        排序评估结果。
    """
    if len(group_ids) != len(y_true_list):
        raise ValueError(f"group_ids 与 y_true_list 长度不一致: {len(group_ids)} vs {len(y_true_list)}")
    if len(group_ids) != len(y_score_list):
        raise ValueError(f"group_ids 与 y_score_list 长度不一致: {len(group_ids)} vs {len(y_score_list)}")
    if len(group_ids) != len(candidate_ids_list):
        raise ValueError(f"group_ids 与 candidate_ids_list 长度不一致: {len(group_ids)} vs {len(candidate_ids_list)}")

    result = RankingMetricsResult(k_list=k_list)
    global_warnings: List[str] = []

    # 逐组评估
    group_results: List[GroupRankingResult] = []
    skipped_groups: List[Any] = []

    for i, (gid, y_true, y_score, candidate_ids) in enumerate(
        zip(group_ids, y_true_list, y_score_list, candidate_ids_list)
    ):
        group_result = evaluate_single_group(
            y_true=y_true,
            y_score=y_score,
            candidate_ids=candidate_ids,
            k_list=k_list,
            metrics=metrics,
            group_id=gid,
        )
        group_results.append(group_result)

        if "no_relevant_items" in group_result.warnings:
            skipped_groups.append(gid)

    result.group_metrics = group_results

    if skipped_groups:
        global_warnings.append(f"skipped_empty_groups: {len(skipped_groups)} groups with no relevant items")

    # 跨组聚合
    # 收集所有有效组的指标
    metric_values: Dict[str, List[float]] = {}

    for gr in group_results:
        if not gr.metrics:
            continue
        for key, value in gr.metrics.items():
            if key not in metric_values:
                metric_values[key] = []
            metric_values[key].append(value)

    # 计算平均值
    for key, values in metric_values.items():
        if values:
            result.summary_metrics[key] = float(np.mean(values))

    # 元信息
    result.metadata = {
        "task_type": "ranking",
        "num_groups": len(group_ids),
        "num_valid_groups": len([gr for gr in group_results if gr.metrics]),
        "num_skipped_groups": len(skipped_groups),
        "k_list": k_list,
        "metrics_computed": list(result.summary_metrics.keys()),
    }

    result.warnings = global_warnings

    return result


def evaluate_ranking_from_bundle(
    bundle,
    k_list: Optional[List[int]] = None,
    metrics: Optional[List[str]] = None,
) -> RankingMetricsResult:
    """从 PredictionBundle 评估排序任务。

    Parameters
    ----------
    bundle : PredictionBundle
        预测产物。
    k_list : List[int], optional
        评估的 K 值列表。默认使用 bundle.k_list 或 [5, 10, 20]。
    metrics : List[str], optional
        要计算的指标列表。

    Returns
    -------
    RankingMetricsResult
        排序评估结果。
    """
    # 校验 bundle
    errors = bundle.validate()
    if errors:
        raise ValueError(f"PredictionBundle 校验失败: {errors}")

    if bundle.task_type != "ranking":
        raise ValueError(f"task_type 必须是 'ranking'，当前为 '{bundle.task_type}'")

    if bundle.group_ids is None:
        raise ValueError("ranking 任务必须提供 group_ids")

    # 默认 k_list
    if k_list is None:
        k_list = bundle.k_list if bundle.k_list else [5, 10, 20]

    # 按组聚合数据
    group_data: Dict[Any, Dict[str, Any]] = {}

    for i, gid in enumerate(bundle.group_ids):
        if gid not in group_data:
            group_data[gid] = {
                "y_true": [],
                "y_score": [],
                "candidate_ids": [],
            }

        group_data[gid]["y_true"].append(bundle.y_true[i] if bundle.y_true else 0)
        group_data[gid]["y_score"].append(bundle.y_score[i] if bundle.y_score else 0.0)

        if bundle.candidate_ids:
            group_data[gid]["candidate_ids"].append(bundle.candidate_ids[i])

    # 准备评估输入
    group_ids = list(group_data.keys())
    y_true_list = []
    y_score_list = []
    candidate_ids_list = []

    for gid in group_ids:
        data = group_data[gid]
        y_true_list.append(data["y_true"])
        y_score_list.append(np.array(data["y_score"]))

        if data["candidate_ids"]:
            candidate_ids_list.append(data["candidate_ids"])
        else:
            # 如果没有 candidate_ids，使用索引
            candidate_ids_list.append(list(range(len(data["y_score"]))))

    return evaluate_ranking(
        group_ids=group_ids,
        y_true_list=y_true_list,
        y_score_list=y_score_list,
        candidate_ids_list=candidate_ids_list,
        k_list=k_list,
        metrics=metrics,
    )
