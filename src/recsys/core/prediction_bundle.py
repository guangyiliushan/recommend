"""PredictionBundle — 统一预测产物契约。

模型层与评估层之间的唯一数据契约。
所有模型（训练式或非训练式）的 predict() 输出必须遵守此结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PredictionBundle:
    """模型预测结果的标准化容器。

    Pipeline 消费此结构后可安全转发至 Evaluator，
    无需感知具体模型内部实现细节。

    Attributes
    ----------
    task_type : str
        任务类型：``"pointwise"`` / ``"ranking"`` / ``"multitask"``。
    problem_type : str
        问题类型：``"binary"`` / ``"multiclass"`` / ``"multilabel"`` /
        ``"regression"`` / ``"implicit_ranking"`` / ``"listwise_ranking"``。
    y_true : list
        真实标签列表（pointwise: 标量；ranking: 相关性标签；
        multitask 模式下可为空列表）。
    y_score : list
        模型输出的连续分数列表。
    y_pred : list, optional
        阈值化后的离散预测（pointwise 可选）。
    group_ids : list, optional
        ranking 任务的用户/请求分组标识（必须提供）。
    candidate_ids : list, optional
        ranking 任务的推荐候选项 ID 列表。
    pos_label : int, optional
        二分类正类标签值，默认 ``1``。多分类时忽略。
    class_names : list, optional
        多分类任务的类别名称列表。
    score_type : str
        分数类型：``"prob"``（概率）、``"logit"``（原始 logit）、
        ``"raw_score"``（未归一化得分，如相似度）。
    thresholds : list, optional
        pointwise 任务的决策阈值列表。
    sample_weight : list, optional
        样本权重，用于加权评估。长度应与 y_true 一致。
    task_outputs : dict, optional
        multitask 任务各任务头的输出字典。
    task_labels : dict, optional
        multitask 任务各任务头的标签字典。
    task_masks : dict, optional
        multitask 任务各任务头的有效样本 mask。
    k_list : list, optional
        ranking 评估的 Top-K 列表，如 ``[5, 10, 20]``。
    metadata : dict
        附加元信息（dataset_name, model_name, sample_count 等）。
    """

    task_type: str
    problem_type: str = "binary"
    y_true: list = field(default_factory=list)
    y_score: list = field(default_factory=list)
    y_pred: Optional[list] = None
    group_ids: Optional[list] = None
    candidate_ids: Optional[list] = None
    pos_label: Optional[int] = 1
    class_names: Optional[List[str]] = None
    score_type: str = "prob"
    thresholds: Optional[List[float]] = None
    sample_weight: Optional[list] = None
    task_outputs: Optional[Dict[str, list]] = None
    task_labels: Optional[Dict[str, list]] = None
    task_masks: Optional[Dict[str, list]] = None
    k_list: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """校验 bundle 字段是否符合当前 task_type / problem_type 的最低要求。

        Returns
        -------
        list[str]
            校验错误信息列表，空列表表示通过。
        """
        errors: List[str] = []

        if not self.task_type:
            errors.append("task_type 不能为空")
        elif self.task_type not in {"pointwise", "ranking", "multitask"}:
            errors.append(
                f"task_type 必须是 'pointwise'/'ranking'/'multitask'，"
                f"当前为 '{self.task_type}'"
            )

        _VALID_PROBLEM_TYPES = {  # noqa: N806
            "binary", "multiclass", "multilabel",
            "regression", "implicit_ranking", "listwise_ranking",
        }
        if self.problem_type not in _VALID_PROBLEM_TYPES:
            errors.append(
                f"problem_type 无效 '{self.problem_type}'，"
                f"合法值: {sorted(_VALID_PROBLEM_TYPES)}"
            )

        if self.score_type not in {"prob", "logit", "raw_score"}:
            errors.append(
                f"score_type 必须是 'prob'/'logit'/'raw_score'，"
                f"当前为 '{self.score_type}'"
            )

        if self.task_type == "multitask":
            if self.task_outputs is None or self.task_labels is None:
                errors.append("multitask 任务必须提供 task_outputs 和 task_labels")
        else:
            if not self.y_true:
                errors.append("y_true 不能为空")
            if not self.y_score:
                errors.append("y_score 不能为空")
            if len(self.y_true) != len(self.y_score):
                errors.append(
                    f"y_true 与 y_score 长度不一致 "
                    f"({len(self.y_true)} vs {len(self.y_score)})"
                )

        if self.task_type == "ranking" and self.group_ids is None:
            errors.append("ranking 任务必须提供 group_ids")

        return errors

    @property
    def num_samples(self) -> int:
        """样本数量。"""
        return len(self.y_true)

    @property
    def num_groups(self) -> int:
        """分组数量（仅 ranking 有效）。"""
        if self.group_ids is None:
            return 0
        return len(set(self.group_ids))
