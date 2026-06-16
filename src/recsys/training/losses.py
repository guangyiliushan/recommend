"""Loss function library — 可复用损失函数与 LOSS_REGISTRY 工厂。

三层结构：
    - 基础 loss 封装（BCE / CrossEntropy / BCEWithLogits 的薄包装）
    - 项目自定义 loss（BPR / InfoNCE / Focal / MultiTask 等）
    - loss 工厂（get_loss + LOSS_REGISTRY）

职责边界：
    - 只提供 loss callable，给 model.compute_loss() 或 trainer 使用
    - 不做 task routing / threshold / metric 逻辑
    - 不依赖具体模型家族、PredictionBundle 或 dataset schema
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from recsys.core.registry import LOSS_REGISTRY

# ============================================================================
# 基础 loss 薄包装（直接复用 torch，提供统一注册名）
# ============================================================================

class BCELossWrapper(nn.Module):
    """BCELoss 薄包装 — binary classification (CTR/CVR)。

    Parameters
    ----------
    reduction : str
        归约方式，默认 "mean"。
    """

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        super().__init__()
        self._loss = nn.BCELoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(input, target)


class BCEWithLogitsLossWrapper(nn.Module):
    """BCEWithLogitsLoss 薄包装 — binary classification with numerical stability。

    Parameters
    ----------
    reduction : str
        归约方式，默认 "mean"。
    pos_weight : torch.Tensor, optional
        正类权重。
    """

    def __init__(
        self,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=pos_weight
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(input, target)


class CrossEntropyWrapper(nn.Module):
    """CrossEntropyLoss 薄包装 — multi-class。

    Parameters
    ----------
    reduction : str
        归约方式，默认 "mean"。
    weight : torch.Tensor, optional
        类别权重。
    label_smoothing : float
        标签平滑，默认 0.0。
    """

    def __init__(
        self,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._loss = nn.CrossEntropyLoss(
            reduction=reduction, weight=weight, label_smoothing=label_smoothing
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(input, target)


# ============================================================================
# 项目自定义 loss
# ============================================================================

class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss。

    适用场景：隐式反馈排序任务，每个 (user, pos_item, neg_item) 三元组。

    Parameters
    ----------
    reduction : str
        归约方式，默认 "mean"。
    """

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        super().__init__()
        self._reduction = reduction

    def forward(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
    ) -> torch.Tensor:
        """计算 BPR loss。

        Parameters
        ----------
        pos_score : torch.Tensor
            正样本分数。
        neg_score : torch.Tensor
            负样本分数。

        Returns
        -------
        torch.Tensor
        """
        diff = pos_score - neg_score
        loss = -F.logsigmoid(diff)
        if self._reduction == "mean":
            return loss.mean()
        elif self._reduction == "sum":
            return loss.sum()
        return loss


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss。

    适用场景：对比学习，anchor 与 positive 更近，与 negatives 更远。

    Parameters
    ----------
    temperature : float
        温度参数，默认 0.07。
    reduction : str
        归约方式，默认 "mean"。
    """

    def __init__(
        self, temperature: float = 0.07, reduction: str = "mean", **kwargs: Any
    ) -> None:
        super().__init__()
        self._temperature = temperature
        self._reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算 InfoNCE loss。

        Parameters
        ----------
        anchor : torch.Tensor
            锚点 embedding，shape (B, D)。
        positive : torch.Tensor
            正样本 embedding，shape (B, D) 或 (B, 1, D)。
        negatives : torch.Tensor, optional
            负样本 embedding，shape (B, N_neg, D)。

        Returns
        -------
        torch.Tensor
        """
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        # 正样本 logit
        if positive.dim() == 2:
            pos_logit = (anchor * positive).sum(dim=-1, keepdim=True)  # (B, 1)
        else:
            pos_logit = torch.bmm(
                anchor.unsqueeze(1), positive.transpose(1, 2)
            ).squeeze(1)  # (B,)

        pos_logit = pos_logit / self._temperature

        if negatives is not None:
            negatives = F.normalize(negatives, dim=-1)
            neg_logit = torch.bmm(
                anchor.unsqueeze(1), negatives.transpose(1, 2)
            ).squeeze(1)
            neg_logit = neg_logit / self._temperature
            logits = torch.cat([pos_logit, neg_logit], dim=1)
        else:
            logits = pos_logit

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels, reduction=self._reduction)
        return loss


class TOP1Loss(nn.Module):
    """TOP1 loss — session-based recommendation。

    适用场景：GRU4Rec 等 session 推荐模型。

    Parameters
    ----------
    reduction : str
        归约方式，默认 "mean"。
    """

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        super().__init__()
        self._reduction = reduction

    def forward(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """计算 TOP1 loss。

        Parameters
        ----------
        scores : torch.Tensor
            模型输出分数，shape (B, N_items)。
        targets : torch.Tensor
            目标 one-hot 或索引，shape (B,) 或 (B, N_items)。

        Returns
        -------
        torch.Tensor
        """
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=scores.shape[1]).float()

        pos_scores = (scores * targets).sum(dim=-1, keepdim=True)  # (B, 1)
        diff = scores - pos_scores
        loss = torch.sigmoid(diff) + torch.sigmoid(scores**2)
        # 对正样本位置不计算 loss
        loss = loss * (1 - targets)
        loss = loss.sum(dim=-1)

        if self._reduction == "mean":
            return loss.mean()
        elif self._reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Focal loss — 缓解类别不平衡。

    参数
    ----
    alpha : float
        正类权重因子，默认 0.25。
    gamma : float
        聚焦参数，默认 2.0。
    reduction : str
        归约方式，默认 "mean"。
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 Focal loss。

        Parameters
        ----------
        input : torch.Tensor
            概率输出，shape (N, ...)。
        target : torch.Tensor
            目标标签，shape (N, ...)。

        Returns
        -------
        torch.Tensor
        """
        ce_loss = F.binary_cross_entropy(input, target, reduction="none")
        pt = torch.where(target == 1, input, 1 - input)
        focal_weight = (1 - pt) ** self._gamma
        alpha_weight = torch.where(
            target == 1, self._alpha, 1 - self._alpha
        )
        loss = alpha_weight * focal_weight * ce_loss
        if self._reduction == "mean":
            return loss.mean()
        elif self._reduction == "sum":
            return loss.sum()
        return loss


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sigmoid Focal Loss — 函数式版本，直接接受 logits。

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    设计来源：dist/utils.py 的 sigmoid_focal_loss。

    Parameters
    ----------
    logits : torch.Tensor
        原始 logits（sigmoid 前），shape (N,)。
    targets : torch.Tensor
        二分类标签 {0, 1}，shape (N,)。
    alpha : float
        正类权重，范围 (0, 1)。当正样本占主导时，使用 alpha < 0.5 降低正类权重。
    gamma : float
        聚焦参数。gamma=0 退化为标准 BCE；gamma=2 是常用值。
    reduction : str
        归约方式：'mean' | 'sum' | 'none'。

    Returns
    -------
    torch.Tensor
        损失值。
    """
    p = torch.sigmoid(logits)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class MultiTaskLoss(nn.Module):
    """Multi-task weighted loss — 多任务加权损失。

    支持两种模式：
    - weighted_sum: 固定权重加权和
    - uncertainty: 同方差不确定性加权（learnable log_vars）

    Parameters
    ----------
    mode : str
        加权模式："weighted_sum" 或 "uncertainty"。
    task_weights : Dict[str, float], optional
        各任务固定权重，仅 weighted_sum 模式使用。
    num_tasks : int, optional
        任务数量，仅 uncertainty 模式使用。
    """

    def __init__(
        self,
        mode: str = "weighted_sum",
        task_weights: Optional[Dict[str, float]] = None,
        num_tasks: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if mode not in ("weighted_sum", "uncertainty"):
            raise ValueError(
                f"MultiTaskLoss mode 必须是 'weighted_sum' 或 'uncertainty'，"
                f"当前: '{mode}'"
            )
        self._mode = mode
        self._task_weights = task_weights or {}

        if mode == "uncertainty" and num_tasks > 0:
            self._log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        loss_dict: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """聚合多任务损失。

        Parameters
        ----------
        loss_dict : Dict[str, torch.Tensor]
            各任务损失字典。
        weights : Dict[str, float], optional
            运行时权重覆盖。

        Returns
        -------
        torch.Tensor
            总损失。
        """
        if self._mode == "weighted_sum":
            effective_weights = weights or self._task_weights
            total = torch.tensor(0.0, device=next(iter(loss_dict.values())).device)
            for task_name, loss_val in loss_dict.items():
                w = effective_weights.get(task_name, 1.0)
                total = total + w * loss_val
            return total

        # uncertainty mode
        total = torch.tensor(0.0, device=next(iter(loss_dict.values())).device)
        for i, (_task_name, loss_val) in enumerate(loss_dict.items()):
            precision = torch.exp(-self._log_vars[i])
            total = total + precision * loss_val + self._log_vars[i]
        return total


class AdaptiveHuberLoss(nn.Module):
    """Adaptive Huber loss — 鲁棒回归。

    在 |error| <= delta 时表现为 MSE，否则为 MAE。

    Parameters
    ----------
    delta : float
        阈值参数，默认 1.0。
    reduction : str
        归约方式，默认 "mean"。
    """

    def __init__(
        self, delta: float = 1.0, reduction: str = "mean", **kwargs: Any
    ) -> None:
        super().__init__()
        self._loss = nn.HuberLoss(reduction=reduction, delta=delta)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(input, target)


# ============================================================================
# 注册到 LOSS_REGISTRY
# ============================================================================

_LOSS_MAP: Dict[str, Type[nn.Module]] = {
    "bce": BCELossWrapper,
    "bce_with_logits": BCEWithLogitsLossWrapper,
    "cross_entropy": CrossEntropyWrapper,
    "bpr": BPRLoss,
    "info_nce": InfoNCELoss,
    "top1": TOP1Loss,
    "focal": FocalLoss,
    "multi_task": MultiTaskLoss,
    "adaptive_huber": AdaptiveHuberLoss,
}


def _register_all() -> None:
    """将 losses.py 中所有 loss 注册到 LOSS_REGISTRY。

    幂等：已注册项会跳过。
    """
    for name, cls in _LOSS_MAP.items():
        if name not in LOSS_REGISTRY:
            LOSS_REGISTRY.register(name)(cls)


_register_all()


# ============================================================================
# 公共工厂 API
# ============================================================================

def get_loss(name: str, **kwargs: Any) -> nn.Module:
    """从 LOSS_REGISTRY 获取 loss 实例。

    Parameters
    ----------
    name : str
        loss 名称，如 "bce"、"bpr"、"focal"。
    **kwargs : Any
        传递给 loss 构造函数的参数。

    Returns
    -------
    nn.Module
        loss 模块实例。

    Raises
    ------
    KeyError
        loss 名称未注册。
    """
    cls = LOSS_REGISTRY.get(name)
    return cls(**kwargs)


def list_losses() -> list:
    """列出所有已注册 loss 名称。"""
    return LOSS_REGISTRY.list()
