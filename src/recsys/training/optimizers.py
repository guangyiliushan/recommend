"""Optimizer factory — 构建 optimizer 实例与参数组策略。

支持：
    - adam / adamw / sgd / adagrad（稳定基础项）
    - lion / lamb（可选扩展）

职责边界：
    - 输入：模型参数 + 训练配置
    - 输出：optimizer 实例
    - 不做 scheduler、设备/分布式决策、PredictionBundle 操作
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch.nn as nn
from torch.optim import SGD, Adagrad, Adam, AdamW, Optimizer

# ============================================================================
# 配置结构
# ============================================================================

@dataclass
class OptimizerConfig:
    """Optimizer 配置 — 与 TrainingConfig / Hydra 对齐。

    Attributes
    ----------
    name : str
        optimizer 名称："adam" / "adamw" / "sgd" / "adagrad"。
    lr : float
        学习率。
    weight_decay : float
        权重衰减。
    betas : tuple
        Adam/AdamW 的 beta 参数。
    eps : float
        数值稳定参数。
    momentum : float
        SGD momentum。
    use_lion : bool
        是否使用 Lion。
    param_group_config : dict, optional
        参数组策略配置。
    """

    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9
    use_lion: bool = False
    param_group_config: Optional[Dict[str, Any]] = None
    sparse_lr: float = 0.05  # Adagrad 稀疏参数学习率（仅 dual 模式）


# ============================================================================
# 参数组策略
# ============================================================================

def build_param_groups(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    base_lr: float = 1e-3,
    base_weight_decay: float = 1e-5,
) -> List[Dict[str, Any]]:
    """构建参数组：默认组 + no-weight-decay 组 + embedding 特殊学习率组。

    策略通过显式配置开启，不做层名魔法推断。
    如果未提供配置，返回默认单组。

    Parameters
    ----------
    model : nn.Module
        模型实例。
    config : Dict[str, Any], optional
        参数组配置，支持字段：
        - no_decay_patterns : List[str] — 不施加 weight decay 的参数名子串。
        - embedding_lr_scale : float — embedding 学习率缩放因子。
        - embedding_patterns : List[str] — 识别 embedding 参数名的子串。
    base_lr : float
        默认学习率。
    base_weight_decay : float
        默认 weight decay。

    Returns
    -------
    List[Dict[str, Any]]
        PyTorch 参数组列表。
    """
    if config is None:
        return [{"params": model.parameters(), "lr": base_lr, "weight_decay": base_weight_decay}]

    no_decay_patterns = config.get("no_decay_patterns", [])
    emb_lr_scale = config.get("embedding_lr_scale", 1.0)
    emb_patterns = config.get("embedding_patterns", [])

    if not no_decay_patterns and emb_lr_scale == 1.0:
        return [{"params": model.parameters(), "lr": base_lr, "weight_decay": base_weight_decay}]

    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    emb_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # embedding 组
        is_emb = any(p in name for p in emb_patterns) if emb_patterns else False
        if is_emb and emb_lr_scale != 1.0:
            emb_params.append(param)
            continue

        # no-decay 组
        is_no_decay = any(p in name for p in no_decay_patterns)
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: List[Dict[str, Any]] = []
    if decay_params:
        groups.append({
            "params": decay_params,
            "lr": base_lr,
            "weight_decay": base_weight_decay,
        })
    if no_decay_params:
        groups.append({
            "params": no_decay_params,
            "lr": base_lr,
            "weight_decay": 0.0,
        })
    if emb_params:
        groups.append({
            "params": emb_params,
            "lr": base_lr * emb_lr_scale,
            "weight_decay": base_weight_decay,
        })

    # 兜底：所有参数在一个组
    if not groups:
        groups.append({
            "params": model.parameters(),
            "lr": base_lr,
            "weight_decay": base_weight_decay,
        })

    return groups


# ============================================================================
# optimizer 工厂
# ============================================================================

_OPTIMIZER_MAP = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
    "adagrad": Adagrad,
}


def get_optimizer(
    name: str,
    params: Iterable[nn.Parameter],
    **kwargs: Any,
) -> Optimizer:
    """按名称返回 optimizer 实例。

    仅支持 adam / adamw / sgd / adagrad 四个稳定基础项。

    Parameters
    ----------
    name : str
        optimizer 名称。
    params : Iterable[nn.Parameter]
        模型参数。
    **kwargs : Any
        传递给 optimizer 构造函数的参数。

    Returns
    -------
    Optimizer

    Raises
    ------
    ValueError
        不支持的 optimizer 名称。
    """
    cls = _OPTIMIZER_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"不支持的 optimizer: '{name}'。"
            f"可用: {sorted(_OPTIMIZER_MAP.keys())}。"
            f"如需 lion/lamb，请使用 build_optimizer() 并开启 use_lion。"
        )
    return cls(params, **kwargs)


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> Optimizer:
    """根据 OptimizerConfig 和模型参数构建 optimizer。

    支持参数组策略（通过 config.param_group_config 控制）。

    Parameters
    ----------
    model : nn.Module
        模型实例。
    config : OptimizerConfig
        optimizer 配置。

    Returns
    -------
    Optimizer
    """
    param_groups = build_param_groups(
        model,
        config=config.param_group_config,
        base_lr=config.lr,
        base_weight_decay=config.weight_decay,
    )

    if config.name in ("adam", "adamw"):
        if config.use_lion:
            try:
                from torch.optim import Lion  # type: ignore[attr-defined]
                return Lion(
                    param_groups,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    betas=config.betas,
                )
            except ImportError:
                pass  # fallback to adamw

        if config.name == "adam":
            return Adam(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
            )
        else:
            return AdamW(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
            )

    if config.name == "sgd":
        return SGD(
            param_groups,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    if config.name == "adagrad":
        return Adagrad(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    return get_optimizer(config.name, param_groups, lr=config.lr)


def list_optimizers() -> list:
    """列出所有已支持的 optimizer 名称。"""
    return sorted(_OPTIMIZER_MAP.keys())


def build_dual_optimizers(
    model: nn.Module,
    config: OptimizerConfig,
    sparse_lr: float = 0.05,
) -> Tuple[Adagrad, AdamW]:
    """构建 Adagrad(sparse) + AdamW(dense) 双优化器。

    检测 model.get_sparse_params() / get_dense_params()，
    分别构建 Adagrad 和 AdamW，返回二元组供 Lightning 多优化器使用。

    Parameters
    ----------
    model : nn.Module
        模型实例，需实现 get_sparse_params() 和 get_dense_params()。
    config : OptimizerConfig
        密集参数配置。
    sparse_lr : float
        稀疏参数学习率。

    Returns
    -------
    Tuple[Adagrad, AdamW]
        (稀疏参数 Adagrad 优化器, 密集参数 AdamW 优化器)
    """
    sparse_params = model.get_sparse_params()
    dense_params = model.get_dense_params()
    sparse_opt = Adagrad(sparse_params, lr=sparse_lr)
    dense_opt = AdamW(
        dense_params,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    return sparse_opt, dense_opt
