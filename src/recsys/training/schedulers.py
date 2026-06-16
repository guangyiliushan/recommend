"""Learning rate scheduler factory — 构建 scheduler 实例与调度元数据。

支持：
    - cosine / cosine_warmup
    - step / multi_step
    - plateau (ReduceLROnPlateau)
    - onecycle
    - polynomial

关键设计：
    - 统一返回 SchedulerOutput(scheduler, scheduler_type, monitor)
    - scheduler_type 区分 "step" / "epoch" / "plateau" 三种语义
    - warmup 显式化为独立构建函数

职责边界：
    - 输入：optimizer + 配置
    - 输出：SchedulerOutput
    - 不做 optimizer 逻辑、设备/分布式决策、PredictionBundle 操作
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
    _LRScheduler,
)

# ============================================================================
# 配置与输出结构
# ============================================================================

@dataclass
class SchedulerConfig:
    """Scheduler 配置 — 与 TrainingConfig / Hydra 对齐。

    Attributes
    ----------
    name : str
        scheduler 名称。
    warmup_epochs : int
        warmup epoch 数。
    warmup_steps : int
        warmup step 数（优先级高于 warmup_epochs）。
    total_steps : int, optional
        总训练步数，onecycle / cosine_warmup 需要。
    monitor : str, optional
        ReduceLROnPlateau 监控指标名。
    patience : int
        plateau patience。
    factor : float
        plateau / step 衰减因子。
    milestones : List[int], optional
        MultiStepLR 的里程碑 epoch。
    gamma : float
        MultiStepLR / StepLR 衰减率。
    step_size : int
        StepLR step size。
    eta_min : float
        CosineAnnealing 最小学习率。
    max_lr : float or None
        OneCycleLR 的最大学习率。
    pct_start : float
        OneCycleLR 上升比例。
    power : float
        polynomial 幂次。
    """

    name: str = "cosine"
    warmup_epochs: int = 0
    warmup_steps: int = 0
    total_steps: Optional[int] = None
    monitor: str = "val_loss"
    patience: int = 5
    factor: float = 0.1
    milestones: Optional[List[int]] = None
    gamma: float = 0.1
    step_size: int = 10
    eta_min: float = 0.0
    max_lr: Optional[float] = None
    pct_start: float = 0.3
    power: float = 1.0


@dataclass
class SchedulerOutput:
    """Scheduler 工厂统一返回值。

    scheduler_type 语义：
    - "step": 每个 step 后调用 scheduler.step()（大多数 scheduler）
    - "epoch": 每个 epoch 后调用 scheduler.step()（Lightning 默认）
    - "plateau": 需要 monitor 指标（ReduceLROnPlateau）

    Attributes
    ----------
    scheduler : _LRScheduler
        scheduler 实例。
    scheduler_type : str
        调度类型："step" / "epoch" / "plateau"。
    monitor : str or None
        plateau scheduler 的监控指标名，非 plateau 时为 None。
    interval : str
        Lightning 兼容："step" 或 "epoch"。
    frequency : int
        调度频率。
    """

    scheduler: _LRScheduler
    scheduler_type: str = "epoch"
    monitor: Optional[str] = None
    interval: str = "epoch"
    frequency: int = 1

    def __post_init__(self) -> None:
        if self.scheduler_type == "step":
            self.interval = "step"
        elif self.scheduler_type == "plateau":
            self.interval = "epoch"
        else:
            self.interval = "epoch"


# ============================================================================
# 内部工厂表
# ============================================================================

def _build_cosine(
    optimizer: Optimizer,
    config: SchedulerConfig,
    total_steps: Optional[int] = None,
) -> SchedulerOutput:
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps or config.total_steps or 100,
        eta_min=config.eta_min,
    )
    return SchedulerOutput(scheduler=scheduler, scheduler_type="epoch")


def _build_cosine_warmup(
    optimizer: Optimizer,
    config: SchedulerConfig,
    total_steps: Optional[int] = None,
) -> SchedulerOutput:
    total = total_steps or config.total_steps
    warmup = config.warmup_steps or config.warmup_epochs

    if total is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=config.eta_min)
        if warmup > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup],
            )
        return SchedulerOutput(scheduler=scheduler, scheduler_type="epoch")

    remaining = total - warmup
    after = CosineAnnealingLR(optimizer, T_max=max(1, remaining), eta_min=config.eta_min)
    if warmup > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, after],
            milestones=[warmup],
        )
    else:
        scheduler = after
    return SchedulerOutput(scheduler=scheduler, scheduler_type="epoch")


def _build_step(
    optimizer: Optimizer,
    config: SchedulerConfig,
    total_steps: Optional[int] = None,
) -> SchedulerOutput:
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    return SchedulerOutput(scheduler=scheduler, scheduler_type="epoch")


def _build_multi_step(
    optimizer: Optimizer,
    config: SchedulerConfig,
    total_steps: Optional[int] = None,
) -> SchedulerOutput:
    milestones = config.milestones or [30, 60, 90]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=config.gamma)
    return SchedulerOutput(scheduler=scheduler, scheduler_type="epoch")


def _build_plateau(
    optimizer: Optimizer,
    config: SchedulerConfig,
    total_steps: Optional[int] = None,
) -> SchedulerOutput:
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.factor,
        patience=config.patience,
        min_lr=0.0,
    )
    return SchedulerOutput(
        scheduler=scheduler,
        scheduler_type="plateau",
        monitor=config.monitor,
    )


def _build_onecycle(
    optimizer: Optimizer,
    config: SchedulerConfig,
    total_steps: Optional[int] = None,
) -> SchedulerOutput:
    total = total_steps or config.total_steps
    if total is None:
        raise ValueError(
            "OneCycleLR 需要 total_steps 或 config.total_steps。"
            "请通过 build_scheduler(total_steps=...) 传入，"
            "或在 SchedulerConfig 中设置 total_steps。"
        )
    max_lr = config.max_lr or config.eta_min or 1e-3
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total,
        pct_start=config.pct_start,
        anneal_strategy="cos",
    )
    return SchedulerOutput(scheduler=scheduler, scheduler_type="step")


def _build_polynomial(
    optimizer: Optimizer,
    config: SchedulerConfig,
    total_steps: Optional[int] = None,
) -> SchedulerOutput:
    total = total_steps or config.total_steps or 100
    scheduler = PolynomialLR(optimizer, total_iters=total, power=config.power)
    return SchedulerOutput(scheduler=scheduler, scheduler_type="epoch")


_SCHEDULER_MAP = {
    "cosine": _build_cosine,
    "cosine_warmup": _build_cosine_warmup,
    "step": _build_step,
    "multi_step": _build_multi_step,
    "plateau": _build_plateau,
    "onecycle": _build_onecycle,
    "polynomial": _build_polynomial,
}


# ============================================================================
# 公共 API
# ============================================================================

def get_scheduler(
    name: str,
    optimizer: Optimizer,
    **kwargs: Any,
) -> SchedulerOutput:
    """按名称返回 scheduler。

    Parameters
    ----------
    name : str
        scheduler 名称。
    optimizer : Optimizer
        optimizer 实例。
    **kwargs : Any
        传递给底层构建函数的参数。

    Returns
    -------
    SchedulerOutput

    Raises
    ------
    ValueError
        不支持的 scheduler 名称。
    """
    builder = _SCHEDULER_MAP.get(name)
    if builder is None:
        raise ValueError(
            f"不支持的 scheduler: '{name}'。"
            f"可用: {sorted(_SCHEDULER_MAP.keys())}。"
        )
    config = SchedulerConfig(name=name, **kwargs)
    return builder(optimizer, config, total_steps=kwargs.get("total_steps"))


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    total_steps: Optional[int] = None,
) -> SchedulerOutput:
    """根据 SchedulerConfig 构建 scheduler。

    Parameters
    ----------
    optimizer : Optimizer
        optimizer 实例。
    config : SchedulerConfig
        scheduler 配置。
    total_steps : int, optional
        总训练步数。

    Returns
    -------
    SchedulerOutput
    """
    builder = _SCHEDULER_MAP.get(config.name)
    if builder is None:
        raise ValueError(
            f"不支持的 scheduler: '{config.name}'。"
            f"可用: {sorted(_SCHEDULER_MAP.keys())}。"
        )
    return builder(optimizer, config, total_steps=total_steps)


def build_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    after_scheduler: Optional[_LRScheduler] = None,
    warmup_start_factor: float = 0.01,
) -> _LRScheduler:
    """构建 warmup + 主 scheduler 组合。

    独立于 build_scheduler，便于在 trainer.configure_optimizers() 中灵活组合。

    Parameters
    ----------
    optimizer : Optimizer
        optimizer 实例。
    warmup_steps : int
        warmup 步数。
    after_scheduler : _LRScheduler, optional
        warmup 之后的主 scheduler。
    warmup_start_factor : float
        warmup 起始因子。

    Returns
    -------
    _LRScheduler
    """
    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    if after_scheduler is None:
        return warmup
    return SequentialLR(
        optimizer,
        schedulers=[warmup, after_scheduler],
        milestones=[warmup_steps],
    )


def list_schedulers() -> list:
    """列出所有已支持的 scheduler 名称。"""
    return sorted(_SCHEDULER_MAP.keys())
