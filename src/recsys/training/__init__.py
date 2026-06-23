"""Training framework — 训练基础设施层统一导出面。

导出的公共 API：
    - LightningRecommender, TrainerFactory, create_trainer  — 训练主入口
    - build_optimizer, OptimizerConfig, build_param_groups   — optimizer 工厂
    - build_scheduler, SchedulerConfig, SchedulerOutput       — scheduler 工厂
    - build_callbacks                                          — callback 组装
    - get_loss, list_losses                                    — loss 工厂
    - resolve_strategy, StrategyConfig                         — 分布式策略
"""

from recsys.training.callbacks import (  # noqa: F401
    GradientNormMonitor,
    MemoryMonitor,
    RunSummaryCallback,
    build_callbacks,
)
from recsys.training.distributed import (  # noqa: F401
    StrategyConfig,
    check_distributed_available,
    get_strategy_kwargs,
    resolve_strategy,
)
from recsys.training.losses import (  # noqa: F401
    get_loss,
    list_losses,
)
from recsys.training.optimizers import (  # noqa: F401
    OptimizerConfig,
    build_dual_optimizers,
    build_optimizer,
    build_param_groups,
    get_optimizer,
    list_optimizers,
)
from recsys.training.schedulers import (  # noqa: F401
    SchedulerConfig,
    SchedulerOutput,
    build_scheduler,
    build_warmup_scheduler,
    get_scheduler,
    list_schedulers,
)
from recsys.training.trainer import (  # noqa: F401
    LightningRecommender,
    TrainerFactory,
    create_trainer,
)

__all__ = [
    "GradientNormMonitor",
    "MemoryMonitor",
    "RunSummaryCallback",
    "build_callbacks",
    "StrategyConfig",
    "check_distributed_available",
    "get_strategy_kwargs",
    "resolve_strategy",
    "get_loss",
    "list_losses",
    "OptimizerConfig",
    "build_dual_optimizers",
    "build_optimizer",
    "build_param_groups",
    "get_optimizer",
    "list_optimizers",
    "SchedulerConfig",
    "SchedulerOutput",
    "build_scheduler",
    "build_warmup_scheduler",
    "get_scheduler",
    "list_schedulers",
    "LightningRecommender",
    "TrainerFactory",
    "create_trainer",
]
