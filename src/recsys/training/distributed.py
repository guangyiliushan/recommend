"""Distributed training strategy — 策略解析、兼容性检查与 Trainer strategy 参数生成。

当前阶段：只做策略解析与能力校验，不实现完整分布式训练流程。
第一阶段支持：auto / cpu / single-gpu / ddp。
FSDP / DeepSpeed 返回明确的未就绪错误。

职责边界：
    - 只处理训练执行策略
    - 不做 checkpoint 路径协议、日志目录协议、benchmark 并发策略
    - 不依赖 PredictionBundle、BaseDataset 内部实现、Registry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# ============================================================================
# 配置结构
# ============================================================================

@dataclass
class StrategyConfig:
    """分布式策略解析结果。

    Attributes
    ----------
    strategy : str
        Lightning strategy 字符串："auto" / "ddp" / "fsdp" / "deepspeed"。
    accelerator : str
        accelerator："cpu" / "gpu" / "auto"。
    devices : str or int
        设备数："auto" 或整数。
    num_nodes : int
        节点数。
    precision : str or None
        mixed precision："16-mixed" / "bf16-mixed" / None。
    backend : str or None
        后端，如 "nccl"。
    """

    strategy: str = "auto"
    accelerator: str = "auto"
    devices: str = "auto"
    num_nodes: int = 1
    precision: Optional[str] = None
    backend: Optional[str] = None


# ============================================================================
# 策略解析
# ============================================================================

def resolve_strategy(
    runtime_config: Dict[str, Any],
    training_config: Optional[Dict[str, Any]] = None,
) -> StrategyConfig:
    """把 runtime / training 配置映射成 Trainer 可消费的策略参数。

    映射规则：
    - device=cpu  → accelerator="cpu", devices=1, strategy="auto"
    - device=cuda & num_devices=1 → accelerator="gpu", devices=1
    - device=cuda & num_devices>1 → accelerator="gpu", devices=N, strategy="ddp"
    - device=auto → 自动检测 GPU 数量

    Parameters
    ----------
    runtime_config : Dict[str, Any]
        包含 device、num_devices、deterministic 等。
    training_config : Dict[str, Any], optional
        包含 mixed_precision、accumulate_grad_batches 等。

    Returns
    -------
    StrategyConfig
    """
    device = runtime_config.get("device", "auto")
    num_devices = runtime_config.get("num_devices", 1)
    deterministic = runtime_config.get("deterministic", False)

    training_config = training_config or {}
    mixed_precision = training_config.get("mixed_precision", None)

    config = StrategyConfig()

    # 设备解析
    if device == "cpu":
        config.accelerator = "cpu"
        config.devices = "1"
        config.strategy = "auto"
    elif device == "cuda" or device == "gpu":
        config.accelerator = "gpu"
        if num_devices > 1:
            config.devices = str(num_devices)
            config.strategy = _resolve_ddp_strategy()
        else:
            config.devices = "1"
            config.strategy = "auto"
    elif device == "mps":
        config.accelerator = "mps"
        config.devices = "1"
        config.strategy = "auto"
    else:  # "auto"
        config.accelerator = "auto"
        config.devices = "auto"
        config.strategy = "auto"

    # precision
    if mixed_precision:
        mp = mixed_precision.lower()
        if mp in ("fp16", "16", "16-mixed"):
            config.precision = "16-mixed"
        elif mp in ("bf16", "bf16-mixed"):
            config.precision = "bf16-mixed"
        elif mp in ("32", "32-true", "none", "no"):
            config.precision = None

    # deterministic
    if deterministic:
        config.backend = config.backend or "nccl"

    return config


def _resolve_ddp_strategy() -> str:
    """解析当前环境最佳 DDP 策略。"""
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return "ddp"
    return "ddp"


# ============================================================================
# 兼容性检查
# ============================================================================

def check_distributed_available(strategy: str) -> Tuple[bool, Optional[str]]:
    """检查当前环境是否支持指定分布式策略。

    Parameters
    ----------
    strategy : str
        策略名："ddp" / "fsdp" / "deepspeed"。

    Returns
    -------
    Tuple[bool, Optional[str]]
        (是否可用, 不可用时的错误信息)
    """
    strategy_lower = strategy.lower()

    if strategy_lower in ("auto",):
        return True, None

    if strategy_lower == "ddp":
        import torch
        if not torch.cuda.is_available():
            return False, "DDP 需要 CUDA 环境，当前未检测到 GPU。"
        if torch.cuda.device_count() < 2:
            return False, (
                f"DDP 需要至少 2 个 GPU，当前只有 {torch.cuda.device_count()} 个。"
            )
        return True, None

    if strategy_lower == "fsdp":
        return False, (
            "FSDP 尚未启用。"
            "计划在训练主干稳定后（第二阶段）逐步接入。"
            "当前建议使用 ddp 作为多卡方案。"
        )

    if strategy_lower == "deepspeed":
        try:
            import deepspeed  # noqa: F401
            return False, (
                "DeepSpeed 已安装但尚未在本项目中启用。"
                "计划在训练主干稳定后（第三阶段）逐步接入。"
            )
        except ImportError:
            return False, (
                "DeepSpeed 未安装且尚未在本项目中启用。"
                "如需使用，请先 pip install deepspeed。"
            )

    # 未知策略
    return False, (
        f"不支持的分布式策略: '{strategy}'。"
        f"当前支持: auto / ddp。"
        f"计划支持: fsdp / deepspeed。"
    )


# ============================================================================
# Trainer 参数字典生成
# ============================================================================

def get_strategy_kwargs(config: StrategyConfig) -> Dict[str, Any]:
    """生成 pl.Trainer 的 strategy / accelerator / devices 参数字典。

    Parameters
    ----------
    config : StrategyConfig
        策略配置。

    Returns
    -------
    Dict[str, Any]
        可直接传入 pl.Trainer(**kwargs) 的参数字典。
    """
    kwargs: Dict[str, Any] = {
        "accelerator": config.accelerator,
        "devices": config.devices,
        "strategy": config.strategy,
        "num_nodes": config.num_nodes,
    }

    if config.precision:
        kwargs["precision"] = config.precision

    return kwargs
