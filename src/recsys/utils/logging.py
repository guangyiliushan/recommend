"""Unified logging with loguru + TensorBoard/WandB integration.

Key design:
- Explicit setup_logging() called by pipeline, idempotent
- loguru for terminal + file, bridge to standard logging
- Optional tracker (TensorBoard/W&B) via runtime.track_with
- Structured context injection (run_id, dataset, model, seed)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# 错误定义
# ---------------------------------------------------------------------------

class LoggingError(Exception):
    """日志系统错误。"""

    def __init__(
        self,
        message: str,
        code: str = "LOG_SETUP_ERROR",
        phase: str = "runtime",
        hint: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.phase = phase
        self.hint = hint


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class LoggingContext:
    """日志上下文，记录已初始化的日志路径和追踪信息。"""

    run_id: str
    log_file: str
    stderr_file: str = ""
    tracker_type: Optional[str] = None  # "tensorboard" / "wandb" / None
    tracker_dir: Optional[str] = None
    _initialized: bool = False


# ---------------------------------------------------------------------------
# 状态管理
# ---------------------------------------------------------------------------

_ctx: Optional[LoggingContext] = None
_intercept_id: Optional[int] = None


def _is_setup() -> bool:
    """检查日志系统是否已初始化。"""
    return _ctx is not None and _ctx._initialized


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def setup_logging(
    run_id: str,
    output_dir: str,
    log_level: str = "INFO",
    track_with: Optional[str] = None,
) -> LoggingContext:
    """初始化统一日志系统（幂等：重复调用不会重复添加 handler）。

    Parameters
    ----------
    run_id : str
        实验标识。
    output_dir : str
        输出根目录（日志将写到 output_dir/experiments/{run_id}/logs/）。
    log_level : str
        日志级别，默认 "INFO"。
    track_with : str, optional
        追踪后端，``"tensorboard"`` 或 ``"wandb"``。

    Returns
    -------
    LoggingContext

    Raises
    ------
    LoggingError
        日志目录不可写或追踪器初始化失败。
    """
    global _ctx, _intercept_id

    # 幂等：已初始化则返回现有上下文
    if _is_setup() and _ctx is not None:
        return _ctx

    # 创建日志目录
    log_dir = Path(output_dir) / "experiments" / run_id / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise LoggingError(
            f"无法创建日志目录: {log_dir}, {e}",
            code="LOG_DIR_NOT_WRITABLE",
            hint="检查 output_root 权限",
        ) from e

    log_file = str(log_dir / "run.log")
    stderr_file = str(log_dir / "stderr.log")

    # 移除已有 handler（loguru 默认 handler）
    logger.remove()

    # 终端 handler：可读格式
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
        enqueue=True,
    )

    # 文件 handler：完整格式 + 结构化上下文
    logger.add(
        log_file,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{extra[run_id]} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        enqueue=True,
        encoding="utf-8",
    )

    # stderr 文件
    logger.add(
        stderr_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        level="WARNING",
        enqueue=True,
        encoding="utf-8",
    )

    # 桥接标准 logging
    _bridge_standard_logging()

    # 可选追踪器
    tracker_type = None
    tracker_dir = None
    if track_with:
        try:
            tracker_type, tracker_dir = _init_tracker(track_with, output_dir, run_id)
        except Exception as e:
            logger.warning(f"追踪器初始化失败，降级为仅文件日志: {e}")

    _ctx = LoggingContext(
        run_id=run_id,
        log_file=log_file,
        stderr_file=stderr_file,
        tracker_type=tracker_type,
        tracker_dir=tracker_dir,
        _initialized=True,
    )

    # 注入结构化上下文
    logger.configure(extra={"run_id": run_id})

    logger.info(f"日志系统初始化完成 | run_id={run_id} | level={log_level}")
    logger.info(f"日志文件: {log_file}")
    if tracker_type:
        logger.info(f"追踪器: {tracker_type} -> {tracker_dir}")

    return _ctx


def get_logger(name: str = "recsys"):
    """获取 logger 实例（兼容标准 logging 接口）。

    Parameters
    ----------
    name : str
        Logger 名称。

    Returns
    -------
    loguru.Logger
    """
    return logger.bind(name=name)


def log_experiment_summary(
    config: Any,
    metrics: Dict[str, float],
    status: str,
) -> None:
    """记录实验摘要到日志。

    Parameters
    ----------
    config : Any
        实验配置对象。
    metrics : Dict[str, float]
        主指标字典。
    status : str
        实验状态。
    """
    logger.info("=" * 60)
    logger.info("实验摘要")
    logger.info("=" * 60)
    logger.info(f"  状态: {status}")
    logger.info(f"  数据集: {config.data.name}")
    logger.info(f"  模型: {config.model.name} ({config.model.family})")
    logger.info(f"  Seed: {config.runtime.seed}")

    if metrics:
        logger.info("  指标:")
        for k, v in metrics.items():
            logger.info(f"    {k}: {v:.6f}")

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

class _LoguruInterceptHandler(logging.Handler):
    """将标准 logging 消息转发到 loguru。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _bridge_standard_logging() -> None:
    """桥接标准 logging 到 loguru。"""
    global _intercept_id

    if _intercept_id is None:
        handler = _LoguruInterceptHandler()
        logging.basicConfig(handlers=[handler], level=0, force=True)

        # 截获所有标准 logging logger
        for name in logging.root.manager.loggerDict:
            std_logger = logging.getLogger(name)
            std_logger.handlers = []
            std_logger.propagate = True

        _intercept_id = id(handler)


def _init_tracker(
    track_with: str,
    output_dir: str,
    run_id: str,
) -> tuple:
    """初始化追踪器（TensorBoard 或 WandB）。

    Returns (tracker_type, tracker_dir)。
    """
    tracker_dir = str(Path(output_dir) / "experiments" / run_id / "tracker")

    if track_with == "tensorboard":
        _init_tensorboard(tracker_dir)
        return "tensorboard", tracker_dir

    if track_with == "wandb":
        _init_wandb(run_id, tracker_dir)
        return "wandb", tracker_dir

    raise LoggingError(
        f"不支持的追踪器: {track_with}",
        code="TRACKER_INIT_FAILED",
        hint="支持的值: tensorboard, wandb",
    )


def _init_tensorboard(log_dir: str) -> None:
    """初始化 TensorBoard SummaryWriter。"""
    try:
        from torch.utils.tensorboard import SummaryWriter
        SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard 已启动 -> {log_dir}")
    except ImportError as e:
        raise LoggingError(
            "TensorBoard 未安装",
            code="TRACKER_INIT_FAILED",
            hint="pip install tensorboard 后重试，或设置 runtime.track_with=null",
        ) from e


def _init_wandb(run_id: str, log_dir: str) -> None:
    """初始化 WandB。"""
    try:
        import wandb
        wandb.init(
            name=run_id,
            dir=log_dir,
            reinit=True,
        )
        logger.info(f"WandB 已启动 -> {log_dir}")
    except ImportError as e:
        raise LoggingError(
            "WandB 未安装",
            code="TRACKER_INIT_FAILED",
            hint="pip install wandb 后重试，或设置 runtime.track_with=null",
        ) from e
