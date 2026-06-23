"""Training callbacks — 组装 Lightning 现成能力 + 项目级自定义监控。

设计原则：
    - 优先复用 Lightning 内建 Callback，不自己发明训练事件系统
    - 自定义 Callback 只保留少量项目真正需要的（监控、摘要）
    - 输出以日志和轻量 metadata 为主，不直接改写主 artifacts
    - 监控指标命名与 evaluator 保持一致：val_loss / val/roc_auc / val/ndcg@10

公共入口：build_callbacks(config, run_dir, monitor_metric) -> List[pl.Callback]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# 使用 pytorch_lightning 避免与 lightning 包名冲突
import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.callbacks.callback import Callback

# ============================================================================
# 自定义 Callback
# ============================================================================

class GradientNormMonitor(Callback):
    """监控梯度范数，记录到日志。

    Parameters
    ----------
    log_every_n_steps : int
        每隔多少步记录一次。
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self._log_every = log_every_n_steps

    def on_after_backward(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.global_step % self._log_every != 0:
            return

        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5

        pl_module.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)


class MemoryMonitor(Callback):
    """监控 GPU / CPU 内存使用。

    Parameters
    ----------
    log_every_n_epochs : int
        每隔多少 epoch 记录一次。
    """

    def __init__(self, log_every_n_epochs: int = 1) -> None:
        super().__init__()
        self._log_every = log_every_n_epochs

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.current_epoch % self._log_every != 0:
            return

        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                pl_module.log("memory/gpu_allocated_gb", allocated, on_step=False, on_epoch=True)
                pl_module.log("memory/gpu_reserved_gb", reserved, on_step=False, on_epoch=True)
        except Exception:
            pass


class RunSummaryCallback(Callback):
    """训练结束时输出模型摘要信息。

    只记录可观测指标与元信息，不写入 status.json 或其他主 artifacts。
    """

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        logger.info("=" * 60)
        logger.info("训练结束摘要")
        logger.info("=" * 60)
        logger.info(f"  总 epoch 数: {trainer.current_epoch}")
        logger.info(f"  总 step 数: {trainer.global_step}")

        if hasattr(pl_module, "count_parameters"):
            try:
                n_params = pl_module.count_parameters()
                logger.info(f"  模型参数数: {n_params:,}")
            except Exception:
                pass
        elif hasattr(pl_module, "model") and hasattr(pl_module.model, "count_parameters"):
            try:
                n_params = pl_module.model.count_parameters()
                logger.info(f"  模型参数数: {n_params:,}")
            except Exception:
                pass

        # 记录最佳 checkpoint 路径
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                logger.info(f"  最佳模型: {cb.best_model_path}")
                break

        # 记录最终验证指标（若有）
        if trainer.callback_metrics:
            for k, v in sorted(trainer.callback_metrics.items()):
                if isinstance(v, (int, float)):
                    logger.info(f"  {k}: {v:.6f}")

        logger.info("=" * 60)


# ============================================================================
# Callback 组装入口
# ============================================================================

def build_callbacks(
    config: Dict[str, Any],
    run_dir: Path,
    monitor_metric: str = "val_loss",
    track_with: Optional[str] = None,
) -> List[Callback]:
    """根据配置返回 callback 组合。

    Parameters
    ----------
    config : Dict[str, Any]
        训练配置字典，可包含：
        - early_stopping_patience : int
        - save_top_k : int
        - save_last : bool
        - enable_progress_bar : bool
        - gradient_monitor_every : int
        - memory_monitor_every : int
    run_dir : Path
        实验 run 目录，checkpoint 等将写在此下。
    monitor_metric : str
        监控指标名，如 "val_loss"。
    track_with : str, optional
        追踪后端："tensorboard" 或 "wandb"。

    Returns
    -------
    List[Callback]
    """
    callbacks: List[Callback] = []

    # checkpoint 目录
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -- ModelCheckpoint --
    monitor = config.get("monitor_metric", monitor_metric)
    mode = config.get("monitor_mode", "min")
    save_top_k = config.get("save_top_k", 1)
    save_last = config.get("save_last", True)

    callbacks.append(
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}-{" + monitor + ":.4f}",
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            verbose=True,
        )
    )

    # -- EarlyStopping --
    patience = config.get("early_stopping_patience", 5)
    if patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                mode=mode,
                verbose=True,
            )
        )

    # -- LearningRateMonitor --
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # -- 进度条 --
    enable_progress_bar = config.get("enable_progress_bar", True)
    if enable_progress_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=1, leave=False))

    # -- 自定义 Callback --
    grad_every = config.get("gradient_monitor_every", 0)
    if grad_every > 0:
        callbacks.append(GradientNormMonitor(log_every_n_steps=grad_every))

    mem_every = config.get("memory_monitor_every", 0)
    if mem_every > 0:
        callbacks.append(MemoryMonitor(log_every_n_epochs=mem_every))

    callbacks.append(RunSummaryCallback())

    return callbacks
