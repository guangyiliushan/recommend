"""Training framework — Lightning-based Trainer.

将 NeuralRecommender 适配为 LightningModule，并提供 TrainerFactory 编译 pl.Trainer。

核心对象：
    - LightningRecommender : 薄适配器，把 NeuralRecommender 包装为 LightningModule
    - TrainerFactory / create_trainer : 把 TrainingConfig + RuntimeConfig + run_dir 编译为 pl.Trainer

职责边界：
    - trainer 只负责训练生命周期：forward → loss → backward → optimizer/scheduler step
    - 不实现模型语义（forward/loss 仍在 NeuralRecommender 中）
    - 不做 evaluator 级指标计算（validation_step 只记录 val_loss 和简要摘要）
    - 不写 status.json / metrics.json / predictions.parquet
    - 不从 MODEL_REGISTRY / DATASET_REGISTRY 创建模型或数据集
    - 不初始化 TensorBoard/W&B（由调用方传入已准备好的 logger）
    - 不依赖 BaseRecommender.fit() 作为训练主路

数据流：
    batch dict → Batch 视图 → model.forward(batch) → ModelOutput
    → model.compute_loss(batch, output) → loss dict
    → optimizer/scheduler step

预测流（predict_step）：
    batch dict → Batch 视图 → model.forward(batch) → ModelOutput
    → 返回标准化中间预测块（供 pipeline 汇总为 PredictionBundle）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 使用 pytorch_lightning 与 pyproject.toml 依赖保持一致
import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from recsys.core.base_model import Batch, ModelOutput, NeuralRecommender
from recsys.training.callbacks import build_callbacks
from recsys.training.distributed import get_strategy_kwargs, resolve_strategy
from recsys.training.optimizers import OptimizerConfig, build_dual_optimizers, build_optimizer
from recsys.training.schedulers import (
    SchedulerConfig,
    build_scheduler,
)

# ============================================================================
# LightningRecommender — 薄适配器
# ============================================================================

class LightningRecommender(pl.LightningModule):
    """把 NeuralRecommender 适配为 LightningModule。

    真实业务仍在 NeuralRecommender.forward() 和 compute_loss()，
    Lightning 只负责生命周期钩子、日志转发和 optimizer/scheduler 接入。

    Parameters
    ----------
    model : NeuralRecommender
        已初始化的神经网络推荐模型。
    training_config : Dict[str, Any]
        训练配置字典，与 TrainingConfig dataclass 字段对齐。
    optimizer_config : OptimizerConfig
        optimizer 配置。
    scheduler_config : SchedulerConfig
        scheduler 配置。
    total_steps : int, optional
        总训练步数，用于 OneCycleLR 等需要 total_steps 的 scheduler。
    """

    def __init__(
        self,
        model: NeuralRecommender,
        training_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
        total_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self._training_config = training_config or {}
        self._optimizer_config = optimizer_config or OptimizerConfig()
        self._scheduler_config = scheduler_config or SchedulerConfig()
        self._total_steps = total_steps

        # 保存超参数（Lightning 惯例）
        self.save_hyperparameters(ignore=["model"])

    # ------------------------------------------------------------------
    # 核心训练步骤
    # ------------------------------------------------------------------

    def forward(self, batch: Union[Dict[str, Any], Batch]) -> ModelOutput:
        """前向传播。

        接收原始 batch dict 或 Batch 视图，转换为 Batch 后调用模型。
        """
        if not isinstance(batch, Batch):
            batch = self._dict_to_batch(batch)
        return self.model.forward(batch)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> STEP_OUTPUT:
        """训练 step。

        流程：dict → Batch → model.forward() → ModelOutput
              → model.compute_loss() → loss dict → 聚合 total_loss
        """
        b = self._dict_to_batch(batch)
        output = self.model.forward(b)
        losses = self.model.compute_loss(b, output)

        # 聚合损失
        if isinstance(losses, dict):
            total_loss = torch.tensor(0.0, device=self.device)
            for name, loss_val in losses.items():
                # NaN/Inf 检测 — 防止单步异常污染全体参数
                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    logger.warning(
                        "train_step=%d: loss '%s' is NaN/Inf (value=%s), 跳过本步",
                        batch_idx, name, float(loss_val.detach().cpu()),
                    )
                    return None  # Lightning 跳过本步梯度更新
                if name != "loss":
                    self.log(
                        f"train/{name}",
                        loss_val,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                    )
                total_loss = total_loss + loss_val
        else:
            total_loss = losses
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(
                    "train_step=%d: total_loss is NaN/Inf, 跳过本步", batch_idx,
                )
                return None

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_step", total_loss, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> STEP_OUTPUT:
        """验证 step — 含 NaN 守卫。

        只计算 val_loss，不做完整 evaluator 级指标计算。
        """
        b = self._dict_to_batch(batch)
        output = self.model.forward(b)
        losses = self.model.compute_loss(b, output)

        if isinstance(losses, dict):
            total_loss = torch.tensor(0.0, device=self.device)
            for name, loss_val in losses.items():
                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    logger.warning("val loss '%s' is NaN/Inf", name)
                    continue
                if name != "loss":
                    self.log(
                        f"val/{name}",
                        loss_val,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
                total_loss = total_loss + loss_val
        else:
            total_loss = losses

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=self.device)
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def test_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> STEP_OUTPUT:
        """测试 step。"""
        return self.validation_step(batch, batch_idx)

    def predict_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        """预测 step — 无梯度模式，节省显存。

        返回标准化中间预测块，供 pipeline 汇总为 PredictionBundle。
        """
        with torch.no_grad():
            b = self._dict_to_batch(batch)
            output = self.model.forward(b)

        # 提取预测分数
        scores: Optional[torch.Tensor] = None
        if output.probs is not None:
            scores = output.probs
        elif output.scores is not None:
            scores = output.scores

        # 提取标签
        labels: Optional[torch.Tensor] = None
        if b.has("label"):
            labels = b.label
        elif b.has("labels"):
            labels = b.labels

        # 提取 group_ids：优先显式 group_id，其次 user_id（推荐排序中 user 即 group）
        group_ids = None
        if b.has("group_id") and b.group_id is not None:
            group_ids = b.group_id.cpu()
        elif b.has("user_id") and b.user_id is not None:
            group_ids = b.user_id.cpu()

        return {
            "scores": scores.cpu() if scores is not None else None,
            "labels": labels.cpu() if labels is not None else None,
            "task_outputs": {
                k: v.cpu() for k, v in output.task_outputs.items()
            } if output.task_outputs else None,
            "group_ids": group_ids,
            "candidate_ids": (
                b.candidate_item_ids.cpu()
                if b.has("candidate_item_ids") and b.candidate_item_ids is not None
                else None
            ),
            "batch_idx": batch_idx,
        }

    # ------------------------------------------------------------------
    # optimizer / scheduler 配置
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """配置 optimizer 和 scheduler。

        Lightning 标准入口，调用 optimizers.py / schedulers.py 工厂。
        当 config.name == "dual_adagrad_adamw" 时返回双优化器。
        """
        # 双优化器分支：Adagrad(sparse) + AdamW(dense)
        if (
            self._optimizer_config.name == "dual_adagrad_adamw"
            and hasattr(self.model, "get_sparse_params")
            and len(self.model.get_sparse_params()) > 0
        ):
            sparse_opt, dense_opt = build_dual_optimizers(
                self.model,
                self._optimizer_config,
                sparse_lr=self._optimizer_config.sparse_lr,
            )
            scheduler_output = build_scheduler(
                dense_opt,
                self._scheduler_config,
                total_steps=self._total_steps,
            )
            dense_spec: Dict[str, Any] = {
                "optimizer": dense_opt,
            }
            if scheduler_output.scheduler_type == "plateau":
                dense_spec["lr_scheduler"] = {
                    "scheduler": scheduler_output.scheduler,
                    "monitor": scheduler_output.monitor or "val_loss",
                    "interval": scheduler_output.interval,
                    "frequency": scheduler_output.frequency,
                }
            else:
                dense_spec["lr_scheduler"] = {
                    "scheduler": scheduler_output.scheduler,
                    "interval": scheduler_output.interval,
                    "frequency": scheduler_output.frequency,
                }
            return [
                {"optimizer": sparse_opt, "frequency": 1},
                dense_spec,
            ]

        # 原有单优化器路径（向后兼容）
        optimizer = build_optimizer(self.model, self._optimizer_config)

        scheduler_output = build_scheduler(
            optimizer,
            self._scheduler_config,
            total_steps=self._total_steps,
        )

        result: Dict[str, Any] = {
            "optimizer": optimizer,
        }

        if scheduler_output.scheduler_type == "plateau":
            result["lr_scheduler"] = {
                "scheduler": scheduler_output.scheduler,
                "monitor": scheduler_output.monitor or "val_loss",
                "interval": scheduler_output.interval,
                "frequency": scheduler_output.frequency,
            }
        else:
            result["lr_scheduler"] = {
                "scheduler": scheduler_output.scheduler,
                "interval": scheduler_output.interval,
                "frequency": scheduler_output.frequency,
            }

        return result

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _dict_to_batch(self, batch: Dict[str, Any]) -> Batch:
        """把原始 batch dict 转换为 Batch 视图并移到正确设备。

        Parameters
        ----------
        batch : Dict[str, Any]
            原始 batch 字典。

        Returns
        -------
        Batch
            标准 Batch 视图。
        """
        device_batch: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict) and all(
                isinstance(v, torch.Tensor) for v in value.values()
            ):
                device_batch[key] = {
                    k: v.to(self.device) for k, v in value.items()
                }
            else:
                device_batch[key] = value
        return Batch(data=device_batch)

    @property
    def num_training_steps(self) -> Optional[int]:
        """估计总训练步数。"""
        return self._total_steps

    def count_parameters(self) -> int:
        """统计模型参数数量（委托给底层 NeuralRecommender）。"""
        return self.model.count_parameters()


# ============================================================================
# TrainerFactory — 编译 pl.Trainer
# ============================================================================

class TrainerFactory:
    """把 TrainingConfig + RuntimeConfig + run_dir 编译为 pl.Trainer。

    职责：
    - 策略解析（distributed.py）
    - callback 组装（callbacks.py）
    - logger 配置
    - pl.Trainer 构造

    不负责：
    - 读取全局状态
    - 决定 experiment 输出目录结构
    - 初始化 TensorBoard/W&B（只配置 pl.Trainer 的 logger 参数）

    Parameters
    ----------
    training_config : Dict[str, Any]
        训练配置，字段与 TrainingConfig dataclass 对齐。
    runtime_config : Dict[str, Any]
        运行时配置，字段与 RuntimeConfig dataclass 对齐。
    run_dir : Path
        experiment run 目录。
    monitor_metric : str
        监控指标名。
    track_with : str, optional
        "tensorboard" / "wandb" / None。
    """

    def __init__(
        self,
        training_config: Optional[Dict[str, Any]] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
        run_dir: Optional[Path] = None,
        monitor_metric: str = "val_loss",
        track_with: Optional[str] = None,
    ) -> None:
        self._training_config = training_config or {}
        self._runtime_config = runtime_config or {}
        self._run_dir = run_dir or Path("outputs/runs/_fallback")
        self._monitor_metric = monitor_metric
        self._track_with = track_with

    def build(
        self,
        wrap_model: Optional[NeuralRecommender] = None,
    ) -> Tuple[pl.Trainer, Optional[LightningRecommender]]:
        """编译 pl.Trainer。

        若传入 wrap_model，同时返回包装后的 LightningRecommender。

        Parameters
        ----------
        wrap_model : NeuralRecommender, optional
            需要包装的模型。

        Returns
        -------
        Tuple[pl.Trainer, Optional[LightningRecommender]]
        """
        # 1. 策略解析
        strategy_config = resolve_strategy(
            self._runtime_config, self._training_config
        )
        strategy_kwargs = get_strategy_kwargs(strategy_config)

        # 2. logger
        loggers = self._build_loggers()

        # 3. callbacks
        callbacks = build_callbacks(
            self._training_config,
            self._run_dir,
            monitor_metric=self._monitor_metric,
            track_with=self._track_with,
        )

        # 4. trainer kwargs
        trainer_kwargs: Dict[str, Any] = {
            **strategy_kwargs,
            "max_epochs": self._training_config.get("epochs", 10),
            "gradient_clip_val": self._training_config.get("gradient_clip_val", 1.0),
            "gradient_clip_algorithm": self._training_config.get(
                "gradient_clip_algorithm", "norm"
            ),
            "accumulate_grad_batches": self._training_config.get(
                "accumulate_grad_batches", 1
            ),
            "callbacks": callbacks,
            "logger": loggers if loggers else True,
            "log_every_n_steps": self._training_config.get("log_every_n_steps", 50),
            "enable_checkpointing": True,
            "deterministic": self._runtime_config.get("deterministic", False),
            "fast_dev_run": self._runtime_config.get("fast_dev_run", False),
        }

        # 移除 None 项
        trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

        trainer = pl.Trainer(**trainer_kwargs)

        # 5. 包装模型
        lit_model: Optional[LightningRecommender] = None
        if wrap_model is not None:
            total_steps = self._estimate_total_steps(wrap_model)
            lit_model = LightningRecommender(
                model=wrap_model,
                training_config=self._training_config,
                optimizer_config=self._build_optimizer_config(),
                scheduler_config=self._build_scheduler_config(),
                total_steps=total_steps,
            )

        return trainer, lit_model

    def _build_loggers(self) -> List[Any]:
        """构建 Lightning loggers。"""
        loggers: List[Any] = []
        track_with = self._track_with or self._runtime_config.get("track_with")

        if track_with == "tensorboard":
            tb_dir = self._run_dir / "logs" / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            loggers.append(
                TensorBoardLogger(
                    save_dir=str(tb_dir.parent),
                    name=tb_dir.name,
                )
            )
        elif track_with == "wandb":
            try:
                from pytorch_lightning.loggers import WandbLogger
                wb_dir = self._run_dir / "logs" / "wandb"
                wb_dir.mkdir(parents=True, exist_ok=True)
                loggers.append(
                    WandbLogger(
                        save_dir=str(wb_dir),
                        project=self._runtime_config.get("experiment_name", "recbench"),
                    )
                )
            except ImportError:
                logger.warning("wandb 未安装，跳过 WandbLogger。")

        return loggers

    def _build_optimizer_config(self) -> OptimizerConfig:
        """从 training_config 构建 OptimizerConfig。"""
        tc = self._training_config
        return OptimizerConfig(
            name=tc.get("optimizer", "adam"),
            lr=tc.get("learning_rate", 1e-3),
            weight_decay=tc.get("weight_decay", 1e-5),
            param_group_config=tc.get("param_group_config"),
            sparse_lr=tc.get("sparse_lr", 0.05),
        )

    def _build_scheduler_config(self) -> SchedulerConfig:
        """从 training_config 构建 SchedulerConfig。"""
        tc = self._training_config
        return SchedulerConfig(
            name=tc.get("scheduler") or "cosine",
            warmup_epochs=tc.get("warmup_epochs", 0),
            patience=tc.get("early_stopping_patience", 5),
            monitor=self._monitor_metric,
        )

    def _estimate_total_steps(self, model: NeuralRecommender) -> Optional[int]:
        """估算总训练步数。"""
        # 默认无法精确估算，由 pipeline 显式传入
        return None


# ============================================================================
# 便捷函数
# ============================================================================

def create_trainer(
    model: NeuralRecommender,
    training_config: Optional[Dict[str, Any]] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    run_dir: Optional[Path] = None,
    monitor_metric: str = "val_loss",
    track_with: Optional[str] = None,
) -> Tuple[pl.Trainer, LightningRecommender]:
    """一次性创建 trainer 和 wrapped model。

    供 pipeline 直接调用，减少装配复杂度。

    Parameters
    ----------
    model : NeuralRecommender
        已初始化的神经网络推荐模型。
    training_config : Dict[str, Any], optional
        训练配置。
    runtime_config : Dict[str, Any], optional
        运行时配置。
    run_dir : Path, optional
        experiment run 目录。
    monitor_metric : str
        监控指标名。
    track_with : str, optional
        追踪后端。

    Returns
    -------
    Tuple[pl.Trainer, LightningRecommender]
        (配置好的 Trainer, 包装后的 LightningRecommender)
    """
    factory = TrainerFactory(
        training_config=training_config,
        runtime_config=runtime_config,
        run_dir=run_dir,
        monitor_metric=monitor_metric,
        track_with=track_with,
    )
    trainer, lit_model = factory.build(wrap_model=model)
    assert lit_model is not None, "create_trainer 必须传入 model"
    return trainer, lit_model
