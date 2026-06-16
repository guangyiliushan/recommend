"""Configuration management — Hydra + YAML + dataclass hybrid.

RecBenchConfig dataclass hierarchy:
    RecBenchConfig
    ├── ExperimentConfig (name, tags, notes, ...)
    ├── DataConfig (name, data_dir, batch_size, split_ratios, ...)
    ├── ModelConfig (name, family, task_type, params, ...)
    ├── TrainingConfig (epochs, lr, optimizer, scheduler, ...)
    ├── EvaluationConfig (metrics, ranking_k, threshold, ...)
    └── RuntimeConfig (device, seed, deterministic, output_root, log_level, ...)

Support:
    - YAML file loading via OmegaConf
    - Hydra CLI override with ConfigStore
    - Structured config validation
    - Fully resolved config snapshots for reproducibility
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# 错误定义
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """配置错误，携带统一错误码、阶段与提示。"""

    def __init__(
        self,
        message: str,
        code: str = "CONFIG_VALIDATION_ERROR",
        phase: str = "config",
        hint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.phase = phase
        self.hint = hint
        self.details = details or {}


# ---------------------------------------------------------------------------
# 配置 dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """单次实验元信息。"""

    name: str = "default_experiment"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    track_with: Optional[str] = None  # "tensorboard" / "wandb" / None


@dataclass
class DataConfig:
    """数据集与 DataLoader 配置。"""

    name: str = "taac2026"
    data_dir: str = "./data"
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    batch_size: int = 256
    num_workers: int = 4
    max_seq_len: int = 50
    min_seq_len: int = 2
    neg_sample_count: int = 4


@dataclass
class ModelConfig:
    """模型选择与通用参数。"""

    name: str = "deepfm"
    family: str = "deep_ctr"
    task_type: str = "pointwise"
    problem_type: str = "binary"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """训练超参数。"""

    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    warmup_epochs: int = 0
    early_stopping_patience: int = 5
    gradient_clip_val: Optional[float] = None
    mixed_precision: Optional[str] = None  # "fp16" / "bf16" / None
    accumulate_grad_batches: int = 1


@dataclass
class EvaluationConfig:
    """评估配置。"""

    metrics: List[str] = field(default_factory=lambda: ["roc_auc", "log_loss"])
    ranking_k: Optional[List[int]] = None  # e.g. [5, 10, 20]
    threshold: float = 0.5
    generate_curves: bool = True
    statistical_test: Optional[str] = None


@dataclass
class RuntimeConfig:
    """运行时环境配置（设备、日志、可复现等跨组件设置）。"""

    device: str = "auto"  # "cpu" / "cuda" / "mps" / "auto"
    seed: int = 42
    deterministic: bool = False
    log_level: str = "INFO"
    output_root: str = "outputs"
    resume_from: Optional[str] = None
    fast_dev_run: bool = False
    num_devices: int = 1


@dataclass
class RecBenchConfig:
    """顶层实验配置，聚合所有子模块配置。"""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # 元信息（运行时填充）
    _raw_omegaconf: Optional[DictConfig] = field(default=None, repr=False, compare=False)
    _resolved_at: Optional[str] = field(default=None, repr=False, compare=False)
    _config_hash: Optional[str] = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Hydra ConfigStore 注册
# ---------------------------------------------------------------------------

def _register_config_store() -> None:
    """将 dataclass schema 注册为 Hydra Structured Config。

    仅在有 Hydra 环境时执行，否则为无操作。
    """
    try:
        from hydra.core.config_store import ConfigStore  # type: ignore[import-untyped]

        cs = ConfigStore.instance()
        cs.store(name="base_recbench", node=RecBenchConfig)
        cs.store(
            name="base_experiment",
            group="experiment",
            node=ExperimentConfig,
        )
        cs.store(name="base_data", group="dataset", node=DataConfig)
        cs.store(name="base_model", group="model", node=ModelConfig)
        cs.store(name="base_training", group="training", node=TrainingConfig)
        cs.store(name="base_evaluation", group="evaluation", node=EvaluationConfig)
        cs.store(name="base_runtime", group="runtime", node=RuntimeConfig)
    except ImportError:
        pass  # Hydra 未安装时静默跳过
    except Exception:
        pass  # 重复注册等异常静默跳过


_register_config_store()


# ---------------------------------------------------------------------------
# YAML 加载与合并
# ---------------------------------------------------------------------------

def _load_yaml(path: Union[str, Path]) -> DictConfig:
    """加载 YAML 文件为 OmegaConf DictConfig。

    Parameters
    ----------
    path : Union[str, Path]
        YAML 文件路径。

    Returns
    -------
    DictConfig

    Raises
    ------
    ConfigError
        文件不存在或解析失败。
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"配置文件不存在: {path}",
            code="CONFIG_FILE_NOT_FOUND",
            hint=f"检查 --config 路径: {path}",
        )
    try:
        return OmegaConf.load(path)
    except Exception as e:
        raise ConfigError(
            f"YAML 解析失败: {e}",
            code="CONFIG_PARSE_ERROR",
            hint="检查 YAML 语法是否正确",
        ) from e


def _omegaconf_to_dataclass(cfg: DictConfig) -> RecBenchConfig:
    """将 OmegaConf DictConfig 转换为 RecBenchConfig。

    Parameters
    ----------
    cfg : DictConfig
        OmegaConf 配置对象。

    Returns
    -------
    RecBenchConfig
    """
    schema = OmegaConf.structured(RecBenchConfig)
    merged = OmegaConf.merge(schema, cfg)
    resolved = OmegaConf.to_container(merged, resolve=True)
    assert isinstance(resolved, dict)

    return RecBenchConfig(
        experiment=ExperimentConfig(**resolved.get("experiment", {})),
        data=DataConfig(**resolved.get("data", {})),
        model=ModelConfig(**resolved.get("model", {})),
        training=TrainingConfig(**resolved.get("training", {})),
        evaluation=EvaluationConfig(**resolved.get("evaluation", {})),
        runtime=RuntimeConfig(**resolved.get("runtime", {})),
        _raw_omegaconf=cfg,
    )


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def load_config(
    config_path: Union[str, Path, None] = None,
    overrides: Optional[List[str]] = None,
) -> RecBenchConfig:
    """加载并合成完整配置。

    支持两种模式：
    1. Hydra 模式：当通过 Hydra 入口运行时，从 Hydra 上下文获取配置。
    2. 独立模式：直接加载 YAML 文件，适用于脚本和测试。

    Parameters
    ----------
    config_path : Union[str, Path, None]
        配置文件路径。为 None 时尝试从 Hydra 上下文获取。
    overrides : List[str], optional
        CLI 覆盖列表，如 ["model=deepfm", "training.learning_rate=3e-4"]。

    Returns
    -------
    RecBenchConfig
        fully resolved 配置对象。

    Raises
    ------
    ConfigError
        配置加载或校验失败。
    """
    overrides = overrides or []

    # 尝试 Hydra 模式
    try:
        from hydra.core.hydra_config import HydraConfig  # type: ignore[import-untyped]
        hydra_cfg = HydraConfig.get()
        if hydra_cfg is not None:
            cfg = hydra_cfg.job.config  # 现有 Hydra 合成配置
            config = _omegaconf_to_dataclass(cfg)
            config = resolve_paths(config, str(Path.cwd()))
            validate_config(config)
            return config
    except Exception:
        pass

    # 独立模式
    if config_path is None:
        config_path = Path("configs/config.yaml")

    cfg = _load_yaml(config_path)

    # 应用命令行覆盖
    for override in overrides:
        try:
            cfg = OmegaConf.update(cfg, override, merge=True)
        except Exception as e:
            raise ConfigError(
                f"命令行覆盖失败: {override}: {e}",
                code="CONFIG_VALIDATION_ERROR",
                hint=f"检查覆盖语法: {override}",
            ) from e

    config = _omegaconf_to_dataclass(cfg)
    config = resolve_paths(config, str(Path.cwd()))

    # 冻结配置 hash
    config._config_hash = _compute_config_hash(config)
    from datetime import datetime, timezone
    config._resolved_at = datetime.now(timezone.utc).isoformat()

    validate_config(config)
    return config


def validate_config(config: RecBenchConfig) -> None:
    """第二层语义校验：在 dataclass schema 之外执行逻辑校验。

    Parameters
    ----------
    config : RecBenchConfig

    Raises
    ------
    ConfigError
        语义校验失败。
    """
    # split_ratios 和必须为 1
    ratios = config.data.split_ratios
    if not (0.999 <= sum(ratios) <= 1.001):
        raise ConfigError(
            f"split_ratios 和须为 1，当前和为 {sum(ratios)}，"
            f"值为 {ratios}",
            code="CONFIG_VALIDATION_ERROR",
            hint="确保 split_ratios 为三个正数且和为 1，如 (0.8, 0.1, 0.1)",
        )

    # 所有 ratio 必须为正
    if any(r <= 0 for r in ratios):
        raise ConfigError(
            f"split_ratios 各项必须为正数，当前: {ratios}",
            code="CONFIG_VALIDATION_ERROR",
        )

    # metrics 不能为空
    if not config.evaluation.metrics:
        raise ConfigError(
            "evaluation.metrics 不能为空",
            code="CONFIG_VALIDATION_ERROR",
            hint="至少指定一个指标，如 ['roc_auc', 'log_loss']",
        )

    # ranking_k 合法性（若提供）
    if config.evaluation.ranking_k is not None:
        for k in config.evaluation.ranking_k:
            if not isinstance(k, int) or k <= 0:
                raise ConfigError(
                    f"ranking_k 必须为正整数，当前包含: {k}",
                    code="CONFIG_VALIDATION_ERROR",
                    hint="ranking_k 应为正整数或正整数列表，如 [5, 10, 20]",
                )

    # learning_rate 不能为负
    if config.training.learning_rate <= 0:
        raise ConfigError(
            f"learning_rate 必须为正数，当前: {config.training.learning_rate}",
            code="CONFIG_VALIDATION_ERROR",
        )

    # seed 合法性
    if config.runtime.seed < 0:
        raise ConfigError(
            f"seed 必须为非负整数，当前: {config.runtime.seed}",
            code="CONFIG_VALIDATION_ERROR",
        )


def resolve_paths(config: RecBenchConfig, cwd: str) -> RecBenchConfig:
    """路径归一化：输入路径基于 cwd，输出路径基于 output_root。

    Parameters
    ----------
    config : RecBenchConfig
    cwd : str
        当前工作目录（通常为项目根目录）。

    Returns
    -------
    RecBenchConfig
    """
    # 输入路径：基于 cwd
    data_dir = config.data.data_dir
    if not Path(data_dir).is_absolute():
        config.data.data_dir = str(Path(cwd) / data_dir)

    # 输出路径：基于 output_root
    output_root = config.runtime.output_root
    if not Path(output_root).is_absolute():
        config.runtime.output_root = str(Path(cwd) / output_root)

    return config


def get_config_snapshot(config: RecBenchConfig) -> Dict[str, Any]:
    """返回可序列化的配置快照。

    用于写入 outputs/experiments/{run_id}/config.yaml。

    Parameters
    ----------
    config : RecBenchConfig

    Returns
    -------
    Dict[str, Any]
    """
    snapshot = asdict(config)
    # 移除内部字段
    snapshot.pop("_raw_omegaconf", None)
    snapshot.pop("_resolved_at", None)
    snapshot.pop("_config_hash", None)

    # 注入元信息
    snapshot["_meta"] = {
        "schema_version": "1.0.0",
        "resolved_at": config._resolved_at,
        "config_hash": config._config_hash,
    }

    return snapshot


def _compute_config_hash(config: RecBenchConfig) -> str:
    """计算配置内容的稳定短哈希。

    Parameters
    ----------
    config : RecBenchConfig

    Returns
    -------
    str
        8 位十六进制哈希。
    """
    stable = {
        "experiment": config.experiment.name,
        "data": config.data.name,
        "model": config.model.name,
        "seed": config.runtime.seed,
    }
    raw = json.dumps(stable, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:8]
