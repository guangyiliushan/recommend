"""Utilities: config, logging, reproducibility, profiling, device management.

Public API surface — re-exports only stable, externally-consumed symbols.
Internal helpers and implementation details are NOT re-exported here.
"""

from recsys.utils.config import (
    ConfigError,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    RecBenchConfig,
    RuntimeConfig,
    TrainingConfig,
    get_config_snapshot,
    load_config,
    validate_config,
)
from recsys.utils.device import (
    DeviceError,
    DeviceInfo,
    get_device,
    get_device_info_summary,
)
from recsys.utils.logging import (
    LoggingContext,
    LoggingError,
    get_logger,
    log_experiment_summary,
    setup_logging,
)
from recsys.utils.profiling import (
    LatencyInfo,
    MemoryInfo,
    ParameterInfo,
    ProfilingConfig,
    ProfilingError,
    ProfilingResult,
    count_parameters,
    get_memory_usage,
    measure_inference_latency,
    profile_model,
)
from recsys.utils.reproducibility import (
    DeterministicInfo,
    ReproducibilityError,
    SeedInfo,
    deterministic_mode,
    get_reproducibility_summary,
    set_seed,
)

__all__ = [
    # config
    "RecBenchConfig",
    "ExperimentConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "RuntimeConfig",
    "ConfigError",
    "load_config",
    "validate_config",
    "get_config_snapshot",
    # device
    "DeviceInfo",
    "DeviceError",
    "get_device",
    "get_device_info_summary",
    # logging
    "LoggingContext",
    "LoggingError",
    "setup_logging",
    "get_logger",
    "log_experiment_summary",
    # reproducibility
    "SeedInfo",
    "DeterministicInfo",
    "ReproducibilityError",
    "set_seed",
    "deterministic_mode",
    "get_reproducibility_summary",
    # profiling
    "ProfilingConfig",
    "ProfilingResult",
    "ProfilingError",
    "ParameterInfo",
    "LatencyInfo",
    "MemoryInfo",
    "profile_model",
    "count_parameters",
    "measure_inference_latency",
    "get_memory_usage",
]
