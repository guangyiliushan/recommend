"""全链路导入测试：确保所有公共模块可正常导入。"""


def test_top_level_import():
    """顶层包导出通畅。"""
    from recsys import (  # noqa: F401
        MODEL_REGISTRY,
        PredictionBundle,
        auto_discover_models,
        get_model,
        list_models,
    )


def test_evaluation_import():
    """评估层导入通畅。"""
    from recsys.evaluation import EvaluationConfig, EvaluationResult, evaluate  # noqa: F401


def test_pipeline_import():
    """pipeline 层导入通畅。"""
    from recsys.pipeline.experiment import ExperimentConfig, run_experiment  # noqa: F401


def test_benchmark_import():
    """benchmark 层导入通畅。"""
    from recsys.pipeline.benchmark import BenchmarkConfig, ResumeMode, run_benchmark  # noqa: F401


def test_training_import():
    """训练层导入通畅。"""
    from recsys.training import (  # noqa: F401
        LightningRecommender,
        TrainerFactory,
        build_callbacks,
        build_optimizer,
        build_scheduler,
        create_trainer,
        get_loss,
        list_losses,
        resolve_strategy,
    )


def test_utils_import():
    """工具层导入通畅。"""
    from recsys.utils import (  # noqa: F401
        ConfigError,
        DeviceError,
        LoggingError,
        ProfilingError,
        ReproducibilityError,
    )
