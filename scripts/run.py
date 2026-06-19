"""Hydra 主入口 — 通过 YAML 组合配置运行单次实验。

将 ``configs/`` 目录下的 Hydra 配置树加载、校验，并通过桥接函数
``recbench_to_experiment_config()`` 转换为 pipeline 层可消费的
``ExperimentConfig``，然后委托 ``run_experiment()`` 执行。

Usage::

    # 使用 config.yaml 默认配置运行
    uv run python scripts/run.py

    # 组合不同数据集和模型
    uv run python scripts/run.py dataset=taac2025 model=classical/itemcf

    # 命令行覆盖单个参数
    uv run python scripts/run.py model.params.similarity=iuf \\
        data.split_mode=random runtime.seed=43

    # 指定数据根目录
    uv run python scripts/run.py data.data_dir=./custom_data

Config schema
-------------
顶层 ``RecBenchConfig`` 由 ``src/recsys/utils/config.py`` 定义，包含六个子树：

- ``experiment`` — 实验元信息（name, tags, notes）
- ``data``      — 数据集与 DataLoader 配置（name, split_mode, batch_size ...）
- ``model``     — 模型选择与通用参数（name, family, task_type, params）
- ``training``  — 训练超参数（epochs, lr, optimizer ...）
- ``evaluation``— 评估配置（metrics, ranking_k, generate_curves ...）
- ``runtime``   — 运行时环境（device, seed, log_level, output_root）

组合机制
--------
Hydra 的 ``defaults: [_self_]`` 允许 CLI 按 group 名组合 YAML：

- ``dataset=taac2026``  加载 ``configs/dataset/taac2026.yaml`` 覆写 ``data`` 子树
- ``model=classical/itemcf`` 加载 ``configs/model/classical/itemcf.yaml`` 覆写 ``model`` 子树
- ``data.split_mode=random`` 点号路径覆写单个叶子字段

CLI override 优先级高于 YAML 配置，高于 dataclass 默认值。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from recsys import auto_discover_models

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Hydra 主入口函数。

    ``@hydra.main`` 装饰器自动完成以下步骤：
    1. 加载 ``configs/config.yaml``
    2. 按 defaults / CLI override 合并 dataset/model 等子树 YAML
    3. 注入 ``DictConfig`` 到 ``cfg`` 参数
    4. 设置 Hydra 工作目录

    Parameters
    ----------
    cfg : DictConfig
        Hydra 合成的结构化配置对象（OmegaConf DictConfig）。
    """
    # ---- 0. discover models ----
    print("Discovering models...")
    auto_discover_models()

    # ---- 1. DictConfig → RecBenchConfig ----
    from recsys.utils.config import (
        _omegaconf_to_dataclass,
        recbench_to_experiment_config,
        resolve_paths,
        validate_config,
    )
    config = _omegaconf_to_dataclass(cfg)
    config = resolve_paths(config, str(Path.cwd()))
    validate_config(config)

    print(f"\nStarting experiment: {config.experiment.name}")
    print(
        f"  Model: {config.model.name}, "
        f"Dataset: {config.data.name}, "
        f"Seed: {config.runtime.seed}"
    )

    # ---- 2. bridge → pipeline ExperimentConfig ----
    exp_config = recbench_to_experiment_config(config)

    # ---- 3. execute ----
    from recsys.pipeline.experiment import run_experiment
    result = run_experiment(exp_config)

    # ---- 4. report ----
    print(f"\nExperiment completed: {result.status.value}")
    if result.succeeded:
        print(f"  Summary metrics: {json.dumps(result.summary_metrics, indent=2)}")
        print("  Artifacts:")
        for key, path in result.artifact_paths.items():
            print(f"    {key}: {path}")
    else:
        print(f"  Error: {result.error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
