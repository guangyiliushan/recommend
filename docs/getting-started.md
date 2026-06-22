---
title: Getting Started
description: 环境配置、CLI 入口全景、常用命令与当前推荐入门路径
---

# Getting Started

## 环境约定

本仓库统一使用 uv 管理 Python 环境与依赖，所有命令以 `uv run` 为前缀，与 CI 和文档示例保持一致。

## 安装依赖

执行以下命令安装开发、测试与文档构建所需依赖：

```
uv sync --extra dev
```

## CLI 入口全景

RecBench 提供七条 CLI 入口，覆盖从数据准备到性能对比的完整场景：

| 脚本 | 用途 | 典型场景 |
|:---|:---|:---|
| `scripts/run.py` | Hydra 主入口，通过 YAML 组合配置运行单次实验 | 日常开发、参数组合探索 |
| `scripts/run_single.py` | 命令行直参模式运行单次实验 | 快速验证、无 Hydra 依赖的 CI 环境 |
| `scripts/run_benchmark.py` | 批量 Benchmark，支持矩阵展开与配置文件 | 多模型×多数据集×多 seed 全面对比 |
| `scripts/run_ablation.py` | 消融实验矩阵执行器 | 单超参数变体对比分析 |
| `scripts/download_data.py` | 通过 HuggingFace 下载 TAAC 数据集 | 首次使用前的数据准备 |
| `scripts/benchmark_data_pipeline.py` | 数据管线性能基准测试 | 对比存储格式、压缩算法与计算后端 |
| `scripts/generate_report.py` | 从 Benchmark 的 summary.csv 生成 Markdown 对比报告 | 实验完成后生成可读报告 |

### run.py — Hydra 主入口

基于 Hydra 配置框架的单次实验入口。从 `configs/config.yaml` 加载默认配置，支持通过 CLI 覆盖任意配置字段，以及通过 `dataset=` 和 `model=` 前缀组合不同数据集的 YAML 子树和模型配置。例如：

```
uv run python scripts/run.py
uv run python scripts/run.py dataset=taac2025 model=classical/itemcf
uv run python scripts/run.py model.params.similarity=iuf data.split_mode=random runtime.seed=43
```

### run_single.py — 命令行直参模式

提供完整的命令行参数覆盖，支持两种运行模式：

- **argparse 模式**（默认）：通过 `--model`、`--dataset`、`--seed` 等参数直接指定实验配置，额外提供 `--min-action-type`（TAAC 2025 行为过滤）和 `--split-mode`（时序/随机切分）参数
- **Hydra 模式**：通过 `--hydra` 标记启用，从 YAML 加载基准配置并支持 `--hydra-overrides` 覆盖

典型用法：

```
uv run python scripts/run_single.py --model itemcf --dataset taac2026_data_sample --split-mode temporal --seed 42
uv run python scripts/run_single.py --model hyformer --dataset taac2026_second_round --split-mode temporal --epochs 10 --lr 3e-4 --metrics roc_auc log_loss accuracy
```

### run_benchmark.py — 批量 Benchmark

支持三种运行模式：

- **配置文件模式**：通过 `--config` 指定 YAML 矩阵配置文件（如 `configs/experiment/benchmark_classical.yaml`）
- **命令行模式**：通过 `--models` 和 `--datasets` 直接指定参与矩阵的组件
- **Hydra 模式**：通过 `--hydra` 标记从 `configs/config.yaml` 注入默认参数

关键参数包括 `--resume-mode`（恢复策略）和 `--max-concurrent`（并发控制）。典型用法：

```
uv run python scripts/run_benchmark.py --config configs/experiment/benchmark_classical.yaml
uv run python scripts/run_benchmark.py --models itemcf hyformer --datasets taac2026_data_sample
```

### run_ablation.py — 消融实验

对单一超参数构建多组变体配置，逐组执行实验并输出 CSV 对比表。ItemCF 预置了四组消融参数空间（`similarity`、`top_k_neighbors`、`recommend_k`、`normalize`），也可通过 `--vary` 自定义参数名和候选值列表。典型用法：

```
uv run python scripts/run_ablation.py --model itemcf --dataset taac2026_data_sample
uv run python scripts/run_ablation.py --model itemcf --vary top_k_neighbors 10 20 50 100
```

### download_data.py — 数据集下载

维护注册名到 HuggingFace 仓库 ID 的映射表，支持 `taac2025_1M`、`taac2025_10M`、`taac2026_data_sample` 和 `taac2026_second_round` 四个数据集的命令行下载。通过 `--cache-dir` 指定本地缓存目录。MovieLens-1M 在数据集适配器内部自行处理下载，不通过此脚本。

### benchmark_data_pipeline.py — 数据管线性能基准

生成推荐风格的合成数据集，对存储格式（CSV/Parquet/Feather/ORC）、压缩算法和计算后端的全组合矩阵进行读写性能测量，输出多种格式的聚合报告。

### generate_report.py — 报告生成

从 Benchmark 产出的 `summary.csv` 加载实验数据，生成包含速度对比（含加速比）、准确性对比和内存使用的 Markdown 性能对比报告。

## 常用命令

### 静态检查

```
uv run ruff check .
```

### 测试

```
uv run pytest -v
```

当前测试覆盖仍在逐步建设中，测试通过不等于所有模块都已经完整落地。

### 构建文档站

```
uv run zensical build --strict --clean
```

## 当前推荐阅读顺序

如果你想先理解项目，建议依次阅读：

1. `README.md` — 项目概览与快速定位
2. `docs/index.md` — 文档站索引
3. `docs/concepts/architecture.md` — 系统架构
4. `docs/concepts/configuration.md` — 配置系统
5. `docs/project/pipeline.md` — 单实验执行管线
6. `docs/project/benchmarking.md` — 批量 Benchmark 与聚合
7. `docs/project/evaluation.md` — 评估体系

## 当前推荐代码入口

如果你要从代码快速建立整体认知，建议优先看以下文件：

- `src/recsys/core/registry.py` — 模型与数据集的注册机制
- `src/recsys/core/base_model.py` — 推荐模型的抽象基类与 Batch 视图
- `src/recsys/core/prediction_bundle.py` — 统一预测产物契约
- `src/recsys/pipeline/experiment.py` — 单实验执行编排器
- `src/recsys/pipeline/benchmark.py` — 批量 Benchmark 调度器
- `src/recsys/evaluation/evaluator.py` — 评估编排入口

## 当前最适合验证的路径

当前仓库中作为最小闭环理解入口的最佳路径为：

1. 通过模型自动发现获取已注册模型列表，理解 `itemcf`（非训练、ranking）和 `hyformer`（训练型、pointwise）两种核心模型
2. 理解 `run_experiment()` 的双路径执行机制：非训练路径（直接 fit+predict）和训练路径（DataLoader → LightningRecommender → Trainer）
3. 理解评估层如何根据 `PredictionBundle` 的 `task_type` 自动分流到分类指标或排序指标
4. 理解 `run_benchmark()` 如何展开模型×数据集×种子的三维矩阵，以及 Reporter 如何将多个实验结果聚合为排行榜和稳定性报告

## Python API 快速入门

### 模型发现

通过 `auto_discover_models()` 触发模型自动发现导入，通过 `list_models()` 列出所有已注册模型名称，通过 `get_model("itemcf")` 按注册键获取模型类并按需配置参数（如 `similarity="cosine"`）实例化。

### 单实验执行

通过构造 `ExperimentConfig` 对象指定实验名称、数据集注册键名、模型注册键名、随机种子和输出目录，传入 `run_experiment()` 获得结构化 `ExperimentResult`。数据配置中可指定 `root_dir`（数据缓存目录）和 `split_mode`（`"temporal"` 时序切分或 `"random"` 随机切分）。非训练模型（如 itemcf）无需训练配置，评估配置中可指定 ranking 指标列表和 Top-K 值。

### 训练型模型执行

训练型模型（如 hyformer）需额外提供训练配置（`epochs`、`batch_size`、`learning_rate` 等）和模型配置（通过 `params` 字典传入 `d_model`、`emb_dim` 等结构参数）。评估配置中指定分类指标（如 `roc_auc`、`log_loss`）。Pipeline 会自动从数据集实例采集用户数、物品数等元信息并传入模型构造函数。

## 仓库重要目录

| 目录 | 说明 |
|:---|:---|
| `src/recsys` | 核心源码（core/pipeline/evaluation/training/data/utils） |
| `configs` | Hydra 配置树（主配置、数据集子树、实验矩阵、模型子树） |
| `docs` | MkDocs 文档站源码（concepts/project/experiments） |
| `tests` | 测试用例 |
| `outputs` | 实验与 Benchmark 结果输出目录 |
| `.github/workflows` | CI/CD 自动化工作流 |

## 当前需要特别注意的现实边界

- 当前最清晰可运行的模型是 `itemcf`（非训练、ranking）和 `hyformer`（训练型、pointwise），其余约 50 个模型文件的注册装饰器仍处于注释状态
- 已注册可用的数据集包括：`movielens_1m`（经典序列推荐）、`taac2025_1M` 和 `taac2025_10M`（生成式推荐）、`taac2026_data_sample` 和 `taac2026_second_round`（CTR/CVR 预估）
- `itemcf` 对所有已注册数据集均兼容，`hyformer` 主要适配 TAAC 2026 的点式预估格式
- TAAC 2026 数据集在加载时自动执行 Dense ID Remap，将原始稀疏用户/物品 ID 压缩为从 1 开始的连续整数
- `--split-mode` 控制数据切分策略（`temporal` 按时序切分用于推荐系统标准场景，`random` 随机切分用于点式评估场景），通过命令行参数透传到数据集适配器
- `--min-action-type` 控制 TAAC 2025 的正交互过滤阈值（设为 1 时仅保留点击行为，适用于隐式反馈推荐）
- `train_hyformer.py` 是独立的模型结构调试脚本，使用随机内存数据直接训练，不走数据集适配器和 Pipeline
- `configs/experiment/` 下三个配置文件处于活跃状态（`benchmark_all.yaml`、`benchmark_classical.yaml`、`benchmark_deep_ctr.yaml`），四个为预留模板（内容已注释）
- Hydra 配置覆盖的优先级为：CLI 参数 > YAML 子树配置 > dataclass 默认值

## 未来展望

- 剩余约 50 个模型（DeepFM、SASRec、DCN、ESMM、HSTU 等）的注册激活与端到端验证
- `criteo_kaggle`、`taobao_behavior` 等公开基准数据集适配器的实现与注册
- Benchmark 预设配置的 Hydra 统一解析与 Reporter 聚合流程的完整融合
- 多模态嵌入作为训练输入的端到端 Pipeline 路径

## 参考

- [配置系统文档](concepts/configuration.md)
- [Pipeline 文档](project/pipeline.md)
- [Benchmark 文档](project/benchmarking.md)
- [数据集文档](project/datasets.md)
