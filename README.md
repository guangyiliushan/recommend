# RecBench

RecBench 是一个推荐系统 Benchmark 与工程框架项目，目标是在统一配置、数据适配、模型契约、训练基础设施、评估协议和产物规范下，逐步构建可复现、可比较、可维护的推荐实验平台。

## 当前状态

仓库当前已经完成的核心能力包括：

- **核心层**（`src/recsys/core`）：注册表机制、基础模型契约（`BaseRecommender` / `NeuralRecommender`）、统一预测产物 `PredictionBundle`
- **工具层**（`src/recsys/utils`）：Hydra + YAML + dataclass 混合配置系统、设备自动探测、loguru 日志追踪、可复现性保障（种子设置 + 确定性模式）、性能画像（参数量/延迟/显存/FLOPs）
- **数据层**（`src/recsys/data`）：四个数据集适配器（TAAC 2025 / TAAC 2026 / MovieLens-1M / Synthetic）、惰性序列切分 `SequenceSplit`、Dense ID Remap、离线预处理（chunk-aware 读取 + 列式转换）、特征工程、多策略负采样、EDA 报告生成
- **训练层**（`src/recsys/training`）：PyTorch Lightning 训练封装（`LightningRecommender` + `TrainerFactory`）、四类优化器、七种学习率调度器、回调组装（含梯度监控和显存监控）、九种损失函数（含 BPR/InfoNCE/Focal/MultiTask）、分布式策略解析
- **评估层**（`src/recsys/evaluation`）：四层分离架构（点式分类指标 / 分组排序指标 / 编排路由 / 可视化导出），支持 ranking/pointwise/multitask 自动分流
- **管线层**（`src/recsys/pipeline`）：单实验八阶段执行主干（含训练型双路径）、批量 Benchmark 矩阵展开与四种恢复策略、Reporter 聚合报告（含趋势和稳定性分析）
- **模型层**：`itemcf`（经典协同过滤，非训练 ranking 基线）和 `hyformer`（双塔 pointwise，已通过训练型路径验证）
- **文档体系**：`docs/` 下与当前仓库实现对齐的概念文档、项目文档和实验目录

当前已注册的数据集：

| 数据集 | 注册键 | 规模 | 适用任务 |
|:---|:---|:---|:---|
| TAAC 2026 基础样本 | `taac2026_data_sample` | 1,000 行 × 120 列 | CTR/CVR 点式预估 |
| TAAC 2026 进阶样本 | `taac2026_second_round` | 1,000 行 × 142 列 | CTR/CVR + Domain 序列 |
| TAAC 2025 1M | `taac2025_1M` | ~100 万序列 | 生成式序列推荐 |
| TAAC 2025 10M | `taac2025_10M` | ~1,000 万序列 | 大规模生成式推荐 |
| MovieLens-1M | `movielens_1m` | 6,040 用户 × ~3,900 物品 | 经典序列推荐 |
| Synthetic | `synthetic` | 可配置 | 性能基准测试 |

当前仍需明确的限制：

- 除 `itemcf` 和 `hyformer` 外，约 50 个模型家族文件仍是占位实现，注册装饰器处于注释状态
- `configs/experiment/` 下四个 Benchmark 配置文件为预留模板（planned），内容已被注释
- `criteo_kaggle`、`taobao_behavior` 等公开基准数据集适配器尚未实现

## 为什么做这个项目

- 用统一契约承接经典协同过滤、深度 CTR/CVR、序列推荐、多任务与后续扩展方向
- 用一致的配置、评估和产物协议降低实验漂移
- 用工程化主干替代一组彼此割裂的脚本
- 让本地开发与 CI 中的实验流程更可复现

## 当前最小闭环

当前代码仓库中，最清晰的可运行路径是：

1. 使用已注册的数据集适配器加载数据（TAAC 2026 / MovieLens-1M / TAAC 2025 / 合成数据）
2. 通过模型注册表实例化 `itemcf`（非训练，ranking）或 `hyformer`（训练，pointwise）
3. 走 `run_experiment()` 的非训练式或训练式路径
4. 通过评估器自动生成指标
5. 落盘 `config.yaml`、`status.json`、`metrics.json`、`predictions.parquet`、`curves/` 与 `checkpoints/`
6. 通过 `run_benchmark()` 与 `Reporter` 聚合多次实验结果

## 项目结构

```
.
├── configs/                  # Hydra 配置树（主配置、数据集子树、实验矩阵、模型子树）
├── docs/                     # MkDocs 文档站源码
├── scripts/                  # CLI 实验入口（7 条命令）
├── src/recsys/
│   ├── core/                 # 注册表、基础契约、PredictionBundle
│   ├── data/                 # 数据集适配器、离线预处理、特征工程、负采样、EDA
│   ├── evaluation/           # 指标计算、评估编排、排序、可视化
│   ├── models/               # 模型家族与注册入口
│   ├── pipeline/             # 单实验、Benchmark、Reporter
│   ├── training/             # Trainer、callbacks、loss、optimizer、scheduler
│   └── utils/                # 配置、日志、设备、可复现、profiling
├── tests/                    # 测试目录
└── .github/workflows/        # CI、文档与发布相关工作流
```

## 快速开始

### 环境准备

仓库统一使用 uv 管理 Python 环境与依赖：

```
uv sync --extra dev
```

### 常用命令

```
uv run ruff check .          # 静态检查
uv run pytest -v              # 运行测试
uv run zensical build --strict --clean  # 构建文档站
```

当前测试覆盖仍在逐步建设中，测试通过不等于所有模块都已经完整落地。

## CLI 入口全景

RecBench 提供七条 CLI 入口，覆盖从数据准备到性能对比的完整场景：

| 脚本 | 用途 | 典型用法 |
|:---|:---|:---|
| `scripts/run.py` | Hydra 主入口，YAML 组合配置运行单次实验 | `uv run python scripts/run.py` |
| `scripts/run_single.py` | 命令行直参模式，支持 argparse 和 Hydra 双模式 | `uv run python scripts/run_single.py --model itemcf --dataset taac2026_data_sample` |
| `scripts/run_benchmark.py` | 批量 Benchmark，三种运行模式 + 恢复策略 + 并发控制 | `uv run python scripts/run_benchmark.py --config configs/experiment/benchmark_all.yaml` |
| `scripts/run_ablation.py` | 消融实验矩阵，预置 ItemCF 四组参数空间 | `uv run python scripts/run_ablation.py --model itemcf --dataset taac2026_data_sample` |
| `scripts/download_data.py` | HuggingFace 数据集下载 | `uv run python scripts/download_data.py --dataset taac2026_data_sample` |
| `scripts/benchmark_data_pipeline.py` | 数据管线性能基准（格式/压缩/后端对比） | `uv run python scripts/benchmark_data_pipeline.py --rows 1000000` |
| `scripts/generate_report.py` | 从 summary.csv 生成 Markdown 性能对比报告 | `uv run python scripts/generate_report.py --input outputs/benchmarks/benchmark_classical/` |

### 更多运行示例

**单实验（非训练模型 ItemCF + MovieLens）**：
```
uv run python scripts/run_single.py --model itemcf --dataset movielens_1m --split-mode temporal --seed 42
```

**单实验（训练模型 HyFormer + 自定义超参数）**：
```
uv run python scripts/run_single.py --model hyformer --dataset taac2026_second_round --split-mode temporal --epochs 10 --lr 3e-4 --metrics roc_auc log_loss accuracy
```

**批量 Benchmark（经典协同过滤基线）**：
```
uv run python scripts/run_benchmark.py --config configs/experiment/benchmark_classical.yaml
```

**全矩阵 Benchmark（itemcf + hyformer × 全部数据集）**：
```
uv run python scripts/run_benchmark.py --config configs/experiment/benchmark_all.yaml --max-concurrent 2
```

**消融分析（ItemCF 邻居数扫描）**：
```
uv run python scripts/run_ablation.py --model itemcf --dataset movielens_1m --vary top_k_neighbors 10 20 50 100
```

## Python API 快速入门

### 模型发现与实例化

通过 `auto_discover_models()` 触发模型自动发现导入，通过 `list_models()` 列出所有已注册模型名称。非训练模型（如 ItemCF）通过 `get_model("itemcf")` 获取类后直接指定参数（如 `similarity="cosine"`、`top_k_neighbors=50`、`recommend_k=10`）实例化。训练型模型（如 HyFormer）需额外传入结构参数字典（`d_model`、`emb_dim`）和通过 Pipeline 自动采集的 `schema_metadata`（用户数、物品数等）。

### 单实验执行

通过构造 `ExperimentConfig` 对象指定实验名称、数据集注册键名、模型注册键名、随机种子和输出目录，传入 `run_experiment()` 获得结构化 `ExperimentResult`。数据配置中可指定 `root_dir`（数据缓存目录）和 `split_mode`（`"temporal"` 时序切分或 `"random"` 随机切分）。非训练模型无需训练配置，评估配置中指定 ranking 指标（如 `ndcg_at_k`、`recall_at_k`、`mrr`）和 Top-K 值。

训练型模型需额外提供训练配置（`epochs`、`batch_size`、`learning_rate`）和模型配置（通过 `params` 字典传入结构参数），评估配置中指定分类指标（如 `roc_auc`、`log_loss`）。

### 批量 Benchmark

通过构造 `BenchmarkConfig` 对象指定 Benchmark 名称、模型列表、数据集列表、种子列表、恢复模式和输出路径，传入 `run_benchmark()` 获得 `BenchmarkResult`。结果中包含 `summary.csv`、`leaderboard.csv`、`failures.csv` 路径和可通过 `succeeded_runs` / `failed_runs` 属性筛选的运行列表。

## 文档入口

- [文档首页](docs/index.md)
- [快速入门](docs/getting-started.md)
- [系统架构](docs/concepts/architecture.md)
- [配置系统](docs/concepts/configuration.md)
- [Pipeline 指南](docs/project/pipeline.md)
- [评估指南](docs/project/evaluation.md)
- [Benchmark 指南](docs/project/benchmarking.md)
- [数据集指南](docs/project/datasets.md)
- [模型集成指南](docs/project/models.md)
- [开发指南](docs/project/development.md)

## 核心设计要点

### 数据集与切分

TAAC 2026 数据集在加载时自动执行 Dense ID Remap，将原始稀疏用户/物品 ID 压缩为从 1 开始的连续整数，0 保留为填充槽位。支持通过 `split_mode` 选择按时序切分（`temporal`，用过去预测未来）或随机切分（`random`，适用于 CTR/CVR 点式评估）。TAAC 2025 支持通过 `min_action_type` 参数过滤曝光行为，仅保留点击交互。

### 模型能力路由

Pipeline 根据模型的 `Capability.TRAINABLE` 标记自动选择执行路径。非训练路径直接从数据集切分中提取用户-物品交互对（优先通过快速提取接口），调用 `fit()` + `predict()`。训练路径通过 DataLoader 构建、`LightningRecommender` 适配和 `pl.Trainer` 编排完成端到端训练与推理。

### 评估自动分流

评估器根据 `PredictionBundle` 的 `task_type` 字段自动分流：ranking 任务走逐组排序指标（NDCG@K、MRR、HitRate@K 等），pointwise 任务走分类指标（ROC-AUC、PR-AUC、LogLoss 等），multitask 任务逐任务头分别评估后汇总。

## 开发原则

- 优先稳定共享契约，再扩展模型数量
- 文档必须诚实反映仓库真实状态
- Python 依赖与命令统一使用 `uv`
- 新功能变更应同步更新文档与必要测试
- 在共享运行时尚未稳定前，避免大批量引入模型实现
- 切分模式通过 `data.split_mode` 统一控制，CLI 参数（`--split-mode`）负责透传
- 配置覆盖优先级为：CLI 参数 > YAML 子树配置 > dataclass 默认值

## 未来展望

- 剩余约 50 个模型（DeepFM、SASRec、DCN、ESMM、HSTU、DSTN 等）的注册激活与端到端验证
- `criteo_kaggle`、`taobao_behavior` 等公开基准数据集的适配器实现
- 四个 planned Benchmark 配置文件的模型就位后启用
- FSDP 和 DeepSpeed 分布式策略接入（`distributed.py` 已预留检查逻辑）
- 多模态嵌入作为训练输入的端到端 Pipeline 路径标准化
- Reporter v3 的 LaTeX 表格导出与统计显著性检验

## 贡献

在提交较大改动前，请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 和 `docs/` 中的相关页面。当前最有价值的贡献方向是增强共享契约、完善文档准确性和巩固最小可运行闭环。

## License

本项目使用 MIT License，详见 `pyproject.toml`。
