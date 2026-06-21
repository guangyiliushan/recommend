# RecBench

RecBench 是一个推荐系统 Benchmark 与工程框架项目，目标是在统一配置、数据适配、模型契约、训练基础设施、评估协议和 artifact 规范下，逐步构建可复现、可比较、可维护的推荐实验平台。

## 当前状态

仓库当前已经完成的核心能力包括：

- `src/recsys/core`：注册表、基础模型契约、`PredictionBundle`
- `src/recsys/utils`：结构化配置、设备探测、日志、可复现性、profiling
- `src/recsys/training`：Lightning 训练封装、callbacks、loss、optimizer、scheduler、分布式策略解析
- `src/recsys/evaluation`：pointwise、ranking、multitask 评估与曲线导出
- `src/recsys/pipeline`：单实验主干（含训练型路径）、批量 Benchmark 调度、聚合报告
- `src/recsys/models/classical/`：`itemcf`（非训练排序基线）+ `dssm`（双塔神经网络，首个可训练模型）
- `docs/`：与当前仓库实现对齐的技术文档
- `scripts/run_single.py`、`scripts/run_benchmark.py`：可用的 CLI 入口

当前仍需明确的限制包括：

- 除 `itemcf` 和 `dssm` 外，多数模型家族文件仍是占位实现
- `scripts/run_ablation.py`、`scripts/download_data.py` 等辅助脚本仍待完善

换句话说，RecBench 目前已经具备“共享运行时主干 + 最小可运行基线”的基础，但还不是“全模型全部完工”的成品 Benchmark 套件。

## 为什么做这个项目

- 用统一契约承接经典协同过滤、深度 CTR/CVR、序列推荐、多任务与后续扩展方向
- 用一致的配置、评估和 artifact 协议降低实验漂移
- 用工程化主干替代一组彼此割裂的脚本
- 让本地开发与 CI 中的实验流程更可复现

## 当前最小闭环

当前代码仓库中，最清晰的可运行路径是：

1. 使用已注册的数据集适配器加载数据
2. 通过模型注册表实例化 `itemcf`（非训练）或 `dssm`（训练）
3. 走 `run_experiment()` 的非训练式或训练式路径
4. 通过 `evaluate()` 生成指标
5. 落盘 `config.yaml`、`status.json`、`metrics.json`、`predictions.parquet` 与 `curves/`
6. 通过 `run_benchmark()` 与 `Reporter` 聚合多次实验结果

## 项目结构

```text
.
|-- configs/                  # 配置与实验矩阵
|-- docs/                     # 文档站源码
|-- scripts/                  # CLI 实验入口（run_single / run_benchmark）
|-- src/recsys/
|   |-- core/                 # 注册表、基础契约、PredictionBundle
|   |-- data/                 # Dataset adapter 与离线大数据预处理（chunked read、格式转换、特征工程、负采样、缓存/checkpoint、数据库导出）
|   |-- evaluation/           # 指标、evaluator、ranking、visualization
|   |-- models/               # 模型家族与注册入口
|   |-- pipeline/             # 单实验、Benchmark、Reporter
|   |-- training/             # Trainer、callbacks、loss、optimizer、scheduler
|   `-- utils/                # 配置、日志、设备、可复现、profiling
|-- tests/                    # 测试目录
`-- .github/workflows/        # CI、文档与发布相关工作流
```

## 快速开始

### 1. 同步环境

仓库统一使用 `uv` 管理 Python 环境与依赖：

```bash
uv sync --extra dev
```

### 2. 运行静态检查

```bash
uv run ruff check .
```

### 3. 运行测试

```bash
uv run pytest -v
```

说明：当前仓库的 `tests/` 可能仍为空或覆盖有限，因此测试结果更多反映“已写测试的通过情况”，并不等于全量功能验证。

### 4. 构建文档站

```bash
uv run zensical build --strict --clean
```

## Python API 示例

### 模型发现

```python
from recsys import auto_discover_models, get_model, list_models

auto_discover_models()
print(list_models())

# 非训练模型
ItemCF = get_model("itemcf")
model = ItemCF(similarity="cosine", top_k_neighbors=50, recommend_k=10)

# 训练模型（DSSM 双塔神经网络）
DSSM = get_model("dssm")
model = DSSM(
    config={"embed_dim": 64, "hidden_dims": [128, 64]},
    schema_metadata={"num_users": 10000, "num_items": 5000},
)
```

### 单实验运行

```python
from recsys.pipeline.experiment import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    experiment_name="demo_itemcf",
    dataset_name="taac2026_data_sample",
    model_name="itemcf",
    seed=42,
    output_dir="./outputs/experiments",
    data_config={"root_dir": "./data"},
    evaluation_config={
        "primary_metric": "ndcg@10",
        "ranking_k": [10],
        "generate_curves": False,
    },
)

result = run_experiment(cfg)
print(result.status)
print(result.summary_metrics)
print(result.artifact_paths)
```

### 批量 Benchmark

```python
from recsys.pipeline.benchmark import BenchmarkConfig, ResumeMode, run_benchmark

bench_cfg = BenchmarkConfig(
    benchmark_name="demo_benchmark",
    models=["itemcf"],
    datasets=["taac2026_data_sample"],
    seeds=[42, 43],
    resume_mode=ResumeMode.SUCCESSFUL_SKIP,
    max_concurrent_runs=1,
    output_root="./outputs",
    experiment_output_dir="./outputs/experiments",
)

bench_result = run_benchmark(bench_cfg)
print(bench_result.status)
print(bench_result.summary_path)
print(bench_result.report_path)
```

### 命令行入口

```bash
# 单实验（非训练模型）
uv run python scripts/run_single.py --model itemcf --dataset taac2026_data_sample --seed 42

# 单实验（训练模型 DSSM）
uv run python scripts/run_single.py --model dssm --dataset taac2026_data_sample --seed 42 --epochs 10 --lr 3e-4

# 批量 Benchmark
uv run python scripts/run_benchmark.py --config configs/experiment/benchmark_classical.yaml

# 数据流水线 Benchmark（格式/压缩/后端对比）
uv run python scripts/benchmark_data_pipeline.py --rows 1000000
```

## 文档入口

- [Documentation Home](docs/index.md)
- [Getting Started](docs/getting-started.md)
- [Architecture](docs/concepts/architecture.md)
- [Configuration Guide](docs/concepts/configuration.md)
- [Public API](docs/project/api-contracts.md)
- [Pipeline Guide](docs/project/pipeline.md)
- [Evaluation Guide](docs/project/evaluation.md)
- [Benchmarking Guide](docs/project/benchmarking.md)
- [Persistence Contracts](docs/project/artifacts.md)
- [Model Integration Guide](docs/project/models.md)
- [Development Guide](docs/project/development.md)

## 开发原则

- 优先稳定共享契约，再扩展模型数量
- 文档必须诚实反映仓库真实状态
- Python 依赖与命令统一使用 `uv`
- 新功能变更应同步更新文档与必要测试
- 在共享运行时尚未稳定前，避免大批量引入模型实现

## 贡献

在提交较大改动前，请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 和 `docs/` 中的相关页面。

## License

本项目使用 MIT License，详见 `pyproject.toml`。
