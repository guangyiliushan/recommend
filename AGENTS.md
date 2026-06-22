# AGENTS.md — RecBench AI 协作开发规范

> 本文件是写给 AI 编码助手（Codex、Copilot、Cursor、Claude Code、TRAE 等）的项目说明书，定义技术栈、工程约定、工作流程与行为边界。
> 面向人类的项目介绍请阅读 [README.md](./README.md)。

---

## 1. 项目身份

- **项目名**：`recbench`（pip 包），源码命名空间 `recsys`
- **定位**：推荐系统基准框架 — 50+ 算法从 2001 到 2026，覆盖经典协同过滤到统一 Transformer 架构
- **语言**：Python 3.10+
- **包管理器**：`uv`（禁止使用 pip/poetry/pnpm 管理 Python 依赖）
- **构建后端**：setuptools（`pyproject.toml`）

### 技术栈速览

| 层 | 核心技术 |
|:---|:---|
| 深度学习 | PyTorch >= 2.1, PyTorch Lightning >= 2.1 |
| 配置管理 | Hydra + OmegaConf + YAML |
| 数据处理 | pandas, numpy, scipy, pyarrow, HuggingFace `datasets` |
| 评估指标 | scikit-learn, 自研 ranking/metrics 模块 |
| 实验追踪 | MLflow, TensorBoard |
| 可视化 | matplotlib, seaborn, plotly, ECharts (EDA 报告) |
| 日志 | loguru |
| 测试 | pytest >= 9.1, pytest-benchmark >= 4 |
| 代码质量 | ruff (line-length=100), black, isort |
| 文档站 | Zensical（`zensical.toml`） |

---

## 2. 快速命令

**AI 每次修改代码后，必须按顺序执行：**

```powershell
# 1. 静态检查（零报错才进入下一步）
uv run ruff check .

# 2. 自动修复可修问题
uv run ruff check --fix .

# 3. 全量测试
uv run pytest -v

# 4. 知识图谱同步（仅代码修改后）
graphify update .
```

**高频使用命令**：

```powershell
# 单一实验（argparse 模式，推荐快速验证）
uv run python scripts/run_single.py --model itemcf --dataset taac2026_data_sample --metrics ndcg_at_k recall_at_k

# 单一实验（Hydra 模式，配置驱动）
uv run python scripts/run.py dataset=taac2026 model=classical/itemcf

# EDA 数据集分析
uv run recsys-dataset-eda --dataset taac2026_data_sample

# 批量 Benchmark
uv run python scripts/run_benchmark.py --config configs/experiment/benchmark_classical.yaml

# 仅运行聚焦测试（避免全量耗时）
uv run pytest tests/test_models/test_itemcf.py -v
uv run pytest tests/test_data/test_eda_cli.py -v
```

---

## 3. 项目架构

### 3.1 分层架构（自上而下依赖）

```
pipeline/     → 实验编排、Benchmark 调度、报告聚合
models/       → 7 个模型家族（classical/deep_ctr/feature_cross/generative/pcvr/sequence/unified）
training/     → Lightning 训练器、损失函数、优化器、调度器、分布式策略
evaluation/   → 评估编排、分类指标、排序指标、可视化
data/         → 数据集适配器、EDA 分析、预处理、特征工程、负采样
  ├── datasets/   → TAAC2025/2026、MovieLens-1M、Synthetic
  └── eda/        → 10 个统计模块 + CLI + 渲染 + 报告
core/         → 抽象基类（BaseDataset, BaseRecommender, NeuralRecommender）、注册表、PredictionBundle
utils/        → 配置桥接、设备探测、日志、进度条、可复现性
```

### 3.2 关键目录速查

| 目录 | 职责 | AI 修改策略 |
|:---|:---|:---|
| `src/recsys/core/` | 抽象契约层 | ⚠️ 先询问，修改影响所有下游 |
| `src/recsys/models/` | 50+ 模型定义 | ✅ 可自由新增/修改模型文件 |
| `src/recsys/data/datasets/` | 数据集适配器 | ✅ 可新增，修改时需保持 `_load_raw()` 契约 |
| `src/recsys/data/eda/` | EDA 分析管线 | ✅ 可修改，遵循 stats/ 模块统一契约 |
| `src/recsys/pipeline/` | 实验执行主干 | ⚠️ 先询问，修改影响所有实验 |
| `src/recsys/training/` | 训练基础设施 | ⚠️ 先询问，修改影响所有训练模型 |
| `scripts/` | CLI 入口脚本 | ✅ 可自由修改 |
| `configs/` | YAML 配置文件 | ✅ 可自由修改 |
| `tests/` | 测试套件 | ✅ 必须同步更新 |
| `docs/` | 文档站源码 | ✅ 代码变更后同步更新 |

### 3.3 模型注册机制

**只有通过 `@MODEL_REGISTRY.register()` 装饰的模型才可被 CLI/实验使用**。目前仅两个模型已注册可用：

| 模型 | 注册名 | 家族 | 训练 | 任务 |
|:---|:---|:---|:---|:---|
| ItemBasedCF | `itemcf` | classical | 非训练（fit/predict） | ranking |
| HyFormerAdapter | `hyformer` | unified | 可训练（NeuralRecommender） | pointwise |

其余 ~50 个模型文件的 `@MODEL_REGISTRY.register()` 均被注释，为预留占位。激活新模型时必须：
1. 取消注释 `@MODEL_REGISTRY.register()`
2. 在 `src/recsys/models/__init__.py` 或对应家族 `__init__.py` 中导入
3. 运行 `uv run python -c "from recsys import auto_discover_models; print(list_models())"` 验证注册

**数据集同样是注册制**，`DATASET_REGISTRY.register()` → `data/dataset_registry.py` 的 import 触发。

---

## 4. 代码约定

### 4.1 导入规范

```python
# ✅ 正确：绝对导入
from recsys.core.registry import MODEL_REGISTRY
from recsys.pipeline.experiment import run_experiment

# ❌ 禁止：src 前缀
from src.recsys.core.registry import MODEL_REGISTRY

# ✅ 导入顺序：future → stdlib → 第三方 → 本地
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from recsys.core.base_model import NeuralRecommender
```

### 4.2 命名约定

| 类型 | 规则 | 示例 |
|:---|:---|:---|
| 模型注册名 | 小写 + 下划线 | `itemcf`, `hyformer` |
| 数据集注册名 | 小写 + 下划线 | `taac2026_data_sample`, `movielens_1m` |
| 类名 | PascalCase | `ItemBasedCF`, `SparsityResult` |
| 函数/变量 | snake_case | `analyze_sparsity`, `user_id` |
| 常量 | UPPER_SNAKE | `_PROFILE_STATS_MAP`, `MAX_ROWS` |
| 私有成员 | 单下划线前缀 | `_sim_matrix`, `_abs_sim_matrix` |

### 4.3 EDA Stats 模块契约

所有 `stats/*.py` 模块遵循统一契约，新增模块时必须遵守：

```python
@dataclass
class XxxResult:
    """分析结果。"""
    # ... 业务字段 ...
    skipped: bool = False
    skip_reason: Optional[str] = None

def analyze(df: pd.DataFrame, **kwargs) -> XxxResult:
    """接受 DataFrame，返回 Result。所需列缺失时设 skipped=True。"""
```

- **输入**：`pd.DataFrame` 是唯一数据契约
- **输出**：`@dataclass` Result，含 `skipped: bool` + `skip_reason: Optional[str]`
- **自检**：自动检测所需列，缺失时优雅跳过

### 4.4 配置约定

- 默认配置在 dataclass 字段默认值中定义
- YAML 文件用于 Hydra 组合覆盖：`configs/config.yaml` → `configs/dataset/*.yaml` + `configs/model/**/*.yaml`
- CLI 参数为最高优先级：`--model itemcf` 覆盖 YAML
- 实验输出统一到 `outputs/` 目录（已在 `.gitignore`）

### 4.5 Ruff 规则

```toml
line-length = 100
target-version = "py310"
select = ["E", "F", "W", "I", "N", "B", "SIM"]
ignore = ["E501"]  # 行长度由 black 处理
```

**提交前必须 `uv run ruff check .` 零报错。**

---

## 5. 开发工作流

### 5.1 标准修改流程

```
1. 理解现状 → 阅读相关源码 + 测试
2. 设计方案 → 最小必要改动，避免过度工程
3. 实现变更 → 编辑文件
4. 静态检查 → uv run ruff check .
5. 运行测试 → uv run pytest -v
6. 失败修复 → 仅修复与本次变更相关的失败
7. 图谱同步 → graphify update .
```

### 5.2 测试策略

| 层级 | 位置 | 运行频率 | 命令 |
|:---|:---|:---|:---|
| 聚焦测试 | 按文件名筛选 | 每次改动 | `uv run pytest tests/test_xxx.py -v` |
| 全量测试 | `tests/` | 提交前 | `uv run pytest -v` |
| 集成测试 | 标记 `@pytest.mark.integration` | 需要网络时 | `uv run pytest -v -m "integration"` |
| 基准测试 | 标记 `@pytest.mark.benchmark` | 性能回归时 | `uv run pytest -v --benchmark-only` |

**测试组织**：`tests/` 目录镜像 `src/recsys/` 结构 — `tests/test_data/` 对 `src/recsys/data/`，`tests/test_models/` 对 `src/recsys/models/`。

### 5.3 提交信息格式

```
<type>(<scope>): <subject>

- type: feat / fix / refactor / test / docs / chore
- scope: eda / itemcf / pipeline / data / training / config
- subject: 中文简要描述（50 字内）
```

示例：
```
feat(eda): 新增 sparsity 模块，支持 Gini 系数与洛伦兹曲线
fix(itemcf): predict 空结果不再触发 ModelContractError
refactor(data): _TabularSplit 预计算可变长特征最大维度
```

---

## 6. 行为边界

### ✅ 必须执行
- 修改代码后运行 `uv run ruff check .` 和 `uv run pytest -v`
- 代码变更后运行 `graphify update .` 同步知识图谱
- 新增模型/数据集时必须同步更新 `__init__.py` 的导入链
- 新增 `stats/` 模块时遵循 `XxxResult + analyze()` 契约

### ⚠️ 先询问用户
- 修改 `src/recsys/core/` 中的抽象基类（影响所有下游）
- 修改 `src/recsys/pipeline/experiment.py` 的 `run_experiment()` 签名
- 引入新依赖（先检查 `pyproject.toml` 是否已有同类库）
- 修改 `.github/workflows/` CI/CD 配置

### 🚫 绝对禁止
- 修改或删除 `.env`、密钥文件、凭证
- 手动修改 `uv.lock`（应通过 `uv lock` 或 `uv sync` 生成）
- 提交 `outputs/`、`__pycache__/`、`.npz` 缓存文件
- 绕过 ruff 检查强行提交代码
- 修改 `dist/` 目录下的官方竞赛代码
- 直接修改 `docs/assets/figures/eda/` 下的图表文件（应由 EDA 管线生成）

---

## 7. 常见问题排查

| 症状 | 排查路径 |
|:---|:---|
| `ruff check` 报错 | `uv run ruff check --fix .` 自动修复 → 手动处理剩余问题 |
| 测试失败 | 先运行聚焦测试定位：`uv run pytest tests/test_xxx.py -v -x` |
| 模型注册不生效 | `uv run python -c "from recsys import auto_discover_models; print(list_models())"` |
| Hydra 配置报错 | 确认 YAML 包含 `# @package _global_`，字段在正确子树下 |
| EDA 模块跳过 | 检查 stats 返回值 `skipped=True` 及 `skip_reason` |
| ItemCF 内存溢出 | 大数据集设置 `--min-action-type 1` 过滤，或使用 `--max-rows` |
| TAAC2025 加载 OOM | 使用 `analyze --subset seq --max-rows 500000` 而非整表加载 |

---

## 8. 知识图谱

本项目在 `graphify-out/` 维护了知识图谱（god nodes + community structure + cross-file relationships）。

- 当用户输入 `/graphify` 时，调用 `skill: "graphify"`
- 代码库问题优先使用 `graphify query "<问题>"`（需 `graphify-out/graph.json` 存在）
- 关系查询使用 `graphify path "<A>" "<B>"`
- 概念解释使用 `graphify explain "<概念>"`
- 代码修改后运行 `graphify update .` 保持图谱最新

---

## 9. 补充资源

本文档遵循渐进式披露原则，详细信息请阅读对应文档：

| 文档 | 内容 |
|:---|:---|
| [README.md](./README.md) | 项目介绍与快速上手 |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | 贡献指南（环境搭建、PR 粒度） |
| [docs/concepts/architecture.md](./docs/concepts/architecture.md) | 系统架构详解 |
| [docs/concepts/configuration.md](./docs/concepts/configuration.md) | 配置系统说明 |
| [docs/project/development.md](./docs/project/development.md) | 开发规范与优先级 |
| [docs/project/models.md](./docs/project/models.md) | 模型集成指南 |
| [docs/project/datasets.md](./docs/project/datasets.md) | 数据集适配说明 |
| [docs/project/benchmarking.md](./docs/project/benchmarking.md) | Benchmark 执行指南 |
| [docs/project/api-contracts.md](./docs/project/api-contracts.md) | 公共 API 契约 |
| [docs/project/pipeline.md](./docs/project/pipeline.md) | Pipeline 流程说明 |
| [.trae/documents/items-cumulative-summary.md](./.trae/documents/items-cumulative-summary.md) | 已完成计划全局总结 |
| [.trae/documents/items-cumulative-details.md](./.trae/documents/items-cumulative-details.md) | 所有已完成计划完整技术记录 |
