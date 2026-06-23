---
title: Project Structure
description: 当前仓库目录结构、职责边界与新增文件落点规则
---

# Project Structure

## 概述

RecBench 的目录结构需要同时服务于三个目标：运行时边界清晰、文档与代码同步维护、新增模型/数据集/实验时的可扩展性。本页描述当前仓库已经采用的真实结构与实际职责，不是脱离实现的理想规划。

## 顶层目录

当前顶层布局如下：

```
.
├── configs/                  # Hydra 配置树
├── docs/                     # MkDocs 文档站源码
├── scripts/                  # CLI 入口与工具脚本
├── src/recsys/               # 唯一 Python 包真相源
├── tests/                    # 测试
├── outputs/                  # 实验与 Benchmark 结果（不入版本控制）
├── .github/workflows/        # CI/CD 自动化
├── pyproject.toml            # 项目元信息与依赖
├── uv.lock                   # uv 锁定文件
├── README.md                 # 项目入口文档
└── CONTRIBUTING.md           # 贡献指南
```

## 顶层职责

### `configs/`

保存 Hydra 配置树，按功能分组组织：

```
configs/
├── config.yaml                        # 主配置文件（Hydra 默认入口）
├── dataset/
│   ├── movielens_1m.yaml
│   ├── taac2025.yaml
│   └── taac2026.yaml
├── experiment/
│   ├── benchmark_all.yaml             # 活跃
│   ├── benchmark_classical.yaml       # 活跃
│   ├── benchmark_deep_ctr.yaml        # 活跃
│   ├── benchmark_feature_cross.yaml   # planned
│   ├── benchmark_pcvr.yaml            # planned
│   ├── benchmark_sequence.yaml        # planned
│   └── benchmark_unified_gen.yaml     # planned
└── model/
    ├── README.txt
    └── classical/
        └── itemcf.yaml
```

不应放入 Python 业务实现、实验产物或原始数据文件。

### `docs/`

保存 MkDocs 文档站源码，按功能领域分目录：

```
docs/
├── index.md                  # 文档站首页
├── getting-started.md        # 快速入门
├── concepts/                 # 概念与设计
│   ├── architecture.md
│   └── configuration.md
├── project/                  # 工程文档（API、管线、评估等）
│   ├── api-contracts.md
│   ├── artifacts.md
│   ├── benchmarking.md
│   ├── datasets.md
│   ├── development.md
│   ├── evaluation.md
│   ├── models.md
│   ├── pipeline.md
│   └── structure.md
├── experiments/              # 实验说明目录
│   └── index.md
├── analysis/                 # 数据集 EDA 自动分析报告
│   ├── index.md
│   └── dataset-eda/          # 按数据集分子目录
│       ├── movielens_1m/
│       ├── taac2025_1m/
│       ├── taac2026_data_sample/
│       └── taac2026_second_round/
├── assets/                   # 静态资源（EDA 图表 JSON）
│   ├── echarts-loader.js
│   └── figures/eda/          # 按数据集分子目录的 ECharts JSON
├── guides/                   # 使用指南
│   └── index.md
├── papers/                   # 论文引用与参考
│   └── index.md
└── operations/               # 运维文档
    ├── overview.md
    └── maintenance.md
```

不应放入自动生成结果或与站点无关的临时草稿。EDA 报告和图表通过 `analysis/` 和 `assets/` 目录按数据集隔离存储，遵循命名空间分离原则。

### `scripts/`

承载所有 CLI 实验入口与工具脚本。当前共七个脚本：

| 脚本                         | 用途                 | 关键能力                                                               |
| :--------------------------- | :------------------- | :--------------------------------------------------------------------- |
| `run.py`                     | Hydra 主入口         | YAML 组合 + CLI 覆盖，六子树桥接为 Pipeline 配置                       |
| `run_single.py`              | 单实验 argparse 入口 | 双模式（argparse + Hydra），支持 `--split-mode` 和 `--min-action-type` |
| `run_benchmark.py`           | 批量 Benchmark       | 三种运行模式、四种恢复策略、受控并发                                   |
| `run_ablation.py`            | 消融实验矩阵         | 预置 ItemCF 四组参数空间，输出 CSV 对比表                              |
| `download_data.py`           | HuggingFace 数据下载 | 维护注册名到仓库 ID 映射，本地缓存                                     |
| `benchmark_data_pipeline.py` | 数据管线性能基准     | 格式×压缩×后端全矩阵读写性能测量                                       |
| `generate_report.py`         | 报告生成             | 从 summary.csv 生成 Markdown 性能对比报告                              |

`train_hyformer.py` 是独立的模型结构调试脚本，使用随机内存数据直接训练，不走数据集适配器和 Pipeline。生产环境的训练应使用 `run_single.py` 或 `run.py`。

### `src/recsys/`

项目的唯一 Python 包真相源。所有核心契约与运行时主干必须落在此目录下。

### `tests/`

契约测试、回归测试、聚焦行为验证。当前结构如下：

```
tests/
├── test_data/                         # 数据层测试
│   ├── test_dataset_registry.py
│   ├── test_eda_*.py                  # EDA 各统计模块测试
│   ├── test_feature_engineering.py
│   ├── test_movielens_1m.py
│   ├── test_negative_sampling.py
│   ├── test_preprocessor.py
│   └── test_taac2026_split.py
├── test_models/                       # 模型测试
│   ├── test_hyformer.py
│   ├── test_itemcf.py
│   └── test_itemcf_benchmark.py
├── test_pipeline/                     # 管线测试
│   ├── test_benchmark.py
│   └── test_experiment.py
├── test_metrics/                      # 指标测试
│   ├── test_classification.py
│   └── test_ranking.py
├── conftest.py                        # pytest 共享 fixture
├── test_config_bridge.py              # 配置桥接测试
├── test_imports.py                    # 导入完整性测试
└── test_registry.py                   # 注册表测试
```

### `outputs/`

单实验结果（`outputs/runs/{run_id}/`）与批量 Benchmark 聚合结果（`outputs/benchmarks/{benchmark_name}/`）的输出目录。默认不建议纳入版本控制。

## `src/recsys/` 分层

当前主要分层及文件清单：

```
src/recsys/
├── __init__.py                         # 包入口（auto_discover_models 等）
├── core/                               # 公共契约
│   ├── base_dataset.py                 # 数据集抽象基类
│   ├── base_model.py                   # 模型抽象基类 + Batch 视图
│   ├── prediction_bundle.py            # 统一预测产物
│   └── registry.py                     # 注册表机制
├── data/                               # 数据层
│   ├── datasets/                       # 数据集适配器
│   │   ├── movielens_1m.py
│   │   ├── synthetic.py
│   │   ├── taac2025.py
│   │   └── taac2026.py
│   ├── eda/                            # 自动化 EDA 报告
│   │   ├── stats/                      # 统计子模块（7 个维度）
│   │   │   ├── distribution.py         # 分布分析
│   │   │   ├── effectiveness.py        # 有效性分析
│   │   │   ├── missing.py              # 缺失值分析
│   │   │   ├── overview.py             # 概览统计
│   │   │   ├── sequence.py             # 序列行为分析
│   │   │   ├── user_item.py            # 用户-物品分析
│   │   │   └── vector.py               # 向量分析
│   │   ├── cli.py                      # EDA CLI 入口
│   │   ├── context.py                  # 分析上下文
│   │   ├── report.py                   # 报告生成
│   │   ├── render.py                   # ECharts 渲染
│   │   └── sampler.py                  # 数据采样
│   ├── dataset_registry.py             # 数据集注册 + 后端/格式发现
│   ├── feature_engineering.py          # 特征工程（分块 + 向量）
│   ├── negative_sampling.py            # 负采样（五种策略 + 数据库后端）
│   ├── preprocessor.py                 # 离线预处理管线
│   └── split_utils.py                  # 公共序列切分工具
├── evaluation/                         # 评估层（四层分离）
│   ├── evaluator.py                    # 评估编排与路由
│   ├── metrics.py                      # 点式分类指标
│   ├── ranking.py                      # 分组排序指标
│   └── visualization.py                # 曲线与图表导出
├── models/                             # 模型层
│   ├── model_registry.py               # 模型注册发现入口
│   ├── classical/                      # 经典协同过滤（itemcf 已注册）
│   │   ├── item_based_cf.py
│   │   └── ...                         # 其余占位
│   ├── deep_ctr/                       # 深度 CTR（占位）
│   ├── feature_cross/                  # 特征交叉（占位）
│   ├── generative/                     # 生成式推荐（占位）
│   ├── pcvr/                           # 多任务 CVR（占位）
│   ├── sequence/                       # 序列推荐（占位）
│   └── unified/                        # 统一架构（hyformer 已注册）
│       ├── hyformer.py
│       └── ...                         # 其余占位
├── pipeline/                           # 管线层
│   ├── experiment.py                   # 单实验八阶段主干
│   ├── benchmark.py                    # 批量 Benchmark 调度
│   └── reporter.py                     # 结果聚合报告
├── training/                           # 训练层
│   ├── trainer.py                      # Lightning 训练封装
│   ├── callbacks.py                    # 回调组装
│   ├── distributed.py                  # 分布式策略解析
│   ├── losses.py                       # 损失函数库（9 种）
│   ├── optimizers.py                   # 优化器工厂（4 种）
│   └── schedulers.py                   # 调度器工厂（7 种）
└── utils/                              # 横切工具
    ├── config.py                       # 配置系统（Hydra + YAML + dataclass）
    ├── device.py                       # 设备探测与管理
    ├── logging.py                      # loguru 统一日志 + 实验追踪
    ├── profiling.py                    # 性能画像（参数/延迟/显存/FLOPs）
    ├── progress.py                     # 分层级进度追踪
    └── reproducibility.py              # 可复现性保障
```

## 各层职责详述

### `core/` — 公共契约

负责注册表机制、基础模型契约（`BaseRecommender` 和 `NeuralRecommender`）、统一预测产物 `PredictionBundle`、数据集抽象基类 `BaseDataset` 及其轻量包装 `SplitDataset`。这些是 Pipeline、训练层和评估层共同依赖的公共契约，不包含具体业务实现。

### `data/` — 数据层

负责四个数据集适配器（TAAC 2025/2026、MovieLens-1M、Synthetic）的加载与切分、数据集注册入口与能力发现（`dataset_registry.py` 维护后端/格式/压缩/负采样/特征工程全注册表）、公共序列切分工具 `SequenceSplit`（惰性序列切分，供多数据集复用）、离线预处理管线（chunk-aware 读取、列式格式转换、自动类型降级、多级缓存、指纹增量处理、数据库后端导出）、特征工程（`ChunkFeatureEngineer` 分块拟合与转换 + `VectorFeatureEngineer` 向量处理）、多策略负采样（uniform/popularity/in-batch/hard/mixed + PostgreSQL TABLESAMPLE 后端）以及自动化 EDA 报告生成（从概览到向量分析的七个统计维度，ECharts JSON 输出）。

`eda/` 子模块按职责细分：`stats/` 目录下七个文件各自负责一个统计维度，`cli.py` 提供 `recsys-dataset-eda` 命令行入口，`report.py` 与 `render.py` 负责报告和图表生成，`context.py` 维护分析配置上下文，`sampler.py` 负责大数据集预采样。

### `evaluation/` — 评估层

采用四层分离架构：`metrics.py` 实现点式分类指标的三层计算体系（原子统计层 → 派生指标层 → 阈值无关层），`ranking.py` 实现分组排序指标的逐组计算再跨组聚合，`evaluator.py` 根据 `task_type` 自动分流到合适的指标路径，`visualization.py` 消费已计算的结构化结果输出曲线 JSON 文件和可选图表。所有指标提供规范英文键名和中英文别名映射。

### `models/` — 模型层

按模型家族组织目录，通过 `model_registry.py` 实现模型自动发现与注册。模型家族目录完整（classical / deep_ctr / feature_cross / generative / pcvr / sequence / unified），但当前仅两个家族中的两个模型已完成注册并激活：`itemcf`（经典协同过滤，非训练 ranking 基线）和 `hyformer`（双塔 pointwise，已通过训练型路径验证）。其余约 50 个模型文件的注册装饰器处于注释状态。

### `pipeline/` — 管线层

边界清晰拆分为三个文件：`experiment.py` 负责单实验八阶段执行主干（含训练型和非训练双路径），`benchmark.py` 负责批量 Benchmark 的矩阵展开、四种恢复策略和受控并发调度，`reporter.py` 负责消费实验结果列表生成 CSV/JSON/HTML 聚合产物（含趋势和稳定性分析）。

### `training/` — 训练层

提供完整的 PyTorch Lightning 训练基础设施：`trainer.py` 将 `NeuralRecommender` 适配为 `LightningRecommender` 并通过 `TrainerFactory` 编译 `pl.Trainer`；`callbacks.py` 组装 Lightning 内置回调与自定义监控回调（梯度范数、GPU 显存、训练摘要）；`losses.py` 基于注册表提供九种损失函数（BCE/BCEWithLogits/CrossEntropy/BPR/InfoNCE/TOP1/Focal/MultiTask/AdaptiveHuber）；`optimizers.py` 支持四种优化器（Adam/AdamW/SGD/Adagrad）及参数组策略；`schedulers.py` 支持七种学习率调度策略（cosine/cosine_warmup/step/multi_step/plateau/onecycle/polynomial）；`distributed.py` 负责策略解析与兼容性检查（DDP 已就绪，FSDP/DeepSpeed 预留）。

### `utils/` — 横切工具

六个模块覆盖实验运行的通用关注点：`config.py` 实现 Hydra + YAML + dataclass 混合配置系统（六子树 dataclass、校验、快照、Pipeline 桥接、ConfigStore 注册）；`device.py` 实现自动设备探测与能力检查（CUDA/MPS/CPU + AMP/bf16 支持）；`logging.py` 基于 loguru 构建统一日志系统（终端/文件/追踪器三通道）；`profiling.py` 提供模型性能画像（参数量/推理延迟/GPU 显存/FLOPs）；`progress.py` 实现环境变量控制的分层级进度追踪（Benchmark 并发友好）；`reproducibility.py` 提供种子设置与确定性模式控制。

## 新文件落点规则

新增文件前按以下顺序判断：

1. 确定文档类型：概念、工程、实验、指南、论文还是运维
2. 确定代码层级：公共抽象还是具体实现
3. 确定管线层级：单实验逻辑还是批量 Benchmark 逻辑
4. 确定领域归属：数据、模型、训练、评估还是横切工具

具体落点规则：

| 新增内容        | 代码落点                                      | 文档落点                             |
| :-------------- | :-------------------------------------------- | :----------------------------------- |
| 新数据集适配器  | `src/recsys/data/datasets/{name}.py`          | `docs/project/datasets.md`           |
| 新数据集配置    | `configs/dataset/{name}.yaml`                 | `docs/project/datasets.md`           |
| 新模型          | `src/recsys/models/{family}/{name}.py`        | `docs/project/models.md`             |
| 新模型配置      | `configs/model/{family}/{name}.yaml`          | `docs/project/models.md`             |
| 新实验套件      | `configs/experiment/{name}.yaml`              | `docs/experiments/{name}.md`         |
| 新运行时工具    | `src/recsys/utils/` 或 `src/recsys/training/` | `docs/guides/` 或 `docs/operations/` |
| 新 EDA 分析维度 | `src/recsys/data/eda/stats/{name}.py`         | `docs/analysis/`                     |

## 需要避免的反模式

- 在 `scripts/` 中堆放核心业务逻辑——业务实现应在 `src/recsys/` 中
- 在 `models/` 中写数据预处理——预处理属于 `data/` 层
- 在 `data/` 中写模型专属逻辑——模型逻辑属于 `models/` 层
- 把实验产物混进源码目录——产物应在 `outputs/` 中
- 因目录预留而误写功能状态——目录存在不等于功能已实现
- 引用已过时的模块名或入口名——当前训练型模型示例为 `hyformer` 而非 `dssm`

## 当前现实边界

- 模型家族目录完整（7 个家族），但仅 `classical/itemcf` 和 `unified/hyformer` 已完成注册激活
- `configs/experiment/` 下三个配置文件处于活跃状态，四个为 planned（内容已注释）
- `configs/model/` 下仅 `classical/itemcf.yaml` 已落定，其余模型的 YAML 配置尚未组织
- `benchmark_data_pipeline.py` 和 `generate_report.py` 等工具脚本已完整实现，非"待完善"状态
- 文档目录与代码实现保持同步，分析报告和 EDA 图表按数据集命名空间隔离存储

## 未来展望

以下目录预留或结构规划尚未完全落地：

- `configs/model/` 下其余模型家族（deep_ctr/sequence/unified/feature_cross/pcvr/generative）的 YAML 配置文件待对应模型注册激活后补充
- `docs/experiments/` 下具体的实验页（如 `baseline-itemcf.md`、`baseline-hyformer.md`）待撰写
- `docs/guides/` 和 `docs/papers/` 的详细内容页待补充
- 四个 planned Benchmark 配置文件待模型就位后启用
