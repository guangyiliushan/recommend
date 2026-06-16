---
title: Public API
description: RecBench 公共 Python API 与 CLI 契约、参数、返回格式、错误模型和示例
---

# Public API

## 目标

当前仓库尚未提供 HTTP 服务型 API，因此这里定义的“公共 API”是：

- 稳定的 Python 编程接口
- 面向用户的 CLI 入口
- 后续可演进为服务接口的统一契约

这份文档的目标是明确哪些接口可以被视为项目公共入口、它们的参数与返回、异常语义和调用方式。

## API 分层

当前公共 API 建议分为三层：

- Core API：注册表与基础抽象
- Runtime API：数据、评估、实验、benchmark 主干
- CLI API：`scripts/` 下的用户入口

## 适用范围说明

当前仓库中，真正已有源码骨架支撑的公共 API 主要包括：

- `Registry`
- `BaseDataset`
- dataset registration side effects
- 计划中的 `run_experiment()`
- 计划中的 `run_benchmark()`
- 计划中的 evaluator pipeline

其中部分接口已经存在代码定义，部分接口仍是仓库明确承诺但尚未完全实现的目标契约。文档中会明确区分两者。

## 统一错误模型

由于当前项目主要是 Python API，而不是 HTTP API，错误信息以异常和状态文件为主。

推荐统一使用如下错误模型语义：

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | `str` | 稳定错误码 |
| `phase` | `str` | 发生阶段，如 `config` / `data` / `training` |
| `message` | `str` | 人类可读错误说明 |
| `details` | `dict` | 附加上下文 |
| `hint` | `str \| null` | 推荐修复建议 |

### 推荐错误码

| 错误码 | 说明 |
|---|---|
| `CONFIG_VALIDATION_ERROR` | 配置校验失败 |
| `REGISTRY_ITEM_NOT_FOUND` | 注册表项不存在 |
| `REGISTRY_DUPLICATE_ITEM` | 注册重复 |
| `DATASET_NOT_LOADED` | 数据集尚未完成 `load()` |
| `DATASET_SPLIT_NOT_FOUND` | 请求的 split 不存在 |
| `MODEL_CONTRACT_ERROR` | 模型契约不满足要求 |
| `EVALUATION_CONTRACT_ERROR` | 评估输入不符合约定 |
| `ARTIFACT_WRITE_ERROR` | 结果或制品写入失败 |
| `BENCHMARK_RESUME_ERROR` | benchmark 恢复状态异常 |

当前代码里尚未统一实现这些错误码，但推荐后续逐步把异常和状态文件收敛到这套语义。

## Core API

## `Registry`

源码位置：

- `src/recsys/core/registry.py`

### 作用

- 注册模型、数据集、指标和 loss
- 提供按名称查找与按元信息筛选
- 提供基于导入副作用的 auto-discovery

### 构造函数

```python
Registry(name: str)
```

### 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | `str` | 是 | 注册表名称，如 `model`、`dataset` |

### 返回

- `Registry` 实例

### 可能异常

| 异常 | 推荐错误码 | 触发条件 |
|---|---|---|
| `KeyError` | `REGISTRY_DUPLICATE_ITEM` | 注册重复项 |
| `KeyError` | `REGISTRY_ITEM_NOT_FOUND` | 获取不存在的项 |

## `Registry.register()`

```python
register(name: str, **metadata: Any) -> Callable[[Type[T]], Type[T]]
```

### 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | `str` | 是 | 注册名称 |
| `metadata` | `dict` | 否 | 元信息，如 `family`、`task_type` |

### 返回

- 一个装饰器，用于注册类

### 示例

```python
from recsys.core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register(
    "itemcf",
    family="classical",
    task_type="ranking",
    supports_training=False,
)
class ItemBasedCF:
    ...
```

## `Registry.get()`

```python
get(name: str) -> Type[Any]
```

### 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | `str` | 是 | 注册名称 |

### 返回

- 已注册的类对象

### 可能异常

| 异常 | 推荐错误码 | 触发条件 |
|---|---|---|
| `KeyError` | `REGISTRY_ITEM_NOT_FOUND` | 名称未注册 |

### 示例

```python
ModelCls = MODEL_REGISTRY.get("itemcf")
model = ModelCls(...)
```

## `Registry.get_metadata()`

```python
get_metadata(name: str) -> Dict[str, Any]
```

### 返回

- 指定注册项的元信息字典

## `Registry.list()`

```python
list() -> List[str]
```

### 返回

- 当前注册表所有名称，按字典序排序

## `Registry.list_by()`

```python
list_by(key: str, value: Any) -> List[str]
```

### 用途

- 按元信息筛选注册项

### 示例

```python
ranking_models = MODEL_REGISTRY.list_by("task_type", "ranking")
```

## `Registry.auto_discover()`

```python
auto_discover(package_path: str) -> int
```

### 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `package_path` | `str` | 是 | Python 包路径，如 `recsys.models` |

### 返回

- 成功导入的模块数量

## Dataset Runtime API

## `BaseDataset`

源码位置：

- `src/recsys/core/base_dataset.py`

### 作用

- 定义统一数据集 adapter 契约
- 封装 `load()`、`get_split()`、`get_dataloader()` 等公共行为

### 构造函数

```python
BaseDataset(
    root_dir: str = "./data",
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    max_seq_len: int = 50,
    min_seq_len: int = 2,
    neg_sample_count: int = 4,
    **kwargs: Any,
)
```

### 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `root_dir` | `str` | 否 | 数据根目录 |
| `split_ratios` | `Tuple[float, float, float]` | 否 | 训练、验证、测试切分比例 |
| `max_seq_len` | `int` | 否 | 最大序列长度 |
| `min_seq_len` | `int` | 否 | 最小序列长度 |
| `neg_sample_count` | `int` | 否 | 负采样数量配置 |
| `kwargs` | `dict` | 否 | 数据集专属参数 |

### 子类必须实现

- `dataset_name`
- `dataset_url`
- `feature_cols`
- `label_col`
- `num_users`
- `num_items`
- `_load_raw()`
- `_prepare_splits()`
- `__len__()`
- `__getitem__()`

## `BaseDataset.load()`

```python
load() -> BaseDataset
```

### 作用

- 执行完整数据加载流程
- 内部调用 `_load_raw()` 和 `_prepare_splits()`

### 返回

- 当前 dataset 实例自身

### 可能异常

| 异常 | 推荐错误码 | 触发条件 |
|---|---|---|
| 任意加载异常 | `CONFIG_VALIDATION_ERROR` 或数据层细分错误 | 数据源不可用、参数异常等 |

### 示例

```python
from recsys.data.datasets.taac2026 import TAAC2026DataSample

dataset = TAAC2026DataSample(root_dir="./data").load()
```

## `BaseDataset.get_split()`

```python
get_split(split: str) -> Dataset[Any]
```

### 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `split` | `str` | 是 | `train` / `val` / `test` / `full` |

### 返回

- 对应 split 的 `Dataset`

### 可能异常

| 异常 | 推荐错误码 | 触发条件 |
|---|---|---|
| `ValueError` | `DATASET_SPLIT_NOT_FOUND` | split 名非法 |
| `RuntimeError` | `DATASET_NOT_LOADED` | 尚未调用 `load()` |

## `BaseDataset.get_dataloader()`

```python
get_dataloader(
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs: Any,
) -> DataLoader
```

### 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `split` | `str` | 否 | 目标 split |
| `batch_size` | `int` | 否 | batch 大小 |
| `num_workers` | `int` | 否 | DataLoader worker 数量 |
| `shuffle` | `bool` | 否 | 是否打乱，仅训练 split 生效 |
| `kwargs` | `dict` | 否 | 透传给 `DataLoader` 的参数 |

### 返回

- `torch.utils.data.DataLoader`

### 示例

```python
loader = dataset.get_dataloader("train", batch_size=256, num_workers=4)
```

## Dataset Registration API

源码位置：

- `src/recsys/data/dataset_registry.py`

### 作用

- 导入各个 dataset adapter，触发注册副作用

### 当前契约

当前该模块以显式导入方式工作，不提供复杂参数接口。

推荐调用方式：

```python
import recsys.data.dataset_registry  # noqa: F401
from recsys.core.registry import DATASET_REGISTRY

print(DATASET_REGISTRY.list())
```

## Runtime Pipeline API

## `run_experiment()`

源码位置：

- `src/recsys/pipeline/experiment.py`

### 当前状态

- 仓库已明确承诺此公共入口
- 当前尚未完成具体实现

### 推荐签名

```python
run_experiment(config: RecBenchConfig) -> ExperimentResult
```

### 输入参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `config` | `RecBenchConfig` | 是 | 完整实验配置对象 |

### 推荐返回格式

| 字段 | 类型 | 说明 |
|---|---|---|
| `run_id` | `str` | 唯一实验标识 |
| `status` | `str` | `succeeded` / `failed` / `skipped` |
| `summary_metrics` | `dict` | 关键指标摘要 |
| `artifact_paths` | `dict` | 配置、日志、指标、checkpoint 等路径 |
| `error` | `dict \| null` | 失败时的结构化错误 |

### 推荐错误码

- `CONFIG_VALIDATION_ERROR`
- `REGISTRY_ITEM_NOT_FOUND`
- `MODEL_CONTRACT_ERROR`
- `EVALUATION_CONTRACT_ERROR`
- `ARTIFACT_WRITE_ERROR`

### 推荐示例

```python
from recsys.pipeline.experiment import run_experiment
from recsys.utils.config import RecBenchConfig

result = run_experiment(config)
print(result["status"])
print(result["summary_metrics"])
```

## `run_benchmark()`

源码位置：

- `src/recsys/pipeline/benchmark.py`

### 当前状态

- 仓库已明确承诺此公共入口
- 当前尚未完成具体实现

### 推荐签名

```python
run_benchmark(config: RecBenchConfig) -> BenchmarkResult
```

### 推荐返回格式

| 字段 | 类型 | 说明 |
|---|---|---|
| `benchmark_name` | `str` | benchmark 名称 |
| `runs` | `list` | 各单实验结果摘要 |
| `summary_path` | `str` | 汇总 CSV 路径 |
| `leaderboard_path` | `str` | 排行榜路径 |
| `failures_path` | `str` | 失败列表路径 |
| `status` | `str` | `succeeded` / `partial_success` / `failed` |

### 推荐错误语义

- 单次 run 失败不应阻塞整个 benchmark
- benchmark 级错误应仅用于配置展开失败、结果目录异常、聚合失败等问题

## Evaluator API

源码位置：

- `src/recsys/evaluation/evaluator.py`

### 当前承诺

注释中已明确三类对外行为：

- `collect_predictions(model, dataloader)`
- `evaluate_model(model, dataloader, config)`
- `generate_curves(y_true, y_score)`

### 推荐输入

- 统一的 `PredictionBundle`
- 或模型 + dataloader + evaluation config 的组合

### 推荐返回

- `summary_metrics`
- `task_metrics`
- `curve_artifacts`
- `group_metrics`
- `metadata`

详细评估契约请参考 [Evaluation Guide](evaluation.md)。

## CLI API

## `scripts/run_single.py`

源码位置：

- `scripts/run_single.py`

### 当前状态

- 入口骨架已存在
- 推荐长期成为单实验命令行入口

### 推荐用法

```bash
uv run python scripts/run_single.py --config configs/experiment/single.yaml
uv run python scripts/run_single.py model=deepfm dataset=taac2026
```

### 推荐职责

- 接收 CLI 参数
- 调用配置加载器
- 调用 `run_experiment()`

### 不应承担

- 直接实现实验主逻辑

## `scripts/run_benchmark.py`

源码位置：

- `scripts/run_benchmark.py`

### 当前状态

- 入口骨架已存在
- 推荐长期成为批量 benchmark 命令行入口

### 推荐用法

```bash
uv run python scripts/run_benchmark.py --config configs/experiment/benchmark_classical.yaml
uv run python scripts/run_benchmark.py experiment=benchmark_classical runtime=ci
```

### 推荐职责

- 接收 benchmark 配置
- 调用 `run_benchmark()`
- 输出进度与结果摘要

## 返回格式规范

对所有公共运行时 API，推荐返回结果遵守以下原则：

- 优先返回结构化对象，而不是裸字符串
- 主结果与错误信息分离
- artifact 路径显式暴露
- 需要汇总的指标单独放在 `summary_metrics`

## 调用示例

### 查询模型注册表

```python
from recsys.core.registry import MODEL_REGISTRY

available = MODEL_REGISTRY.list()
print(available)
```

### 加载数据集并构造 DataLoader

```python
from recsys.data.datasets.taac2026 import TAAC2026DataSample

dataset = TAAC2026DataSample(root_dir="./data").load()
train_loader = dataset.get_dataloader("train", batch_size=256)
```

### 运行单实验

```python
result = run_experiment(config)
if result["status"] != "succeeded":
    print(result["error"])
```

### 运行 benchmark

```python
benchmark_result = run_benchmark(config)
print(benchmark_result["leaderboard_path"])
```

## 兼容性与版本化规则

公共 API 一旦被文档声明，就应遵守以下规则：

- 新增字段优先向后兼容
- 删除字段必须经过版本说明
- 错误码语义不得随意复用
- CLI 入口参数变更必须同步更新文档与示例

## 需要避免的反模式

请尽量避免：

- 把内部私有函数当作公共 API 宣传
- 返回结构随版本频繁变化
- 没有错误语义，只抛出不明异常
- CLI 和 Python API 契约不一致
- 文档承诺与源码骨架不一致

## 一句话总结

对 RecBench 来说，最佳实践不是“先写一堆调用示例”，而是：

- 先收敛稳定的 Python 与 CLI 公共入口
- 再统一参数、返回与错误语义
- 用文档把当前已承诺的 API 边界固定下来
