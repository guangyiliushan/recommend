---
title: Model Integration Guide
description: 当前模型契约、注册元信息与真实实现范围
---

# Model Integration Guide

## 目标

RecBench 的模型层文档重点不是列出所有计划支持的算法，而是明确：

- 模型如何注册
- 模型如何声明能力
- 模型需要哪些输入特征
- 模型如何产出统一预测结果
- 当前哪些模型已经真实可运行

## 当前模型契约

当前模型层围绕下面几个核心对象构建：

- `BaseRecommender`
- `NeuralRecommender`
- `Capability`
- `Batch`
- `ModelOutput`
- `PredictionBundle`

这意味着模型接入的核心不是继承一个“万能基类”，而是遵守最薄的通用契约，并在需要时显式声明能力。

## 当前模型发现入口

当前推荐通过 `recsys.models` 或 `recsys` 顶层入口访问模型发现 API：

- `auto_discover_models()`
- `get_model(name)`
- `get_model_metadata(name)`
- `list_models()`
- `list_models_by_family(family)`
- `list_models_by_task_type(task_type)`

示例：

```python
from recsys.models import auto_discover_models, get_model

auto_discover_models()
ItemCF = get_model("itemcf")
model = ItemCF(similarity="cosine", top_k_neighbors=50, recommend_k=10)
```

## 注册元信息

当前模型注册表元信息至少应包括：

- `name`
- `family`
- `year`
- `task_type`
- `supports_training`
- `required_features`
- `default_metrics`

这些元信息会被用于：

- 运行时实例化
- 文档与调试输出
- 模型筛选
- 任务类型判断

## 当前模型家族目录

仓库当前按家族组织模型目录：

- `classical`
- `deep_ctr`
- `sequence`
- `feature_cross`
- `pcvr`
- `unified`
- `generative`

但目录存在不等于该家族已经实现完毕。文档必须把“目录结构”与“真实可运行模型”区分开。

## 当前真实实现范围

### 已明确完成并可写入文档的模型

当前已实现并可通过训练型路径或非训练路径运行的模型是：

- `itemcf` — 非训练式协同过滤基线
- `dssm` — 双塔神经网络，首个可训练模型

`itemcf` 具有以下特征：

- 家族：`classical`
- `task_type = "ranking"`
- `problem_type = "implicit_ranking"`
- `supports_training = False`
- 已实现 `fit()` 与 `predict()`
- 能产出标准 `PredictionBundle`

`dssm` 具有以下特征：

- 家族：`classical`
- `task_type = "pointwise"`
- `problem_type = "binary"`
- `supports_training = True`
- 继承 `NeuralRecommender`，实现 `forward()` 和 `compute_loss()`
- 双塔结构：user embedding + MLP → item embedding + MLP → cosine similarity
- 训练时使用 BCE loss
- 可通过 `run_experiment()` 的训练型路径完整运行

### 当前仅目录预留或占位的模型

下列经典模型文件当前仍是 TODO 或空文件，不应写成已支持：

- `matrix_factorization.py`
- `factorization_machine.py`
- `gru4rec.py`
- `model_based_cf.py`
- `user_based_cf.py`

同样，其他模型家族下的大量文件也应默认视为“目录预留或占位”，除非其代码真实完成并完成注册。

## `itemcf` 使用示例

```python
from recsys.models import auto_discover_models, get_model

auto_discover_models()
ItemCF = get_model("itemcf")

model = ItemCF(
    similarity="cosine",
    top_k_neighbors=50,
    recommend_k=10,
    normalize=True,
)
```

当前 `itemcf` 的关键参数包括：

- `similarity`：`cosine` 或 `iuf`
- `top_k_neighbors`
- `recommend_k`
- `normalize`

## 模型接入要求

新增模型时，当前至少应满足：

1. 通过注册表完成注册
2. 明确声明 `task_type`、`supports_training` 和 `required_features`
3. 能输出标准化预测结果
4. 与 evaluator 契约兼容
5. 有最基本的测试或最小运行验证

## 非训练模型与训练模型

### 非训练模型

适合像 `itemcf` 这样：

- 直接实现 `fit()`
- 直接实现 `predict()`
- 不依赖 Lightning 训练循环

### 训练模型

适合像未来的 DeepFM、DIN、SASRec 这类神经模型：

- 继承 `NeuralRecommender`
- 实现 `forward()` 和 `compute_loss()`
- 通过 training 基础设施接入 Trainer

当前文档必须明确：训练基础设施已通过 `_execute_trainable_path()` 接入 experiment 主流程，首个可训练模型 `dssm` 可作为完整样板的参考实现。

## 接入前检查

新增模型前请确认：

1. 数据层已提供所需特征
2. 模型输出可以转换为 `PredictionBundle`
3. 默认指标与任务类型一致
4. 配置参数有合理默认值
5. 若模型是训练型模型，需同时考虑其与 training 主干的兼容性

## 当前推荐的接入顺序

结合仓库现状，当前更合理的接入顺序是：

1. 继续以 `itemcf` 为基线稳定 ranking 闭环
2. 接通一个训练型样板模型的 experiment 路径
3. 再逐步扩展 sequence、多任务和其他家族

## 当前最重要的结论

RecBench 的模型层当前已经拥有稳定的注册与契约框架，且已通过 `itemcf`（非训练）和 `dssm`（训练）验证了两类模型在统一主干内的共存能力。但“模型家族目录很多”不等于“模型实现很多”，文档必须把已实现模型与大量目录预留/占位文件严格区分开。
