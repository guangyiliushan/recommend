---
title: Model Integration Guide
description: 模型接入标准、元信息和建议样板
---

# Model Integration Guide

## 模型接入目标

RecBench 需要支持多个推荐范式，因此模型接入文档的重点不是规定某一种内部写法，而是统一以下外部契约：

- 如何注册
- 需要哪些输入特征
- 输出哪些标准字段
- 是否支持训练
- 使用哪些评估任务

## 当前模型家族

仓库当前按以下家族组织模型：

- `classical`
- `deep_ctr`
- `sequence`
- `feature_cross`
- `pcvr`
- `unified`
- `generative`

## 建议的注册元信息

每个模型至少应提供：

- `name`
- `family`
- `year`
- `task_type`
- `supports_training`
- `required_features`
- `default_metrics`

## 建议的接入顺序

不要从最复杂的模型开始。推荐顺序如下：

1. `ItemCF` 或 `MF`
2. 一个可训练的 tabular baseline，例如 `DeepFM`
3. 一个 sequence baseline
4. 一个多任务或生成式代表模型

这样可以用最少的实现验证最多的共享能力。

## 接入前检查

新增模型前，请确认：

1. 需要的数据字段已在 dataset adapter 中可用
2. 模型输出可以被 evaluator 消费
3. 配置参数有清晰默认值
4. 失败时能提供可读错误信息
5. 至少有一条聚焦测试

## 当前阶段的推荐策略

在当前项目阶段，模型文档要强调两点：

- 模型是“计划支持”还是“已经可运行”
- 该模型依赖的共享基础设施是否已经落地

避免在 README 或文档里把目录中的占位文件描述成已完成实现。
