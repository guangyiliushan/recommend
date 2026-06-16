---
title: Experiments
description: 实验文档目录、benchmark 套件、baseline 页面与模型实验说明规范
---

# Experiments

## 目录目标

`docs/experiments/` 用于承接所有“实验面”的文档，而不是承接所有工程规范文档。

这个目录主要描述：

- benchmark 套件
- baseline 与 baseline+ 页面
- 重点模型实验说明
- 实验矩阵与比较规则

## 与其他目录的边界

- 概念与架构原理放在 `docs/concepts/`
- 数据、评估、pipeline、API、契约放在 `docs/project/`
- 工程实践文章放在 `docs/guides/`
- 论文笔记放在 `docs/papers/`
- 运维流程放在 `docs/operations/`

## 推荐后续结构

建议后续逐步演进到：

```text
docs/experiments/
|-- index.md
|-- benchmark-classical.md
|-- benchmark-performance.md
|-- baseline.md
|-- baseline-plus.md
|-- symbiosis.md
|-- rankup.md
|-- interformer.md
|-- onetrans.md
|-- tokenformer.md
`-- unirec.md
```

## 页面分类规则

### benchmark 套件页

适合描述：

- 这套实验跑哪些模型
- 这套实验跑哪些数据集
- 主指标是什么
- 如何复现

### baseline 页面

适合描述：

- 基线集合
- 推荐执行顺序
- 为什么这些基线足以构成最小对照组

### 单模型实验页

适合描述：

- 模型背景
- 当前仓库实现状态
- 所需数据字段
- 推荐配置
- 指标与结果解释方式

## 文件命名规范

统一使用小写加连字符：

- `benchmark-classical.md`
- `benchmark-performance.md`
- `baseline-plus.md`
- `interformer.md`

## 页面最小模板

每个实验页建议至少包含：

1. 页面目标
2. 适用模型或实验套件
3. 输入数据要求
4. 推荐配置
5. 评估方式
6. 输出 artifact
7. 常见失败与排查

## 当前阶段建议

当前仓库还未完成完整 benchmark runtime，因此这个目录应先用于：

- 固定实验命名
- 固定实验矩阵说明
- 防止未来每条实验线随意命名和混放

## 一句话总结

`docs/experiments/` 的目标不是堆放结果截图，而是把“要跑什么实验、怎么比较、怎么复现”固定成一套稳定页面体系。
