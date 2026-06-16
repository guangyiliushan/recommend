---
title: Papers
description: 论文解读、方法背景与研究映射目录
---

# Papers

## 目录目标

`docs/papers/` 用于收纳与项目直接相关的论文解读、方法背景说明和研究映射。

这个目录的目标不是复制论文原文，而是回答：

- 这篇论文和项目哪一层有关
- 它对应仓库中的哪个模型或模块
- 对工程实现有什么启发或约束

## 推荐内容范围

适合放在这里的内容包括：

- PCVR 数据管道论文解读
- 多任务建模相关论文映射
- 序列推荐关键论文
- 生成式推荐代表工作梳理

## 推荐后续结构

```text
docs/papers/
|-- index.md
|-- pcvr-data-pipeline.md
|-- rankup-paper.md
|-- interformer-paper.md
|-- onetrans-paper.md
`-- unirec-paper.md
```

## 页面模板建议

每篇论文解读建议至少包含：

1. 论文基本信息
2. 核心问题与方法概述
3. 与 RecBench 的对应模块
4. 需要的数据字段
5. 对配置、训练或评估的影响
6. 当前仓库实现状态

## 命名规范

推荐统一采用：

- `{topic}-paper.md`
- 或 `{paper-short-name}.md`

避免：

- 带年份和版本号的杂乱命名
- 中文英文混杂且无固定规则

## 当前阶段建议

当前最值得优先补的论文页是：

1. `pcvr-data-pipeline.md`
2. `rankup-paper.md`
3. `interformer-paper.md`

## 一句话总结

`docs/papers/` 的职责是建立“论文概念到工程落点”的桥梁，而不是做纯学术摘抄。
