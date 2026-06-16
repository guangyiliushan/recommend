---
title: Guides
description: 工程实践、调优技巧与专题说明目录
---

# Guides

## 目录目标

`docs/guides/` 用于沉淀“如何更好地使用当前基础设施”的专题说明。

它更适合回答：

- 这件事在当前仓库里怎么做
- 哪些配置组合更合适
- 有哪些限制和注意事项

而不是回答公共 API 或工程契约本身。

## 当前最值得扩展的主题

结合当前代码实现，最适合优先补成指南页的主题包括：

- 学习率调度与 warmup 选择
- 设备选择与 `get_device()` 使用方式
- profiling 与性能分析
- 日志与追踪后端使用方式
- 多 GPU 与 `ddp` 运行建议

这些主题已经有真实代码落点，不是空泛预留。

## 当前不应误写成指南页的内容

下面这些内容更适合留在其他目录：

- 公共契约：放 `docs/project/`
- 架构原则：放 `docs/concepts/`
- 运维流程：放 `docs/operations/`
- 论文背景：放 `docs/papers/`

## 推荐页面模板

每个 guide 页面建议至少包含：

1. 主题目标
2. 当前仓库中的适用范围
3. 推荐配置或 API 用法
4. 常见误区
5. 风险与注意事项
6. 关联文档

## 文件命名规范

- 使用小写
- 使用连字符
- 名称直接反映主题

例如：

- `learning-rate-scheduling.md`
- `device-selection.md`
- `profiling-and-optimization.md`

## 当前最重要的结论

`docs/guides/` 当前应聚焦“围绕已实现基础设施给出实践建议”，而不是为尚未落地的功能提前建立指南目录。
