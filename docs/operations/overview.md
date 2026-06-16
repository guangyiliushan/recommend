---
title: Operations Overview
description: 仓库维护、文档站构建与结果目录管理总览
---

# Operations Overview

## 目录目标

`docs/operations/` 用于沉淀仓库维护和日常运维流程。

它主要回答：

- 仓库如何维护
- 文档站如何构建
- 输出目录如何管理
- 哪些流程属于维护者职责

## 当前重点

结合当前仓库状态，operations 目录最重要的关注点包括：

- 本地与 CI 中的依赖和检查命令保持一致
- 文档站构建始终可通过 `zensical build --strict --clean`
- `outputs/` 下实验结果与 Benchmark 聚合结果的管理
- 维护文档、README、配置和代码状态的一致性

## 与其他目录的边界

- 架构原则与概念解释：放 `docs/concepts/`
- API、artifact、pipeline 契约：放 `docs/project/`
- 工程实践技巧：放 `docs/guides/`
- 论文背景：放 `docs/papers/`

## 当前推荐补充的运维主题

- 输出目录清理与归档
- 日志与失败排查入口
- 本地文档站构建说明
- CI 与文档导航一致性检查

## 文件命名规范

- 使用小写
- 使用连字符
- 名称直接反映维护主题

## 当前最重要的结论

`docs/operations/` 当前应该围绕“如何稳定维护当前已实现主干”展开，而不是延展到架构、实验或论文层面的内容。
