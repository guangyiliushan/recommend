---
title: Operations Overview
description: 运维与工具目录、缓存清理、日志管理与本地站点生成总览
---

# Operations Overview

## 目录目标

`docs/operations/` 负责沉淀项目维护、工具使用和运维流程。

它主要回答：

- 项目如何维护
- 文档站如何生成
- 缓存、日志、结果目录如何管理
- 哪些流程属于维护者职责

## 当前已有内容

当前该目录已有：

- [Maintenance Guide](maintenance.md)

它主要覆盖 CI、发布、文档部署和维护优先级。

## 推荐后续结构

建议逐步演进到：

```text
docs/operations/
|-- overview.md
|-- maintenance.md
|-- cache-cleanup.md
|-- log-management.md
`-- local-site-overview.md
```

## 各页面职责

### `maintenance.md`

职责：

- CI、发布、文档部署、维护优先级

### `cache-cleanup.md`

职责：

- 说明如何清理本地缓存、依赖缓存、实验输出缓存

### `log-management.md`

职责：

- 说明日志目录规范、保留策略、错误排查入口

### `local-site-overview.md`

职责：

- 说明如何本地生成文档站、如何校验导航与链接

## 推荐维护主题

当前最值得优先补的 operations 页面包括：

1. 仓库缓存清理
2. 仓库日志管理
3. 本地生成站点总览

## 文件命名规范

统一使用小写与连字符：

- `cache-cleanup.md`
- `log-management.md`
- `local-site-overview.md`

## 运维文档边界

运维文档不应承担：

- 模型原理论文解读
- 数据契约定义
- API 契约定义

这些内容分别应回到 `papers/`、`project/`、`concepts/`。

## 一句话总结

`docs/operations/` 的职责是让维护流程稳定、可执行、可重复，而不是替代项目架构或实验文档。
