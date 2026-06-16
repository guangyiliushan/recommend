---
title: Getting Started
description: 本地环境、常用命令和文档构建方式
---

# Getting Started

## 环境约定

本仓库推荐统一使用 `uv` 管理 Python 环境与依赖，以保持和 GitHub Actions 一致。

## 安装依赖

```bash
uv sync --extra dev
```

这一步会安装开发、测试和文档构建所需的依赖。

## 常用命令

运行测试：

```bash
uv run pytest -v
```

运行静态检查：

```bash
uv run ruff check .
```

构建文档站点：

```bash
uv run zensical build --strict --clean
```

## 仓库重要目录

- `src/recsys`: 主业务代码
- `configs`: 配置入口
- `scripts`: 用户执行脚本
- `tests`: 测试目录
- `docs`: 文档站点源码
- `.github/workflows`: 自动化工作流

## 当前推荐的使用方式

如果你只是希望理解或维护项目，建议优先：

1. 阅读 `README.md`
2. 阅读 `docs/concepts/architecture.md`
3. 查看 `src/recsys/core` 与 `src/recsys/data`

如果你希望开始开发，建议先从以下模块着手：

- `src/recsys/core/registry.py`
- `src/recsys/core/base_dataset.py`
- `src/recsys/core/base_model.py`
- `src/recsys/pipeline/experiment.py`
