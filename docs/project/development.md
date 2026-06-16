---
title: Development Guide
description: 日常开发、提交流程和推荐的实现顺序
---

# Development Guide

## 开发原则

- 先修正契约和边界，再扩展模型数量
- 尽量保持每次提交只完成一个清晰目标
- 修改行为时同时更新文档
- 为关键契约增加测试，而不是只增加实现

## 推荐开发顺序

当前仓库最合理的开发顺序是：

1. `config`
2. `registry`
3. `dataset`
4. `model contract`
5. `experiment pipeline`
6. `trainer`
7. `evaluator`
8. `benchmark runner`

这样做的原因是，模型实现的规模会很大，而共享运行时如果不稳定，后续每个模型都会反复返工。

## 日常检查项

提交前至少执行：

```bash
uv run pytest -v
uv run ruff check .
uv run zensical build --strict --clean
```

## 配置与实现的一致性

需要特别注意以下一致性问题：

- 包路径应统一为 `src/recsys`
- 配置中引用的模型和数据集必须真实存在并可注册
- README、文档与 CI 行为必须一致
- 脚本入口与配置注释不能残留旧路径

## Pull Request 建议

一个好的 PR 应至少回答：

- 解决了什么问题
- 修改了哪些契约
- 对哪些目录有影响
- 如何验证
- 还有哪些后续工作

## 不建议现在优先做的事情

在共享运行时稳定前，不建议优先：

- 同时补很多模型实现
- 引入大量特殊分支逻辑
- 把 benchmark 设计成过于复杂的分布式系统
- 让某个模型依赖单独的 batch 格式而不更新公共契约
