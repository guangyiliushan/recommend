---
title: Development Guide
description: 当前仓库的开发优先级、提交流程与一致性要求
---

# Development Guide

## 开发原则

- 先修正共享契约和边界，再扩展模型数量
- 每次提交尽量只解决一个清晰问题
- 行为变更时同步更新文档
- 为关键契约补测试，而不是只补实现
- 文档必须诚实反映仓库当前状态

## 当前推荐开发顺序

结合当前仓库实际完成度，更合理的推进顺序是：

1. `config` 与公共 API
2. registry 与模型/数据发现
3. dataset adapter 稳定性
4. `PredictionBundle` 与 evaluator 契约
5. 单实验 pipeline
6. 训练型 experiment 路径接线
7. Benchmark 聚合与恢复细化
8. 再逐步扩展模型覆盖

这样做的核心原因是：共享运行时越稳定，后续每新增一个模型的返工成本越低。

## 当前最值得投入的方向

优先级较高的方向包括：

- 接通训练型模型的 experiment 主路径
- 统一配置示例与真实代码字段
- 为 `itemcf` 和公共契约补充测试
- 继续修正文档、脚本与配置中的过时引用
- 收敛数据适配器与评估层的字段约定

## 当前不建议优先做的事情

在共享主干未完全稳定前，不建议优先：

- 一次性补很多模型实现
- 引入大量特殊分支逻辑
- 把 Benchmark 设计成复杂的分布式系统
- 让单个模型依赖独有 batch 格式却不回收为公共契约

## 日常检查项

提交前建议至少执行：

```bash
uv run ruff check .
uv run pytest -v
uv run zensical build --strict --clean
```

如果改动了文档、配置或公共 API，还建议额外检查：

- README 是否仍准确
- 文档内链接是否正确
- 示例命令是否真实可用
- 配置字段是否与 dataclass 定义一致

## 当前特别需要注意的一致性问题

- 包路径应统一围绕 `recsys`，不要残留旧入口
- 配置中的模型名与数据集名必须真实存在并可注册
- README、`docs/`、`configs/` 与代码状态必须一致
- `scripts/` 仍是骨架，不能在文档中写成稳定 CLI
- 目录预留不等于功能已实现

## PR 建议

一个好的 PR 最好回答：

- 解决了什么问题
- 修改了哪些契约或行为
- 影响了哪些模块和文档
- 如何验证
- 后续还剩什么工作

## 推荐的 PR 粒度

- 一个契约修正
- 一个运行时主干增强
- 一个数据适配器修正
- 一个已接通模型样板的集成
- 一组相关文档修订

## 文档与代码同源更新

当前仓库特别强调：

- 修改公共 API 时同步更新 `docs/project/api-contracts.md`
- 修改 experiment / benchmark 行为时同步更新 `pipeline.md`、`benchmarking.md`、`artifacts.md`
- 修改配置字段时同步更新 `configuration.md` 与 README 示例
- 修改功能状态时同步更新首页、README、CONTRIBUTING

## 当前最重要的结论

RecBench 当前最有价值的开发工作，是继续把“已完成的主干”打磨得更稳、更一致，而不是在未接通的运行时主路上快速堆叠更多模型文件。
