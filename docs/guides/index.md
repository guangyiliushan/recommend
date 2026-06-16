---
title: Guides
description: 技术指南、工程实践文章与专题说明目录
---

# Guides

## 目录目标

`docs/guides/` 用于沉淀工程实践、技术专题和方法指南。

它面向的是：

- 想落地工程实践的开发者
- 想复用推荐训练技巧的维护者
- 需要专题说明而不是契约文档的读者

## 推荐内容范围

适合放在这里的主题包括：

- 学习率调度
- 梯度检查点
- Host Device Info
- Online Dataset EDA
- profiling 与性能优化
- 多 GPU 运行建议

## 与其他目录的边界

- 若内容是“项目必须遵守的契约”，放到 `docs/project/`
- 若内容是“概念原理”，放到 `docs/concepts/`
- 若内容是“运维流程”，放到 `docs/operations/`
- 若内容是“论文解读”，放到 `docs/papers/`

## 推荐后续结构

```text
docs/guides/
|-- index.md
|-- learning-rate-scheduling.md
|-- gradient-checkpointing.md
|-- host-device-info.md
|-- online-dataset-eda.md
`-- profiling-and-optimization.md
```

## 页面模板建议

每个 guide 页面建议至少包含：

1. 主题目标
2. 适用场景
3. 当前项目中的落点
4. 推荐配置或做法
5. 风险与注意事项
6. 参考链接或关联文档

## 文件命名规范

统一使用：

- 小写
- 连字符
- 避免缩写歧义

推荐示例：

- `learning-rate-scheduling.md`
- `gradient-checkpointing.md`

## 当前阶段建议

当前项目最值得优先补的 guide 包括：

1. 学习率调度
2. 梯度检查点
3. Host Device Info
4. Online Dataset EDA

## 一句话总结

`docs/guides/` 负责回答“这件事怎么做更好”，而不是回答“项目必须遵守什么契约”。
