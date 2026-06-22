# Contributing to RecBench

感谢你关注 RecBench。

当前仓库最有价值的贡献，不是一次性补很多模型，而是继续增强共享契约、最小可运行闭环、文档准确性和工程一致性。

## 开始前先阅读

请至少先阅读以下文档：

- `README.md` — 项目概览与当前状态
- `docs/index.md` — 文档站索引
- `docs/concepts/architecture.md` — 系统架构
- `docs/concepts/configuration.md` — 配置系统（Hydra + YAML + dataclass）
- `docs/project/development.md` — 开发规范
- `docs/project/models.md` — 模型集成指南
- `docs/project/pipeline.md` — 单实验执行管线

## 当前协作原则

- 统一使用 `uv` 管理 Python 环境与依赖，所有命令以 `uv run` 为前缀
- 所有核心实现都应围绕包根 `src/recsys`
- 优先提交小而可审阅的 PR
- 不要把占位模块或规划能力描述成"已完成"
- 改变公共行为、配置或文档入口时，要同步更新文档
- 改动影响契约或运行行为时，补充对应测试或说明暂未补测的原因
- 新增模型前请确保注册元信息完整并与现有任务契约兼容
- 配置中的模型名和数据集名必须引用已注册名称

## 当前优先级

优先级更高的贡献方向：

- 配置体系与配置校验收敛
- Registry 与公共 API 收敛
- 数据集适配器稳定性增强（含 Dense ID Remap、split_mode、min_action_type）
- 评估器与产物契约完善
- 实验管线与 Benchmark 主干完善
- 文档准确性修正与示例更新
- 测试覆盖补充

当前优先级较低的方向：

- 在共享运行时未稳定前一次性补很多模型
- 大范围重命名或无边界重构
- 绕过配置、注册表或公共契约的功能扩展

## 本地环境

```bash
uv sync --extra dev
```

常用命令：

```bash
uv run ruff check .           # 静态检查
uv run pytest -v              # 运行测试
uv run zensical build --strict --clean  # 构建文档站
```

当前测试覆盖仍在逐步建设中，因此"测试通过"不代表所有路径都已经完备；请结合代码审查与文档一致性一起验证。

## PR 建议

一个好的 PR 需要回答以下问题：

1. 解决了什么问题
2. 修改了哪些契约或行为
3. 影响了哪些模块或文档
4. 如何验证
5. 还存在哪些后续工作

推荐的 PR 粒度：

- 一个契约变更
- 一个运行时主干增强
- 一个数据适配器改进
- 一个已接通的模型样板接入
- 一组相互关联的文档修订

## 模型贡献要求

新增模型时，请至少完成以下步骤：

1. 明确定义注册表元信息（`family`、`modality`、`tasks`）
2. 确认模型与现有任务契约兼容（继承 `BaseRecommender` 或 `NeuralRecommender`）
3. 说明所需输入特征与数据格式要求
4. 说明输出字段与评估方式（rank 指标或分类指标）
5. 增加至少一条聚焦测试，覆盖注册、形状或运行时兼容性
6. 在 `configs/model/` 下添加对应的 YAML 配置文件（按模型家族分子目录）

如果模型需要新的 batch 字段、评估协议或训练分支，应先扩展公共契约，再接入具体模型。

如果模型使用 Embedding 层（如 `nn.Embedding`），需确保数据集已完成 Dense ID Remap（TAAC 2026 已内置），或模型侧已做好 ID 边界校验保护。`0` 在所有数据集中保留为填充槽位，不分配给真实用户或物品。

## 文档贡献要求

文档在本仓库中是一等公民。

如果你修改了开发者可见行为，请在同一个 PR 中同步更新：

- `README.md`
- `docs/` 中对应页面
- 如有需要，更新示例命令、配置片段与产物说明

文档必须遵守核心原则：**只描述当前仓库真实存在的实现能力**，并明确区分"已实现"、"部分实现"和"未实现"三种状态。禁止在文档中使用具体代码块来展示 API——使用自然语言描述功能逻辑和行为。

### 文档使用规范

- 使用自然语言描述功能逻辑和行为，不嵌入代码块
- 操作指南中的 shell 命令示例可以使用代码块格式
- 每个文档章节的标题与内容必须与源文件中的实现保持一致
- 未实现的功能归入"未来展望"章节统一保留

## 当前特别需要注意的问题

- 当前已注册可用的模型仅 `itemcf`（非训练、ranking）和 `hyformer`（训练型、pointwise），其余约 50 个模型文件的注册装饰器仍处于注释状态
- 已注册可用的数据集：`taac2026_data_sample`、`taac2026_second_round`、`taac2025_1M`、`taac2025_10M`、`movielens_1m`、`synthetic`
- `configs/experiment/` 下三个配置文件处于活跃状态（`benchmark_all.yaml`、`benchmark_classical.yaml`、`benchmark_deep_ctr.yaml`），四个为预留模板（planned，内容已注释）
- 引用 `dssm` 作为训练型模型示例已过时，当前训练模型为 `hyformer`
- 不要把模型目录中存在文件等同于模型已可运行——确认模型是否已注册（`MODEL_REGISTRY.list()`）
- 新增模型配置时遵循 `configs/model/{family}/{name}.yaml` 的目录结构

## 未来展望

以下贡献方向将在后续阶段逐步开放：

- 剩余约 50 个模型的注册激活与验证（DeepFM、SASRec、DCN、ESMM、HSTU 等）
- `criteo_kaggle`、`taobao_behavior` 等公开基准数据集适配器的实现
- 四个 planned Benchmark 配置文件的模型就位后启用
- FSDP 和 DeepSpeed 分布式策略的接入与验证
- 多模态嵌入的端到端训练 Pipeline

## 沟通建议

如果你计划做较大的架构调整，建议先开 issue 或 draft PR，对齐方向后再进入实现阶段。
