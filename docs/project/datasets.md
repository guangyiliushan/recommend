---
title: Dataset Guide
description: RecBench 数据层完整指南 — 数据集适配器、Split 语义、Dense ID Remap、预处理管线与数据层能力边界
---

# Dataset Guide

## 概述

RecBench 的数据层负责将原始数据源转换为可被模型和训练管线消费的 Dataset Adapter。它统一了多类数据集的加载、切分、特征工程与负采样流程，通过抽象基类规范适配器接口，并通过注册机制实现数据集的动态发现。

本文档面向需要理解数据层架构或接入新数据集的开发者，涵盖所有已落地的核心组件、数据集适配器、预处理能力及当前数据层边界。

## 核心抽象

### BaseDataset — 数据集抽象基类

所有数据集适配器均继承自 `BaseDataset`（位于 `src/recsys/core/base_dataset.py`），它定义了标准化的数据加载管线：

1. **加载阶段**：调用 `load()` 方法执行 `_load_raw()`（下载/读取原始数据）与 `_prepare_splits(raw)`（构造 train/val/test 三个切分），形成完整的数据集实例。
2. **消费阶段**：通过 `get_dataloader(split, batch_size, ...)` 获取 PyTorch DataLoader，或通过 `get_split(split)` 直接访问底层 Dataset 对象。

每个子类必须声明的元信息属性包括：数据集名称、来源 URL、特征列名列表、标签列名、用户总数与物品总数。此外，子类还可重写 `get_item_pool_stats()` 方法，为负采样模块提供当前过滤条件下的物品池统计信息。

### SequenceSplit — 惰性序列切分

`SequenceSplit`（位于 `src/recsys/data/split_utils.py`）是序列推荐场景的通用 Split 实现，供 TAAC 2025、MovieLens-1M 等用户序列格式数据集复用。

**设计原理**：存储紧凑的用户到物品序列映射（内存复杂度 O(用户数)），而非预展开为扁平样本列表（O(总交互数)）。每次 `__getitem__` 调用通过二分查找在 O(log n) 时间内定位用户，按需计算带标签的 (用户ID, 历史序列, 预测标签) 样本。

**关键参数**：
- `max_seq_len`：序列最大截断长度
- `neg_sample_count`：每个正样本对应的负样本数量
- `candidate_pool`：候选物品池，用于评估阶段的候选集约束

**快速提取接口**：提供 `iter_user_item_pairs_fast()` 和 `extract_user_item_mapping_fast()` 两个方法，支持在实验管线中以 O(用户数) 复杂度（而非 O(总位置数)）提取用户-物品对，供 ItemCF 等非训练模型使用。

### SplitDataset — 通用切分包装器

`SplitDataset`（位于 `src/recsys/core/base_dataset.py`）是一个轻量级的 Dataset 包装器，将内部可迭代对象（如列表或字典）的 `__len__` 和 `__getitem__` 方法直接代理出去，适用于 Synthetic 等简单数据集方案。

## 已实现的数据集适配器

### TAAC 2025 — 生成式推荐数据集

**源码**：`src/recsys/data/datasets/taac2025.py`

TAAC 2025 是腾讯广告算法大赛发布的全模态生成式推荐数据集，提供两个规模变体：

| 数据集名称    | 注册键         | 序列规模         | 候选池规模      | 核心能力                            |
| :------------ | :------------- | :--------------- | :-------------- | :---------------------------------- |
| TencentGR-1M  | `taac2025_1M`  | ~100 万用户序列  | ~66 万候选物品  | 多模态嵌入、候选池、Schema Manifest |
| TencentGR-10M | `taac2025_10M` | ~1000 万用户序列 | ~364 万候选物品 | 同上，面向大规模训练与评估          |

**数据子集**：TAAC 2025 通过 HuggingFace 分 config 发布，支持以下子集的按需独立加载（`load_subset(name)` 方法）：

| 子集名         | 类型     | 说明                                                                                              |
| :------------- | :------- | :------------------------------------------------------------------------------------------------ |
| `seq`          | 行为序列 | 用户行为序列主数据（`user_id`, `seq` 列），包含每条交互的 `item_id`、`action_type` 和 `timestamp` |
| `user_feat`    | 用户特征 | 用户画像与人口统计学特征                                                                          |
| `item_feat`    | 物品特征 | 物品类别与文本特征                                                                                |
| `candidate`    | 候选池   | 检索评估用的候选物品集合                                                                          |
| `mm_emb_text`  | 向量嵌入 | 文本模态的多模态嵌入向量                                                                          |
| `mm_emb_image` | 向量嵌入 | 图像模态的多模态嵌入向量                                                                          |
| `mm_emb_video` | 向量嵌入 | 视频模态的多模态嵌入向量                                                                          |

**核心特性**：

- **行为过滤**：通过 `min_action_type` 参数控制正交互阈值。`action_type=0` 表示曝光，`action_type=1` 表示点击。设置为 1 时仅提取点击作为正交互，适用于 ItemCF 等隐式推荐场景。
- **Schema Manifest 缓存**：`DatasetSchemaManifest` 在首次加载时查询 HuggingFace 元数据（行数估算、列信息、向量维度等），并缓存为本地 JSON。后续调用直接读取缓存，避免重复网络查询。可通过 `ds.get_schema_manifest()` 获取。
- **Arrow 列式优化**：序列预处理阶段使用 PyArrow 列式操作直接从嵌套的 Arrow 结构（`List<Struct<item_id, action_type, timestamp>>`）中提取字段，不经过 Python 字典反序列化，避免大规模数据下的内存膨胀。
- **本地序列缓存**：预处理后的紧凑用户序列映射缓存在本地 NPZ 文件中，首次处理后秒级重载。
- **VectorStore**：多模态嵌入子集以 `VectorStore` 结构返回，存储预采样向量、物品 ID 索引、向量维度和模态类型，支持 L2 范数计算、重复率检测和逐维度统计。
- **候选池持久化**：通过 `get_candidate_pool()` 方法加载候选物品池，首次加载后缓存为 `.pt` 文件。

**配置**：`configs/dataset/taac2025.yaml`

### TAAC 2026 — 广告点击率/转化率预估数据集

**源码**：`src/recsys/data/datasets/taac2026.py`

TAAC 2026 是腾讯广告算法大赛 2026 的轻量级样本数据集，用于本地点式预估（CTR/CVR）任务的快速原型与调试。提供两个变体：

| 数据集名称               | 注册键                  | 数据规模 | 特征列数  | 增量能力             |
| :----------------------- | :---------------------- | :------- | :-------- | :------------------- |
| data_sample_1000         | `taac2026_data_sample`  | 1,000 行 | 约 120 列 | 基础 CTR/CVR         |
| second_round_sample_1000 | `taac2026_second_round` | 1,000 行 | 约 142 列 | 新增 Domain 序列特征 |

**列分组体系**：TAAC 2026 的宽表列被自动识别为以下四组：

- **core**：`user_id`、`item_id`、`label_type`、`label_time`、`timestamp`
- **user_feat**：以 `user_int_`、`user_dense_`、`user_string_` 为前缀的用户侧特征列
- **item_feat**：以 `item_int_`、`item_dense_`、`item_string_` 为前缀的物品侧特征列
- **domain_seq**：以 `domain_` 为前缀的时序行为域序列（仅 second_round 变体）

**内部切分结构 `_TabularSplit`**：每个切分（train/val/test）为独立 `_TabularSplit` 实例。该结构在初始化时预计算各特征组的全局最大展平长度，并在 `__getitem__` 中对所有变长特征进行定长填充/截断，确保 DataLoader 的默认 collate 能够正确堆叠不同样本的 tensor。对于 Domain 序列列，同样预计算全局最大序列长度以保证 collate 一致性。

**切分模式（split_mode）**：TAAC 2026 支持两种数据切分策略：

- `"temporal"`（默认）：按 `timestamp` 字段升序排列后，训练集在前、验证集居中、测试集在后，用过去行为预测未来。这是推荐系统的标准时序切分方式。
- `"random"`：使用固定随机种子打乱数据后按比例分割，适用于 CTR/CVR 点式评估等不需要时序建模的场景。

**Dense ID Remap**：TAAC 2026 在 `_prepare_splits()` 中自动执行稠密 ID 重映射（详见下方 Dense ID Remap 章节）。

**快速迭代接口**：`_TabularSplit` 提供 `iter_user_item_pairs_fast()` 和 `extract_user_item_mapping_fast()` 方法，与 SequenceSplit 的同名方法保持语义一致，直接从原始字典提取数据，跳过张量构造开销。

**配置**：`configs/dataset/taac2026.yaml`

### MovieLens-1M — 经典序列推荐数据集

**源码**：`src/recsys/data/datasets/movielens_1m.py`

MovieLens-1M 是推荐系统领域的经典基准数据集，通过 RecZoo 社区提供的镜像格式发布。数据为 JSON 二维数组格式（每个用户的交互物品序列），已预切分为 train/val/test 三个独立 JSON 文件，不接受运行时的 split_ratios 参数。

| 数据集名称     | 注册键         | 规模                                     | 格式                 |
| :------------- | :------------- | :--------------------------------------- | :------------------- |
| Movielens1M_m1 | `movielens_1m` | 6,040 用户，约 3,900 物品，约 100 万交互 | RecZoo JSON 二维数组 |

**加载策略**：优先使用本地缓存文件，不存在时通过 HTTP 从 HuggingFace 镜像下载。不走 HF datasets API（因嵌套 JSON 格式导致 Viewer 解析失败）。

**短序列过滤**：通过 `min_seq_len` 参数过滤交互数不足的用户（默认最少 2 条交互）。

**EDA 兼容**：实现 `load_subset()` 方法，将预切分的用户序列展平为 `user_id × item_id` 长格式 DataFrame（兼容 TAAC 2025 的子集名映射，如 `"seq"` 和 `"behavior"` 均映射到 train），支持超出 `max_rows` 限制时的随机预采样。

**候选池**：从训练集所有物品中自动构建排序后的候选物品池，供评估阶段使用。

**配置**：`configs/dataset/movielens_1m.yaml`

### Synthetic — 合成交互数据集

**源码**：`src/recsys/data/datasets/synthetic.py`

用于性能基准测试的可控规模合成交互数据，无需外部下载，在内存中按幂律分布生成。可配置参数覆盖用户数、物品数、交互量、稀疏度和评分范围，适用于快速原型和模型结构验证。

| 参数               | 默认值    | 说明                                                           |
| :----------------- | :-------- | :------------------------------------------------------------- |
| `num_users`        | 10,000    | 用户数量                                                       |
| `num_items`        | 5,000     | 物品数量                                                       |
| `num_interactions` | 1,000,000 | 交互总数                                                       |
| `sparsity`         | 无        | 目标稀疏度（设置后自动覆盖 num_interactions）                  |
| `rating_scale`     | 无        | 不设置则为隐式交互；如 `(1, 5)` 则生成显式评分                 |
| `popularity_power` | 0.75      | 长尾分布的幂次参数（与负采样的 `popularity_power` 默认值一致） |
| `seed`             | 42        | 随机种子                                                       |

**切分策略**：按用户级别划分，保证每个用户在 train/val/test 中都有数据项。每个用户的物品列表先打乱，再按比例分配到各切分，最终包装为 `SplitDataset` 实例。

## Dense ID Remap

TAAC 2026 数据集在 `_prepare_splits()` 阶段自动执行稠密 ID 重映射，将原始稀疏 ID 压缩为从 1 开始的连续整数序列。具体流程如下：

1. 从全量原始数据中收集所有唯一的 `user_id` 和 `item_id`
2. 调用 `_build_dense_id_map()` 将原始 ID 排序后映射为 `1..N` 的连续整数（确定性排序保证映射可复现）
3. 调用 `_remap_rows_inplace()` 原地替换所有行中的 `user_id` 和 `item_id` 值
4. 数字 `0` 保留为填充（padding）或未登录词（OOV）槽位，不分配给任何真实用户或物品
5. train/val/test 共用同一映射，确保跨切分编码完全一致

**效果**：`dataset.num_users` 等于重映射后的最大用户 ID，embedding 初始化只需 `num_users + 1` 即可安全覆盖所有 ID，从根本上消除了因原始 ID 稀疏导致的 embedding 越界风险。

## 数据预处理管线

RecBench 提供了一套可组合的离线数据预处理工具链，支持从原始文件到模型就绪特征的全流程处理。所有组件均以 chunk-aware 方式设计，适配超大规模数据场景。

### 离线预处理（Preprocessor）

**源码**：`src/recsys/data/preprocessor.py`

预处理管线负责将原始数据源转换为优化的列式存储格式，并生成统计缓存。

**多后端支持**（按优先级优雅降级）：

| 后端      | 分块读取 | 内存映射 | 类型降级 | 分布式 | 特色能力                      |
| :-------- | :------- | :------- | :------- | :----- | :---------------------------- |
| `pandas`  | 支持     | 支持     | 支持     | 否     | 默认后端，兼容性最佳          |
| `pyarrow` | 支持     | 支持     | 否       | 否     | 列式零拷贝读取，高性能        |
| `polars`  | 支持     | 否       | 支持     | 否     | 惰性扫描、谓词/投影下推       |
| `dask`    | 支持     | 否       | 否       | 是     | 分布式 DataFrame，超内存计算  |
| `vaex`    | 支持     | 支持     | 否       | 否     | 外存内存映射 DataFrame        |
| `modin`   | 支持     | 否       | 支持     | 是     | pandas 兼容的分布式 DataFrame |

**核心能力**：

- **分块低内存读取**：支持通过 pandas chunksize、pyarrow batch、mmap 等方式逐块处理，避免全量加载。
- **列式格式转换**：支持 CSV 转换为 Parquet、Feather、ORC 等列式格式。
- **自动类型降级**：根据数值范围自动将 int64 降级为 int8/16/32、float64 降级为 float32；低基数列自动转为 category 类型以节省内存。
- **多级缓存策略**：支持内存热缓存（hot memory）、磁盘快速缓存（disk）与原始归档（archive）三级缓存，通过 TTL 与容量上限控制。
- **指纹增量处理**：基于文件指纹（路径 + 修改时间 + 内容采样哈希）和配置快照计算缓存键，支持断点续传与增量处理。管线分为五个阶段（ingest → schema_infer → normalize → materialize → stats），每个阶段的完成状态可持久化为 Checkpoint 文件。
- **资源自适应分块**：根据系统可用内存和估计行大小动态计算安全的分块尺寸。

**数据库后端支持**：

- **PostgreSQL Reader**：通过 COPY 协议和 Server-Side Cursor 实现快速批量导出，支持并行分区导出与二进制格式传输。
- **PostgreSQL + pgvector Reader**：在 PostgreSQL Reader 基础上增加向量相似度搜索能力，支持 HNSW/IVFFlat 索引，可直接用于基于向量的硬负样本挖掘。

**输出产物**：预处理完成后生成 `MaterializedDatasetArtifact`，其中包含文件路径、格式、压缩方式、行列数、文件大小、行组数以及关联的元数据和统计文件路径。

### 特征工程（Feature Engineering）

**源码**：`src/recsys/data/feature_engineering.py`

特征工程模块支持在分块数据上进行统计聚合与特征转换，所有中间状态均可序列化缓存。

**ChunkFeatureEngineer**（分块特征工程器）支持的能力：

| 能力         | 实现组件           | 说明                                          |
| :----------- | :----------------- | :-------------------------------------------- |
| 频率编码     | `FrequencyMap`     | 统计类别频率，生成频率特征列                  |
| 目标编码     | `TargetAggregates` | 带平滑参数（默认 10.0）的目标均值编码         |
| 类别字典编码 | `CategoryVocab`    | 按频次降序构建词典，0 号位保留给 UNK          |
| 数值归一化   | `NumericStats`     | 支持 min-max、z-score 和 log1p 三种归一化方法 |
| 哈希特征交叉 | `hash_crossing()`  | 两列组合后通过 MD5 哈希映射到固定桶数         |

所有特征统计信息（频率映射、目标聚合、数值统计、类别词典）汇合在 `FeatureManifest` 中，支持 JSON 序列化与反序列化。工作流程为：先调用 `fit_on_chunks()` 在多个数据块上聚合统计量，再调用 `transform_chunk()` 逐块应用特征变换。

**独立工具函数**：

- `hash_crossing(series_a, series_b, bucket_size)`：对两列做哈希特征交叉
- `embedding_dim_heuristic(n_categories, method)`：根据类别数估算合理的 embedding 维度，支持 Google 四次根规则（`n^(1/4)*2`）、fastai（`n^0.25*1.6`）和 rule_of_thumb（`min(50, sqrt(n))`）三种启发式方法
- `sequence_pad_truncate(sequence, max_len, pad_value, truncate_from)`：序列定长填充/截断工具

**VectorFeatureEngineer**（向量特征工程器）：专为 embedding 向量设计，提供 L2 归一化（用于余弦相似度计算）、向量统计量计算（均值、标准差）、维度规约（PCA 或随机投影）以及查询向量与物品向量之间的批量相似度矩阵计算（支持 cosine、l2 和 inner_product 三种距离度量）。

### 负采样（Negative Sampling）

**源码**：`src/recsys/data/negative_sampling.py`

负采样模块为隐式反馈训练提供多种负样本生成策略，设计上兼容超大规模物品池。

**配置入口 `NegativeSamplingConfig`**：包含策略选择、每正样本负样本数、混合策略权重、物品池来源（自动推断或缓存加载）、最大池内存容量、随机种子、是否排除正样本以及流行度幂次等参数。

**五种采样策略**：

| 策略            | 枚举值       | 采样方式                                  | 适用场景                 |
| :-------------- | :----------- | :---------------------------------------- | :----------------------- |
| 均匀采样        | `uniform`    | 从物品池中均匀随机抽取                    | 通用基线                 |
| 流行度加权      | `popularity` | 按物品出现频次的幂次（可配置）加权采样    | 模拟真实物品分布         |
| In-batch 负采样 | `in_batch`   | 将同一 batch 中其他样本的正样本作为负样本 | 大规模训练，天然大物品池 |
| 硬负样本        | `hard`       | 通过候选池或向量相似度筛选难分负样本      | 提升模型判别力           |
| 混合策略        | `mixed`      | 将多种策略按可配置权重组合后混洗输出      | 平衡多样性与难度         |

**采样器实现**：

- **`NegativeSampler`**（内存采样器）：通过 `fit()` 方法构建物品池（支持直接传入 item 数组及频次，或从 DataFrame 提取），通过 `sample()` 方法批量生成负样本，通过 `sample_per_user()` 方法为每个用户独立采样并自动排除其正样本。
- **`PostgresNegativeSampler`**（数据库采样器）：直接通过 PostgreSQL 的 TABLESAMPLE 语法在数据库端进行高效随机采样，无需将全量物品池加载到内存。支持 BERNOULLI（真随机）和 SYSTEM（块级快速）两种采样方法，以及基于流行度列的加权采样。

**统计缓存 `ItemPoolStats`**：将物品 ID 数组、频次分布和采样概率缓存在本地 NPZ 文件，支持跨运行复用。

**工厂函数 `create_sampler()`**：根据策略名称字符串快速创建配置好的采样器实例，未知策略自动回退到 uniform。

## 数据集注册机制

**源码**：`src/recsys/data/dataset_registry.py`、`src/recsys/core/registry.py`

所有数据集通过 `DATASET_REGISTRY` 注册，采用副作用注册模式 —— 在 `dataset_registry.py` 中显式导入各数据集模块（如 `import recsys.data.datasets.movielens_1m`），触发模块内 `@DATASET_REGISTRY.register(...)` 装饰器的执行，将数据集键名及其元信息写入全局注册表。

每条注册记录包含以下元信息：
- 数据集家族（`family`）：如 `"classical"`（经典数据集）或 `"deep_ctr"`（深度 CTR 预估数据集）
- 模态（`modality`）：如 `"sequential"`、`"tabular"`、`"time-series"`
- 任务类型（`tasks`）：如 `"ranking"`、`"ctr"`、`"cvr"`
- 多模态能力标记：`supports_multi_subset`、`supports_candidates`、`supports_vector_embeddings`
- 默认 EDA 子集（`default_eda_subset`）

`get_dataset_capabilities(name)` 函数可查询任意已注册数据集的能力元数据，未注册数据集返回全否默认值。

此外，`dataset_registry.py` 维护了后端能力矩阵（各执行后端的特性对比）、压缩编解码能力矩阵、负采样策略注册表、特征工程原语注册表以及数据库后端能力矩阵（PostgreSQL + pgvector），所有注册表均优雅处理可选依赖缺失的情况。

## 数据层能力边界

### 当前已落地的职责

数据集适配器层（Dataset Adapter）负责：
- 原始数据的下载与本地缓存（HTTP 下载 / HuggingFace datasets API / 本地生成）
- Train / Validation / Test 切分的构造与管理
- 样本格式定义（字典形式的 batch，键值包含 `user_id`、`item_id`、`labels`、`candidate_items`、`user_feats`、`item_feats`、`domain_seqs` 等）
- Dense ID Remap 与变长特征 Padding
- 元信息暴露（`num_users`、`num_items`、`feature_cols`、`label_col` 等）

离线预处理管线层负责：
- 分块感知的数据读取与列式格式转换
- 基于分块聚合的特征工程与状态缓存
- 多策略负采样与物品池管理
- 数据库端的批量导出与向量检索
- EDA 自动报告生成

### 已确认的限制

- 多模态语义嵌入（如 TAAC 2025 的 `mm_emb_*` 子集）在当前训练 batch 契约中尚未统一为端到端的可训练特征通道，目前主要用于 EDA 分析与评估阶段。
- EDA 报告以 ECharts JSON 格式输出，查看交互式图表需在 MkDocs 或兼容的浏览器环境中打开。

## 未来展望

以下能力已在设计规划或源码占位中预留，但尚未完整落地：

- **Amazon、Criteo、Taobao 等公开基准数据集适配器**：在数据子包 `__init__.py` 的模块文档中已列出，但当前仅完成了 Synthetic、MovieLens、TAAC 2025 和 TAAC 2026 四个适配器。
- **完整的训练流程示例文档**：当前各数据集的配置文件和训练入口脚本（`scripts/run_single.py`）已可正常驱动训练，但缺乏一份端到端的教程级流程文档。
- **基于 Registry 的动态数据集加载示例**：Registry 机制已就绪，但缺少通过配置键名动态实例化数据集并驱动训练的标准用法文档。
- **多模态 batch 契约统一**：TAAC 2025 的多模态嵌入数据已可通过 `load_subset()` 独立加载，且 Schema Manifest 已缓存维度信息，但将这些嵌入直接作为模型输入的标准化 batch 格式尚未定义。

## 参考

- [BaseDataset 源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/core/base_dataset.py)
- [SequenceSplit 源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/data/split_utils.py)
- [Dataset Registry 源码](https://github.com/nicedemo2014/recbench/blob/main/src/recsys/data/dataset_registry.py)
- [TAAC 2025 论文](https://arxiv.org/abs/2604.04976)
- [MovieLens 数据集](https://grouplens.org/datasets/movielens/1m/)
