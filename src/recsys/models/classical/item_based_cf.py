"""Item-Based Collaborative Filtering (Sarwar et al., 2001).

Sarwar et al. "Item-based collaborative filtering recommendation algorithms"
- 构建物品-物品相似度矩阵（Cosine / IUF）
- Top-K 邻居截断 + 可选归一化
- 预测时基于用户历史物品的相似物品加权打分

模型能力：
- fit：从训练交互数据构建相似度矩阵（非梯度训练）
- predict：对目标用户集生成 Top-K 推荐，产出标准 PredictionBundle

归入 classical 家族，task_type = "ranking"，supports_training = False。

内存优化：
- 使用 scipy.sparse.csr_matrix 存储相似度矩阵
- Top-K 截断减少稀疏矩阵非零元素
- 支持百万级物品规模
"""

from __future__ import annotations

import math
from collections import defaultdict
from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse

from recsys.core.base_model import BaseRecommender, Capability
from recsys.core.prediction_bundle import PredictionBundle
from recsys.core.registry import MODEL_REGISTRY

# ---------------------------------------------------------------------------
# 相似度策略注册表
# ---------------------------------------------------------------------------

def _cosine_weight(user_item_count: int) -> float:
    """Cosine 相似度的共现权重：每次共现贡献 1.0。"""
    return 1.0


def _iuf_weight(user_item_count: int) -> float:
    """IUF (Inverse User Frequency) 权重。

    活跃用户（交互物品多）对共现的贡献被压低，
    公式：1 / log(1 + user_item_count)
    """
    return 1.0 / math.log1p(user_item_count)


# 策略名 → 权重函数
_SIMILARITY_WEIGHTS: Dict[str, Callable[[int], float]] = {
    "cosine": _cosine_weight,
    "iuf": _iuf_weight,
}


# ---------------------------------------------------------------------------
# ItemCF 主类
# ---------------------------------------------------------------------------

@MODEL_REGISTRY.register(
    "itemcf",
    family="classical",
    year=2001,
    task_type="ranking",
    supports_training=False,
    required_features=["user_id", "item_id"],
    default_metrics=["ndcg@10", "hit_rate@10", "recall@10", "mrr"],
)
class ItemBasedCF(BaseRecommender):
    """基于物品的协同过滤推荐器。

    通过用户历史交互构建物品共现矩阵，
    再归一化为相似度矩阵后进行 Top-K 推荐。

    内存优化策略：
    - use_sparse_matmul=False（默认）：直接构建 Top-K 截断的相似度矩阵，
      内存占用 O(n_items × k)，适合小内存机器（<16GB）
    - use_sparse_matmul=True：使用稀疏矩阵乘法 U^T @ U，
      内存占用约 17-20GB，计算速度快，适合大内存机器（≥32GB）

    Parameters
    ----------
    similarity : str
        相似度计算策略：``"cosine"`` 或 ``"iuf"``。
    top_k_neighbors : int
        相似度矩阵中每个物品保留的最近邻数量。
    recommend_k : int
        推荐列表默认长度。
    normalize : bool
        是否对相似度矩阵做最大-最小归一化。
    use_sparse_matmul : bool
        是否使用稀疏矩阵乘法方案（需要大内存，速度快）。
        默认 False，使用内存友好的直接构建方案。
    """

    # 类级别元信息
    model_name = "itemcf"
    model_family = "classical"
    task_type = "ranking"
    problem_type = "implicit_ranking"
    supports_training = False
    required_features = ["user_id", "item_id"]
    default_metrics = ["ndcg@10", "hit_rate@10", "recall@10", "mrr"]

    def __init__(
        self,
        similarity: str = "cosine",
        top_k_neighbors: int = 50,
        recommend_k: int = 10,
        normalize: bool = True,
        use_sparse_matmul: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _validate_similarity(similarity)
        self._similarity = similarity
        self._top_k_neighbors = top_k_neighbors
        self._recommend_k = recommend_k
        self._normalize = normalize
        self._use_sparse_matmul = use_sparse_matmul

        # ---- 内部状态 ----
        # 稀疏相似度矩阵 (n_items x n_items)
        self._sim_matrix: Optional[sparse.csr_matrix] = None
        # 物品 ID → 矩阵索引映射
        self._item_to_idx: Dict[int, int] = {}
        self._idx_to_item: Dict[int, int] = {}
        # 物品被多少人交互过: {item_id: count}
        self._item_popularity: Dict[int, int] = {}

        # 添加推荐能力
        self._capabilities.add(Capability.RECOMMENDER)
        self._capabilities.add(Capability.RANKER)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def _build_similarity_sparse_matmul(
        self, user_items: Dict[int, Any]
    ) -> sparse.csr_matrix:
        """使用稀疏矩阵乘法构建相似度矩阵（需要大内存）。

        策略：U^T @ U 计算共现矩阵，然后归一化。
        内存需求：约 17-20 GB（适合大内存机器）。
        优点：计算速度快，适合 GPU 加速。
        """
        weight_fn = _SIMILARITY_WEIGHTS[self._similarity]
        n_users = len(user_items)
        n_items = len(self._item_to_idx)

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for user_idx, (_user_id, items) in enumerate(user_items.items()):
            if hasattr(items, "__len__"):
                n_user_items = len(items)
            else:
                items = list(items)
                n_user_items = len(items)

            if n_user_items == 0:
                continue

            weight = weight_fn(n_user_items)
            sqrt_weight = math.sqrt(weight)

            for item in items:
                if item in self._item_to_idx:
                    item_idx = self._item_to_idx[item]
                    rows.append(user_idx)
                    cols.append(item_idx)
                    data.append(sqrt_weight)

        user_item_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        # 共现矩阵 = U^T @ U
        cooccurrence = user_item_matrix.T @ user_item_matrix

        # 计算相似度
        item_counts = np.array(
            [self._item_popularity.get(self._idx_to_item[i], 1)
             for i in range(n_items)],
            dtype=np.float32,
        )
        inv_sqrt_counts = np.where(item_counts > 0, 1.0 / np.sqrt(item_counts), 0)
        diag_mat = sparse.diags(inv_sqrt_counts, format="csr", dtype=np.float32)
        sim_matrix = diag_mat @ cooccurrence @ diag_mat

        # 移除对角线
        sim_matrix.setdiag(0.0)
        sim_matrix.eliminate_zeros()

        # Top-K 截断
        sim_matrix = self._truncate_sparse_matrix(sim_matrix)

        # 归一化
        if self._normalize:
            sim_matrix = self._normalize_sparse_matrix(sim_matrix)

        return sim_matrix

    def _build_topk_similarity_direct(
        self, user_items: Dict[int, Any]
    ) -> sparse.csr_matrix:
        """直接构建 Top-K 截断的相似度矩阵，跳过完整共现矩阵。

        策略：
        1. 逐用户处理，累积每个物品对的共现计数
        2. 使用 defaultdict 存储共现，避免预分配大数组
        3. 计算相似度后立即截断 Top-K
        4. 只保留 Top-K 结果，内存占用 O(n_items × k)
        """
        import gc
        from collections import defaultdict

        n_items = len(self._item_to_idx)
        k = self._top_k_neighbors

        # 1. 构建共现矩阵（使用 defaultdict，按需分配）
        # cooc[i][j] = 物品 i 和 j 的共现次数
        cooc: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        weight_fn = _SIMILARITY_WEIGHTS[self._similarity]

        for _user_id, items in user_items.items():
            if hasattr(items, "__len__"):
                n_user_items = len(items)
            else:
                items = list(items)
                n_user_items = len(items)

            if n_user_items < 2:
                continue

            weight = weight_fn(n_user_items)

            # 转换为索引列表
            item_indices = []
            for item in items:
                if item in self._item_to_idx:
                    item_indices.append(self._item_to_idx[item])

            # 更新共现矩阵
            for i in range(len(item_indices)):
                ii = item_indices[i]
                for j in range(i + 1, len(item_indices)):
                    jj = item_indices[j]
                    cooc[ii][jj] += weight
                    cooc[jj][ii] += weight

        # 2. 计算相似度并截断 Top-K
        # 使用 LIL 格式构建稀疏矩阵
        sim_matrix = sparse.lil_matrix((n_items, n_items), dtype=np.float32)

        for i, neighbors in cooc.items():
            if not neighbors:
                continue

            ni = self._item_popularity.get(self._idx_to_item[i], 1)

            # 计算相似度
            sim_scores: List[Tuple[int, float]] = []
            for j, cij in neighbors.items():
                nj = self._item_popularity.get(self._idx_to_item[j], 1)
                denom = math.sqrt(ni * nj)
                if denom > 0:
                    sim = cij / denom
                    sim_scores.append((j, sim))

            # Top-K 截断
            if len(sim_scores) > k:
                sim_scores.sort(key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[:k]

            # 归一化
            if self._normalize and sim_scores:
                max_sim = max(s for _, s in sim_scores)
                sim_scores = [(j, s / max_sim) for j, s in sim_scores]

            # 写入稀疏矩阵
            for j, sim in sim_scores:
                sim_matrix[i, j] = sim

        # 释放共现矩阵内存
        del cooc
        gc.collect()

        return sim_matrix.tocsr()

    def fit(
        self,
        user_item_pairs: Optional[List[Tuple[int, int]]] = None,
        user_items_dict: Optional[Dict[int, Set[int]]] = None,
    ) -> "ItemBasedCF":
        """从训练交互数据构建物品相似度矩阵。

        Parameters
        ----------
        user_item_pairs : list of (user_id, item_id), optional
            训练 split 中的正交互对。
        user_items_dict : dict of {user_id: set of item_ids}, optional
            预分组的用户-物品映射。如果提供，则跳过分组步骤。

        Returns
        -------
        self
        """
        if user_items_dict is not None:
            user_items = user_items_dict
        elif user_item_pairs is not None:
            if not user_item_pairs:
                raise ValueError("user_item_pairs 不能为空")
            user_items = _group_user_items(user_item_pairs)
        else:
            raise ValueError("必须提供 user_item_pairs 或 user_items_dict 之一")

        # 1. 构建物品索引映射
        self._build_item_index(user_items)

        # 2. 构建相似度矩阵（根据内存情况选择策略）
        if self._use_sparse_matmul:
            # 稀疏矩阵乘法方案（需要 ~17GB 内存，速度快）
            self._sim_matrix = self._build_similarity_sparse_matmul(user_items)
        else:
            # 直接构建 Top-K 方案（内存友好，适合小内存机器）
            self._sim_matrix = self._build_topk_similarity_direct(user_items)

        self._fitted = True
        return self

    def predict(
        self,
        user_train_items: Optional[Dict[int, Set[int]]] = None,
        user_test_items: Optional[Dict[int, Set[int]]] = None,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> PredictionBundle:
        """为用户集生成 Top-K 推荐列表。"""
        if user_train_items is None:
            user_train_items = {}
        if user_test_items is None:
            user_test_items = {}

        if not self._fitted:
            raise RuntimeError(
                "ItemBasedCF 尚未完成 fit()，"
                "请先调用 fit(user_item_pairs) 构建相似度矩阵"
            )

        if k is None:
            k = self._recommend_k

        group_ids: List[int] = []
        candidate_ids: List[List[int]] = []
        y_score: List[List[float]] = []
        y_true: List[List[int]] = []

        for user_id, train_items in user_train_items.items():
            if not train_items:
                continue

            # 为该用户生成推荐
            recs = self._score_for_user(train_items)
            # 排序取 Top-K
            top_items, top_scores = _top_k_from_scores(recs, k)

            group_ids.append(user_id)
            candidate_ids.append(top_items)
            y_score.append(top_scores)
            # 该用户在测试集中的真实交互物品
            y_true.append(list(user_test_items.get(user_id, set())))

        return self.export_prediction_bundle(
            y_true=y_true,
            y_score=y_score,
            group_ids=group_ids,
            candidate_ids=candidate_ids,
            score_type="raw_score",
            k_list=[k],
            metadata={
                "similarity": self._similarity,
                "top_k_neighbors": self._top_k_neighbors,
                "num_users": len(group_ids),
            },
        )

    @property
    def num_items(self) -> int:
        """已知物品数量。"""
        return len(self._item_to_idx)

    # ------------------------------------------------------------------
    # 内部算法 - 稀疏矩阵实现
    # ------------------------------------------------------------------

    def _build_item_index(self, user_items: Dict[int, Any]) -> None:
        """构建物品 ID → 矩阵索引的双向映射。"""
        all_items: Set[int] = set()
        for items in user_items.values():
            if hasattr(items, "__iter__"):
                all_items.update(items)
        # 排序确保确定性
        sorted_items = sorted(all_items)
        self._item_to_idx = {item: idx for idx, item in enumerate(sorted_items)}
        self._idx_to_item = {idx: item for item, idx in self._item_to_idx.items()}

        # 计算物品流行度
        self._item_popularity = defaultdict(int)
        for items in user_items.values():
            for item in items:
                self._item_popularity[item] += 1

    def _compute_similarity_sparse(
        self, cooccurrence: sparse.csr_matrix
    ) -> sparse.csr_matrix:
        """将共现矩阵归一化为余弦相似度矩阵。

        sim(i, j) = cooccurrence(i, j) / sqrt(N[i] * N[j])

        使用稀疏矩阵操作避免内存爆炸。
        """
        # 物品流行度向量
        item_counts = np.array(
            [self._item_popularity.get(self._idx_to_item[i], 1)
             for i in range(cooccurrence.shape[0])],
            dtype=np.float32,
        )

        # 对角矩阵 diag = 1 / sqrt(N[i])
        # 避免除零
        inv_sqrt_counts = np.where(item_counts > 0, 1.0 / np.sqrt(item_counts), 0)
        diag_mat = sparse.diags(inv_sqrt_counts, format="csr", dtype=np.float32)

        # 相似度矩阵 = diag_mat @ cooccurrence @ diag_mat
        sim_matrix = diag_mat @ cooccurrence @ diag_mat

        # 移除对角线（物品与自身的相似度）
        sim_matrix.setdiag(0.0)
        sim_matrix.eliminate_zeros()

        return sim_matrix.tocsr()

    def _truncate_sparse_matrix(
        self, sim_matrix: sparse.csr_matrix
    ) -> sparse.csr_matrix:
        """对稀疏相似度矩阵做 Top-K 截断。

        每行只保留最大的 top_k_neighbors 个元素。
        """
        k = self._top_k_neighbors
        if k <= 0:
            return sim_matrix

        n_rows = sim_matrix.shape[0]
        # 使用 lil 格式便于逐行修改
        truncated = sparse.lil_matrix(sim_matrix.shape, dtype=np.float32)

        for i in range(n_rows):
            row = sim_matrix.getrow(i)
            if row.nnz == 0:
                continue

            # 获取非零元素的列索引和值
            cols = row.indices
            vals = row.data

            # 取 Top-K
            if len(vals) > k:
                top_k_idx = np.argpartition(vals, -k)[-k:]
                cols = cols[top_k_idx]
                vals = vals[top_k_idx]

            truncated[i, cols] = vals

        return truncated.tocsr()

    def _normalize_sparse_matrix(
        self, sim_matrix: sparse.csr_matrix
    ) -> sparse.csr_matrix:
        """对稀疏相似度矩阵做最大值归一化。"""
        # 每行的最大值
        max_vals = np.array(sim_matrix.max(axis=1).todense()).flatten()
        # 避免除零
        max_vals = np.where(max_vals > 0, max_vals, 1.0)

        # diag_mat = 1 / max_val
        diag_mat = sparse.diags(1.0 / max_vals, format="csr", dtype=np.float32)

        return diag_mat @ sim_matrix

    def _score_for_user(self, user_items: Set[int]) -> Dict[int, float]:
        """对单个用户的所有候选物品打分。

        使用稀疏矩阵乘法加速计算。
        """
        if self._sim_matrix is None:
            return {}

        # 构建用户历史物品向量 (1 x n_items)
        n_items = len(self._item_to_idx)
        user_vec = np.zeros(n_items, dtype=np.float32)

        for item in user_items:
            if item in self._item_to_idx:
                user_vec[self._item_to_idx[item]] = 1.0

        # 分数向量 = user_vec @ sim_matrix (1 x n_items)
        scores_vec = user_vec @ self._sim_matrix

        # 转换为 {item_id: score} 字典，过滤已交互物品
        scores: Dict[int, float] = {}
        for idx in np.nonzero(scores_vec)[0]:
            item_id = self._idx_to_item[idx]
            if item_id not in user_items:
                scores[item_id] = float(scores_vec[idx])

        return scores


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _validate_similarity(similarity: str) -> None:
    """校验相似度策略名称。"""
    if similarity not in _SIMILARITY_WEIGHTS:
        valid = ", ".join(sorted(_SIMILARITY_WEIGHTS.keys()))
        raise ValueError(
            f"不支持的相似度策略 '{similarity}'，可选: {valid}"
        )


def _group_user_items(
    pairs: List[Tuple[int, int]],
) -> Dict[int, Set[int]]:
    """将 (user_id, item_id) 列表组织为 {user_id: {item_ids}}。"""
    grouped: Dict[int, Set[int]] = defaultdict(set)
    for user_id, item_id in pairs:
        grouped[user_id].add(item_id)
    return dict(grouped)


def _top_k_from_scores(
    scores: Dict[int, float], k: int
) -> Tuple[List[int], List[float]]:
    """从分数字典中提取 Top-K 物品及其分数。"""
    sorted_items = sorted(scores.items(), key=itemgetter(1), reverse=True)[:k]
    items = [i for i, _ in sorted_items]
    vals = [s for _, s in sorted_items]
    return items, vals
