"""Item-Based Collaborative Filtering (Sarwar et al., 2001).

Sarwar et al. "Item-based collaborative filtering recommendation algorithms"
- 构建物品-物品相似度矩阵（Cosine / IUF）
- Top-K 邻居截断 + 可选归一化
- 预测时基于用户历史物品的相似物品加权打分

模型能力：
- fit：从训练交互数据构建相似度矩阵（非梯度训练）
- predict：对目标用户集生成 Top-K 推荐，产出标准 PredictionBundle

归入 classical 家族，task_type = "ranking"，supports_training = False。
"""

from __future__ import annotations

import math
from collections import defaultdict
from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _validate_similarity(similarity)
        self._similarity = similarity
        self._top_k_neighbors = top_k_neighbors
        self._recommend_k = recommend_k
        self._normalize = normalize

        # ---- 内部状态 ----
        # 物品 → 最近邻映射: {item_id: {neighbor_id: similarity_score}}
        self._item_neighbors: Dict[int, Dict[int, float]] = {}
        # 物品被多少人交互过: {item_id: count}
        self._item_popularity: Dict[int, int] = {}

        # 添加推荐能力
        self._capabilities.add(Capability.RECOMMENDER)
        self._capabilities.add(Capability.RANKER)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def fit(self, user_item_pairs: List[Tuple[int, int]]) -> "ItemBasedCF":
        """从训练交互数据构建物品相似度矩阵。

        Parameters
        ----------
        user_item_pairs : list of (user_id, item_id)
            训练 split 中的正交互对。
            应由 dataset adapter 的 ``get_split("train")`` 提取后传入。

        Returns
        -------
        self
        """
        if not user_item_pairs:
            raise ValueError("user_item_pairs 不能为空")

        # 1. 组织 user → items 映射
        user_items = _group_user_items(user_item_pairs)

        # 2. 构建共现矩阵
        cooccurrence, item_counts = self._build_cooccurrence_matrix(user_items)

        # 3. 计算相似度矩阵
        sim_matrix = self._compute_similarity_matrix(cooccurrence, item_counts)

        # 4. Top-K 截断 + 可选归一化
        self._item_neighbors = self._truncate_and_normalize(sim_matrix)
        self._item_popularity = item_counts
        self._fitted = True
        return self

    def predict(
        self,
        user_train_items: Optional[Dict[int, Set[int]]] = None,
        user_test_items: Optional[Dict[int, Set[int]]] = None,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> PredictionBundle:
        """为用户集生成 Top-K 推荐列表。

        Parameters
        ----------
        user_train_items : dict, optional
            {user_id: set of item_ids} — 用户在训练集中的历史交互物品。
            用于触发推荐。
        user_test_items : dict, optional
            {user_id: set of item_ids} — 用户在测试集中的真实交互物品。
            用作 ground truth。
        k : int, optional
            推荐列表长度，默认使用构造时传入的 ``recommend_k``。
        **kwargs : Any
            其他参数（兼容基类签名）。

        Returns
        -------
        PredictionBundle
            包含 task_type="ranking"、problem_type="implicit_ranking"、
            group_ids、y_score、candidate_ids、y_true 的结构化预测产物。
        """
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
        return len(self._item_popularity)

    # ------------------------------------------------------------------
    # 内部算法
    # ------------------------------------------------------------------

    def _build_cooccurrence_matrix(
        self, user_items: Dict[int, Set[int]]
    ) -> Tuple[Dict[int, Dict[int, float]], Dict[int, int]]:
        """构建物品共现矩阵。

        遍历每个用户的物品集合，对任意物品对 (i, j) 增加共现计数。
        计数权重由相似度策略决定（cosine: 1.0 / iuf: 1/log(1+|I_u|)）。

        Returns
        -------
        cooccurrence : dict
            {item_i: {item_j: weighted_count}}
        item_counts : dict
            {item_id: 被多少用户交互过}
        """
        weight_fn = _SIMILARITY_WEIGHTS[self._similarity]
        cooccurrence: Dict[int, Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        item_counts: Dict[int, int] = defaultdict(int)

        for items in user_items.values():
            weight = weight_fn(len(items))
            for i in items:
                item_counts[i] += 1
                for j in items:
                    if i == j:
                        continue
                    cooccurrence[i][j] += weight

        return dict(cooccurrence), dict(item_counts)

    def _compute_similarity_matrix(
        self,
        cooccurrence: Dict[int, Dict[int, float]],
        item_counts: Dict[int, int],
    ) -> Dict[int, Dict[int, float]]:
        """将共现矩阵归一化为余弦相似度矩阵。

        sim(i, j) = cooccurrence(i, j) / sqrt(N[i] * N[j])
        """
        sim_matrix: Dict[int, Dict[int, float]] = {}

        for i, neighbors in cooccurrence.items():
            sim_matrix[i] = {}
            ni = item_counts[i]
            for j, cij in neighbors.items():
                nj = item_counts[j]
                denom = math.sqrt(ni * nj)
                if denom > 0:
                    sim_matrix[i][j] = cij / denom

        return sim_matrix

    def _truncate_and_normalize(
        self, sim_matrix: Dict[int, Dict[int, float]]
    ) -> Dict[int, Dict[int, float]]:
        """对相似度矩阵做 Top-K 截断 + 可选归一化。

        每个物品只保留相似度最高的 ``top_k_neighbors`` 个邻居。
        若 ``normalize=True``，将每个物品的相似度向量除以其最大值。
        """
        result: Dict[int, Dict[int, float]] = {}

        for i, neighbors in sim_matrix.items():
            # 按相似度降序排列，取 Top-K
            sorted_neighbors = sorted(
                neighbors.items(), key=itemgetter(1), reverse=True
            )[: self._top_k_neighbors]
            truncated = dict(sorted_neighbors)

            if self._normalize and truncated:
                max_sim = max(truncated.values())
                truncated = {j: s / max_sim for j, s in truncated.items()}

            result[i] = truncated

        return result

    def _score_for_user(self, user_items: Set[int]) -> Dict[int, float]:
        """对单个用户的所有候选物品打分。

        遍历用户历史物品，聚合它们最近邻的相似度得分。
        过滤用户已交互的物品。

        Returns
        -------
        dict
            {candidate_item_id: aggregated_score}
        """
        scores: Dict[int, float] = defaultdict(float)

        for item in user_items:
            neighbors = self._item_neighbors.get(item, {})
            for neighbor_id, sim in neighbors.items():
                if neighbor_id not in user_items:
                    scores[neighbor_id] += sim

        return dict(scores)


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
    """将 (user_id, item_id) 列表组织为 {user_id: {item_ids}}。

    由 dataset adapter 的 train split 提供原始数据，
    ItemCF 内部完成分组——这是算法必需的中间表示，
    而非数据加载职责。
    """
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
