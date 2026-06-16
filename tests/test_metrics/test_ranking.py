"""Ranking metrics 单元测试：验证 NDCG、MRR、HitRate 等指标计算。"""

import numpy as np
import pytest

from recsys.evaluation.ranking import (
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
    normalize_ranking_metric_name,
)


class TestNormalizeRankingMetricName:
    """指标名称规范化测试。"""

    def test_ndcg_alias(self):
        """ndcg 别名映射到 ndcg_at_k。"""
        assert normalize_ranking_metric_name("ndcg") == "ndcg_at_k"
        assert normalize_ranking_metric_name("NDCG") == "ndcg_at_k"

    def test_mrr_alias(self):
        """MRR 保持不变。"""
        assert normalize_ranking_metric_name("mrr") == "mrr"
        assert normalize_ranking_metric_name("MRR") == "mrr"

    def test_hit_rate_alias(self):
        """hit_rate 别名映射。"""
        assert normalize_ranking_metric_name("hit") == "hit_rate_at_k"
        assert normalize_ranking_metric_name("hitrate") == "hit_rate_at_k"
        assert normalize_ranking_metric_name("HR") == "hit_rate_at_k"

    def test_unknown_name_passthrough(self):
        """未知名称原样返回。"""
        assert normalize_ranking_metric_name("custom_metric") == "custom_metric"


class TestComputeNdcgAtK:
    """NDCG@K 指标计算测试。"""

    def test_perfect_ranking(self):
        """完美排序：正样本排在最前。"""
        y_true = {10, 20}  # 相关项
        y_score = np.array([0.9, 0.8, 0.5, 0.3, 0.1])
        candidate_ids = [10, 20, 30, 40, 50]

        ndcg = compute_ndcg_at_k(y_true, y_score, candidate_ids, k=5)
        assert ndcg == pytest.approx(1.0, rel=1e-3)

    def test_worst_ranking(self):
        """最差排序：正样本排在最后。"""
        y_true = {10, 20}
        y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        candidate_ids = [30, 40, 50, 10, 20]

        ndcg = compute_ndcg_at_k(y_true, y_score, candidate_ids, k=5)
        # 正样本在最后两位，但仍在 top-5 内，NDCG 应该较低但大于 0
        assert 0 < ndcg <= 1

    def test_single_relevant(self):
        """单个相关项。"""
        y_true = {10}
        y_score = np.array([0.9, 0.5, 0.3])
        candidate_ids = [10, 20, 30]

        ndcg = compute_ndcg_at_k(y_true, y_score, candidate_ids, k=3)
        assert ndcg == pytest.approx(1.0, rel=1e-3)


class TestComputeMrr:
    """MRR 指标计算测试。"""

    def test_first_position(self):
        """正样本在第一位。"""
        y_true = {10}
        y_score = np.array([0.9, 0.5, 0.3])
        candidate_ids = [10, 20, 30]

        mrr = compute_mrr(y_true, y_score, candidate_ids)
        assert mrr == pytest.approx(1.0, rel=1e-3)

    def test_second_position(self):
        """正样本在第二位。"""
        y_true = {20}
        y_score = np.array([0.9, 0.5, 0.3])
        candidate_ids = [10, 20, 30]

        mrr = compute_mrr(y_true, y_score, candidate_ids)
        assert mrr == pytest.approx(0.5, rel=1e-3)

    def test_no_relevant(self):
        """无相关项。"""
        y_true = set()
        y_score = np.array([0.9, 0.5, 0.3])
        candidate_ids = [10, 20, 30]

        mrr = compute_mrr(y_true, y_score, candidate_ids)
        assert mrr == 0.0


class TestComputeHitRateAtK:
    """HitRate@K 指标计算测试。"""

    def test_hit_in_top_k(self):
        """正样本在 top-k 内。"""
        y_true = {10}
        y_score = np.array([0.9, 0.5, 0.3])
        candidate_ids = [10, 20, 30]

        hr = compute_hit_rate_at_k(y_true, y_score, candidate_ids, k=2)
        assert hr == 1.0

    def test_hit_not_in_top_k(self):
        """正样本不在 top-k 内。"""
        y_true = {30}
        y_score = np.array([0.9, 0.5, 0.3])
        candidate_ids = [10, 20, 30]

        hr = compute_hit_rate_at_k(y_true, y_score, candidate_ids, k=2)
        assert hr == 0.0

    def test_multiple_relevant(self):
        """多个相关项：只要有一个在 top-k 内就返回 1。"""
        y_true = {10, 20}
        y_score = np.array([0.9, 0.5, 0.3])
        candidate_ids = [10, 20, 30]

        hr = compute_hit_rate_at_k(y_true, y_score, candidate_ids, k=2)
        # 两个相关项，一个在 top-2，HitRate = 1
        assert hr == 1.0


class TestComputeRecallAtK:
    """Recall@K 指标计算测试。"""

    def test_all_relevant_in_top_k(self):
        """所有相关项都在 top-k 内。"""
        y_true = {10, 20}
        y_score = np.array([0.9, 0.8, 0.3])
        candidate_ids = [10, 20, 30]

        recall = compute_recall_at_k(y_true, y_score, candidate_ids, k=2)
        assert recall == 1.0

    def test_partial_recall(self):
        """部分相关项在 top-k 内。"""
        y_true = {10, 20, 30}
        y_score = np.array([0.9, 0.8, 0.3])
        candidate_ids = [10, 20, 30]

        recall = compute_recall_at_k(y_true, y_score, candidate_ids, k=2)
        assert recall == pytest.approx(2/3, rel=1e-3)

    def test_no_relevant_in_top_k(self):
        """无相关项在 top-k 内。"""
        y_true = {30}
        y_score = np.array([0.9, 0.8, 0.3])
        candidate_ids = [10, 20, 30]

        recall = compute_recall_at_k(y_true, y_score, candidate_ids, k=2)
        assert recall == 0.0
