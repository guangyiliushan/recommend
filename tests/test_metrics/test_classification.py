"""Classification metrics 单元测试：验证 Accuracy、AUC、F1 等指标计算。"""

import numpy as np
import pytest

from recsys.evaluation.metrics import (
    compute_accuracy,
    compute_f1,
    compute_precision,
    compute_recall,
    compute_roc_auc,
    normalize_metric_name,
)


class TestNormalizeMetricName:
    """指标名称规范化测试。"""

    def test_accuracy_alias(self):
        """accuracy 别名映射。"""
        assert normalize_metric_name("acc") == "accuracy"
        assert normalize_metric_name("ACC") == "accuracy"

    def test_precision_alias(self):
        """precision 别名映射。"""
        assert normalize_metric_name("ppv") == "precision"
        assert normalize_metric_name("PPV") == "precision"

    def test_recall_alias(self):
        """recall 别名映射。"""
        assert normalize_metric_name("sensitivity") == "recall"
        assert normalize_metric_name("TPR") == "recall"

    def test_f1_alias(self):
        """f1 别名映射。"""
        assert normalize_metric_name("F1") == "f1"
        assert normalize_metric_name("F1-score") == "f1"

    def test_roc_auc_alias(self):
        """roc_auc 别名映射。"""
        assert normalize_metric_name("ROC-AUC") == "roc_auc"
        assert normalize_metric_name("auroc") == "roc_auc"

    def test_unknown_name_passthrough(self):
        """未知名称原样返回。"""
        assert normalize_metric_name("custom_metric") == "custom_metric"


class TestComputeAccuracy:
    """Accuracy 指标计算测试。"""

    def test_perfect_prediction(self):
        """完美预测：所有预测正确。"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])

        acc = compute_accuracy(y_true, y_pred)
        assert acc == pytest.approx(1.0, rel=1e-3)

    def test_random_prediction(self):
        """随机预测：准确率应在合理范围内。"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_pred = np.random.randint(0, 2, size=100)

        acc = compute_accuracy(y_true, y_pred)
        # 随机预测的准确率应在 0.3-0.7 之间
        assert 0.3 <= acc <= 0.7

    def test_all_negative(self):
        """全负样本：验证边界情况。"""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        acc = compute_accuracy(y_true, y_pred)
        assert acc == pytest.approx(1.0, rel=1e-3)


class TestComputePrecision:
    """Precision 指标计算测试。"""

    def test_perfect_precision(self):
        """完美精确率：所有预测为正的都是真正。"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])

        prec = compute_precision(y_true, y_pred)
        assert prec == pytest.approx(1.0, rel=1e-3)

    def test_zero_precision(self):
        """零精确率：所有预测为正的都是假正。"""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0])

        prec = compute_precision(y_true, y_pred)
        assert prec == pytest.approx(0.0, rel=1e-3)


class TestComputeRecall:
    """Recall 指标计算测试。"""

    def test_perfect_recall(self):
        """完美召回率：所有真正都被预测为正。"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])

        rec = compute_recall(y_true, y_pred)
        assert rec == pytest.approx(1.0, rel=1e-3)

    def test_partial_recall(self):
        """部分召回率。"""
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 0])

        rec = compute_recall(y_true, y_pred)
        assert rec == pytest.approx(1/3, rel=1e-3)


class TestComputeF1:
    """F1 指标计算测试。"""

    def test_perfect_f1(self):
        """完美 F1。"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])

        f1 = compute_f1(y_true, y_pred)
        assert f1 == pytest.approx(1.0, rel=1e-3)

    def test_balanced_f1(self):
        """平衡 F1：精确率和召回率相等。"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 0, 0, 1, 0, 0])

        f1 = compute_f1(y_true, y_pred)
        # precision = 1/2, recall = 1/3
        # f1 = 2 * (1/2 * 1/3) / (1/2 + 1/3) = 2/5
        assert f1 == pytest.approx(0.4, rel=1e-3)


class TestComputeRocAuc:
    """ROC-AUC 指标计算测试。"""

    def test_perfect_auc(self):
        """完美 AUC：正样本分数都高于负样本。"""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.95])

        auc = compute_roc_auc(y_true, y_prob)
        assert auc == pytest.approx(1.0, rel=1e-3)

    def test_random_auc(self):
        """随机 AUC：应在 0.5 附近。"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_prob = np.random.rand(100)

        auc = compute_roc_auc(y_true, y_prob)
        # 随机预测的 AUC 应在 0.3-0.7 之间
        assert 0.3 <= auc <= 0.7

    def test_inverted_auc(self):
        """反转 AUC：正样本分数都低于负样本。"""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.9, 0.95, 0.1, 0.2])

        auc = compute_roc_auc(y_true, y_prob)
        assert auc == pytest.approx(0.0, rel=1e-3)
