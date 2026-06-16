"""Deep Structured Semantic Model (Huang et al., 2013).

Huang et al. "Learning deep structured semantic models for web search
using clickthrough data"

Architecture:
- Two-tower: user tower + item tower
- Each tower: Embedding → MLP (ReLU) → L2 normalize
- Similarity: cosine similarity between tower outputs
- Loss: BCE with logits (pointwise binary classification)

This is the first trainable model in RecBench, serving as the
reference implementation for the NeuralRecommender → LightningRecommender
→ Trainer.fit → predict → PredictionBundle path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from recsys.core.base_model import (
    Batch,
    ModelOutput,
    NeuralRecommender,
)
from recsys.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register(
    "dssm",
    family="classical",
    year=2013,
    task_type="pointwise",
    supports_training=True,
    required_features=["user_id", "item_id"],
    default_metrics=["roc_auc", "log_loss"],
)
class DSSM(NeuralRecommender):
    """Deep Structured Semantic Model — two-tower pointwise baseline.

    双塔结构：user tower 和 item tower 各自通过 embedding + MLP
    将 ID 映射到同一语义空间，再通过余弦相似度计算匹配分数。
    训练时使用 BCE 二分类损失。

    Parameters
    ----------
    config : dict
        模型参数配置，支持：
        - embed_dim (int): embedding 维度，默认 64
        - hidden_dims (List[int]): MLP 隐藏层维度，默认 [128, 64]
        - dropout (float): MLP dropout 率，默认 0.0
    schema_metadata : dict
        数据 schema 元信息，至少包含 num_users 和 num_items。
    """

    model_name: str = "dssm"
    model_family: str = "classical"
    task_type: str = "pointwise"
    problem_type: str = "binary"
    supports_training: bool = True
    required_features: List[str] = ["user_id", "item_id"]
    default_metrics: List[str] = ["roc_auc", "log_loss"]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        schema_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, schema_metadata, **kwargs)

        cfg = config or {}
        schema = schema_metadata or {}

        num_users = schema.get("num_users", 10000)
        num_items = schema.get("num_items", 10000)
        embed_dim = cfg.get("embed_dim", 64)
        hidden_dims: List[int] = cfg.get("hidden_dims", [128, 64])
        dropout = cfg.get("dropout", 0.0)

        # Embedding layers
        self.user_emb = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)

        # User MLP: embed_dim → hidden_dims → embed_dim
        self.user_mlp = _build_mlp(embed_dim, hidden_dims, embed_dim, dropout)

        # Item MLP: embed_dim → hidden_dims → embed_dim
        self.item_mlp = _build_mlp(embed_dim, hidden_dims, embed_dim, dropout)

    def forward(self, batch: Batch) -> ModelOutput:
        """前向传播：user tower + item tower → cosine similarity."""
        user_id: Optional[Tensor] = batch.user_id
        item_id: Optional[Tensor] = batch.item_id

        if user_id is None or item_id is None:
            raise ValueError(
                "DSSM 需要 batch 中包含 user_id 和 item_id 字段。"
                "当前 batch 缺少必要字段。"
            )

        # Ensure long type for embedding lookup
        user_id = user_id.long()
        item_id = item_id.long()

        # User tower
        user_emb = self.user_emb(user_id)          # (B, embed_dim)
        user_vec = self.user_mlp(user_emb)          # (B, embed_dim)
        user_vec = nn.functional.normalize(user_vec, p=2, dim=-1)

        # Item tower
        item_emb = self.item_emb(item_id)           # (B, embed_dim)
        item_vec = self.item_mlp(item_emb)           # (B, embed_dim)
        item_vec = nn.functional.normalize(item_vec, p=2, dim=-1)

        # Cosine similarity: dot product of L2-normalized vectors
        scores = (user_vec * item_vec).sum(dim=-1)  # (B,)

        return ModelOutput(scores=scores)

    def compute_loss(
        self,
        batch: Batch,
        output: ModelOutput,
    ) -> Dict[str, Tensor]:
        """计算 BCE with logits 损失。

        scores 是 [-1, 1] 范围的余弦相似度，通过 BCEWithLogitsLoss
        转换为概率空间。label=1 表示正样本（用户与物品交互）。
        """
        scores = output.scores
        if scores is None:
            return {"loss": torch.tensor(0.0, requires_grad=True)}

        label: Optional[Tensor] = batch.label
        if label is None:
            # Fallback: use labels (plural) field
            label = batch.labels

        if label is None:
            raise ValueError(
                "DSSM compute_loss 需要 batch 中包含 label 或 labels 字段。"
            )

        label = label.float()
        loss = nn.functional.binary_cross_entropy_with_logits(scores, label)
        return {"loss": loss}


def _build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    """构建 MLP 层序列：Linear → ReLU → Dropout 重复。"""
    layers: List[nn.Module] = []
    in_dim = input_dim

    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h_dim

    # Final projection to output_dim (no activation — handled by F.normalize in caller)
    layers.append(nn.Linear(in_dim, output_dim))

    return nn.Sequential(*layers)
