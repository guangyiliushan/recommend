"""HyFormer — Hybrid Transformer 推荐模型。

基于 dist/model.py 的 PCVRHyFormer 设计，适配 recsys 框架。
支持多序列建模、特征交叉、稀疏/密集参数分离。

设计来源：dist/model.py 的 PCVRHyFormer 实现。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from recsys.core.base_model import Batch, ModelOutput, NeuralRecommender
from recsys.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register(
    "hyformer",
    family="unified",
    year=2024,
    task_type="pointwise",
    supports_training=True,
    required_features=["user_id", "item_id"],
    default_metrics=["roc_auc", "log_loss"],
)
class HyFormerAdapter(NeuralRecommender):
    """HyFormer 适配器 — 将 PCVRHyFormer 适配到 recsys 框架。

    设计原则：
    - 组合 PCVRHyFormer 核心，而非继承
    - 实现 Batch → ModelInput 转换
    - 输出收敛到 PredictionBundle
    - 支持稀疏/密集参数分离

    当前实现为简化版双塔结构，后续可扩展为完整的多序列 HyFormer。

    Parameters
    ----------
    config : dict, optional
        模型参数配置，支持：
        - d_model (int): 隐藏维度，默认 64
        - emb_dim (int): Embedding 维度，默认 64
        - num_heads (int): 注意力头数，默认 4
        - num_blocks (int): Transformer 块数，默认 2
        - dropout (float): Dropout 率，默认 0.1
    schema_metadata : dict, optional
        数据 schema 元信息，至少包含 num_users 和 num_items。
    """

    model_name: str = "hyformer"
    model_family: str = "unified"
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

        # 模型参数
        self.d_model = cfg.get("d_model", 64)
        self.emb_dim = cfg.get("emb_dim", 64)
        self.num_heads = cfg.get("num_heads", 4)
        self.num_blocks = cfg.get("num_blocks", 2)
        self.dropout = cfg.get("dropout", 0.1)

        # 获取用户/物品数量
        num_users = (
            schema_metadata.get("num_users", 10000) if schema_metadata else 10000
        )
        num_items = (
            schema_metadata.get("num_items", 10000) if schema_metadata else 10000
        )

        # 保存 ID 空间元信息（用于前向边界校验与调试）
        self.user_id_space = (
            schema_metadata.get("user_id_space", "raw") if schema_metadata else "raw"
        )
        self.item_id_space = (
            schema_metadata.get("item_id_space", "raw") if schema_metadata else "raw"
        )
        self.padding_idx = (
            schema_metadata.get("padding_idx", 0) if schema_metadata else 0
        )
        self.max_user_id = (
            schema_metadata.get("max_user_id", num_users) if schema_metadata else num_users
        )
        self.max_item_id = (
            schema_metadata.get("max_item_id", num_items) if schema_metadata else num_items
        )

        # Embedding 层（稀疏参数）
        self.user_emb = nn.Embedding(num_users + 1, self.emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(num_items + 1, self.emb_dim, padding_idx=0)

        # 用户塔（密集参数）
        self.user_proj = nn.Sequential(
            nn.Linear(self.emb_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        # 物品塔（密集参数）
        self.item_proj = nn.Sequential(
            nn.Linear(self.emb_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )

    def forward(self, batch: Batch) -> ModelOutput:
        """前向传播：user tower + item tower → cosine similarity。

        Parameters
        ----------
        batch : Batch
            标准 batch 视图，必须包含 user_id 和 item_id。

        Returns
        -------
        ModelOutput
            模型输出，包含 scores 字段。
        """
        user_id = batch.user_id
        item_id = batch.item_id

        if user_id is None or item_id is None:
            raise ValueError(
                "HyFormer 需要 batch 中包含 user_id 和 item_id 字段。"
                "当前 batch 缺少必要字段。"
            )

        # 确保 long 类型用于 embedding 查找
        user_id = user_id.long()
        item_id = item_id.long()

        # ---- 显式边界校验：防止稀疏/未 remap ID 导致越界 ----
        self._validate_ids(user_id, "user")
        self._validate_ids(item_id, "item")

        # 用户塔
        user_emb = self.user_emb(user_id)  # (B, emb_dim)
        user_vec = self.user_proj(user_emb)  # (B, d_model)
        user_vec = nn.functional.normalize(user_vec, p=2, dim=-1)

        # 物品塔
        item_emb = self.item_emb(item_id)  # (B, emb_dim)
        item_vec = self.item_proj(item_emb)  # (B, d_model)
        item_vec = nn.functional.normalize(item_vec, p=2, dim=-1)

        # 余弦相似度：L2 归一化向量的点积
        scores = (user_vec * item_vec).sum(dim=-1)  # (B,)

        return ModelOutput(scores=scores)

    def _validate_ids(self, ids: Tensor, kind: str) -> None:
        """校验 ID tensor 是否在 embedding 可索引范围内。

        若 ID 越界或为负数，抛出带上下文信息的 ValueError，
        替代 PyTorch 底层 IndexError，便于快速定位数据契约问题。

        Parameters
        ----------
        ids : Tensor
            user_id 或 item_id tensor，shape (B,)。
        kind : str
            ``"user"`` 或 ``"item"``，仅用于错误消息。
        """
        emb = self.user_emb if kind == "user" else self.item_emb
        num_embeddings = emb.num_embeddings
        id_space = self.user_id_space if kind == "user" else self.item_id_space
        declared_count = self.max_user_id if kind == "user" else self.max_item_id

        if ids.numel() == 0:
            return

        min_id = ids.min().item()
        max_id = ids.max().item()

        errors: List[str] = []
        if min_id < 0:
            errors.append(f"存在负值 {kind}_id (最小 {min_id})")
        if max_id >= num_embeddings:
            errors.append(
                f"{kind}_id 最大值 {max_id} >= embedding 大小 {num_embeddings}"
            )

        if errors:
            raise ValueError(
                f"HyFormer {kind} ID 越界：{'；'.join(errors)}。"
                f"id_space={id_space}，schema_metadata 中声明的 num_{kind}s={declared_count}，"
                f"batch 范围=[{min_id}, {max_id}]。"
                f"请确认数据集是否已完成 dense remap（0 为 padding）。"
            )

    def compute_loss(
        self,
        batch: Batch,
        output: ModelOutput,
    ) -> Dict[str, Tensor]:
        """计算 BCE with logits 损失。

        Parameters
        ----------
        batch : Batch
            标准 batch 视图，必须包含 label 或 labels 字段。
        output : ModelOutput
            模型输出，包含 scores 字段。

        Returns
        -------
        Dict[str, Tensor]
            损失字典，包含 "loss" 键。
        """
        scores = output.scores
        if scores is None:
            return {"loss": torch.tensor(0.0, requires_grad=True)}

        label = batch.label if batch.label is not None else batch.labels

        if label is None:
            raise ValueError(
                "HyFormer compute_loss 需要 batch 中包含 label 或 labels 字段。"
            )

        label = label.float()
        loss = nn.functional.binary_cross_entropy_with_logits(scores, label)
        return {"loss": loss}

    def get_sparse_params(self) -> List[nn.Parameter]:
        """返回稀疏参数（Embedding 层）。

        Returns
        -------
        List[nn.Parameter]
            稀疏参数列表，包含 user_emb 和 item_emb 的权重。
        """
        return [self.user_emb.weight, self.item_emb.weight]

    def get_dense_params(self) -> List[nn.Parameter]:
        """返回密集参数（投影层）。

        Returns
        -------
        List[nn.Parameter]
            密集参数列表，包含 user_proj 和 item_proj 的参数。
        """
        return list(self.user_proj.parameters()) + list(self.item_proj.parameters())
