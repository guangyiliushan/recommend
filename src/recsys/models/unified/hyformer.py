"""HyFormer — Hybrid Transformer 推荐模型。

PCVRHyFormer：后点击转化率预估的混合 Transformer 架构。
支持多序列建模、特征交叉、稀疏/密集参数分离、RankMixer 查询增强。

设计来源：dist/model.py 的 PCVRHyFormer 实现。

核心架构（6 阶段）：
  1. NS Token 构建 — 用户/商品静态特征编码为"非序列令牌"
  2. 序列 Token 嵌入 — 多条行为序列映射到统一语义空间
  3. 查询令牌生成 — 为每条序列独立生成 Query Tokens
  4. 多层 HyFormer 块 — 序列演化 → 交叉注意 → 令牌融合 → 查询增强
  5. 输出投影 — 融合所有域查询令牌并降维
  6. 分类器 — 输出转化概率 logits

子模块清单：
  - RotaryEmbedding, rotate_half, apply_rope_to_tensor
  - SwiGLU, RoPEMultiheadAttention, CrossAttention
  - RankMixerBlock, MultiSeqQueryGenerator
  - SwiGLUEncoder, TransformerEncoder, LongerEncoder, create_sequence_encoder
  - MultiSeqHyFormerBlock, GroupNSTokenizer, RankMixerNSTokenizer
  - HyFormerCore (纯 PyTorch 核心)
  - HyFormerAdapter (recsys 框架适配器，NeuralRecommender 子类)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.nn import functional

from recsys.core.base_model import Batch, ModelOutput, NeuralRecommender
from recsys.core.registry import MODEL_REGISTRY

# ============================================================================
# RoPE — Rotary Position Embedding
# ============================================================================


class RotaryEmbedding(nn.Module):
    """预计算并缓存 RoPE cos/sin 值。

    Attributes:
        dim: RoPE 维度（通常为 head_dim）。
        max_seq_len: 最大序列长度。
        base: 旋转频率基数。
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self._inv_freq_dtype = inv_freq.dtype
        self._inv_freq_device = inv_freq.device
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        inv_freq = self.get_buffer("inv_freq")
        assert inv_freq is not None
        t = torch.arange(seq_len, dtype=self._inv_freq_dtype, device=self._inv_freq_device)
        freqs = torch.outer(t, inv_freq)  # (seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """获取指定序列长度的 cos/sin 缓存切片。"""
        cos_buf = self.get_buffer("cos_cached")
        sin_buf = self.get_buffer("sin_cached")
        assert cos_buf is not None and sin_buf is not None, "RoPE cache not built"
        cos = cos_buf[:, :seq_len, :].to(device)
        sin = sin_buf[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    """交换并取反最后维度的前后两半。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """对单个张量施加 RoPE。

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) 或 (B, L, head_dim)
        sin: 同 cos
    Returns:
        (B, num_heads, L, head_dim)
    """
    l_val = x.shape[2]
    cos_ = cos[:, :l_val, :].unsqueeze(1)
    sin_ = sin[:, :l_val, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ============================================================================
# HyFormer 基础组件
# ============================================================================


class SwiGLU(nn.Module):
    """SwiGLU 门控激活：x1 * SiLU(x2)。"""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        return self.fc_out(x1 * functional.silu(x2))


class RoPEMultiheadAttention(nn.Module):
    """RoPE 增强的多头注意力。

    手动投影 Q/K/V 并做多头变换，在投影后可独立对 Q、K 注入 RoPE。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
        q_rope_cos: Optional[Tensor] = None,
        q_rope_sin: Optional[Tensor] = None,
        need_weights: bool = False,  # pylint: disable=unused-argument
    ) -> tuple:
        """多头注意力前向。

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True = padding
            attn_mask: (Lq, Lk) 加法掩码
            rope_cos/sin: KV 侧 RoPE
            q_rope_cos/sin: Q 侧专用 RoPE (LongerEncoder cross-attn)
        """
        b, lq, _ = query.shape
        lk = key.shape[1]

        q_proj = self.W_q(query).view(b, lq, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = self.W_k(key).view(b, lk, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = self.W_v(value).view(b, lk, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            k_proj = apply_rope_to_tensor(k_proj, rope_cos, rope_sin)
            if self.rope_on_q:
                qc = q_rope_cos if q_rope_cos is not None else rope_cos
                qs = q_rope_sin if q_rope_sin is not None else rope_sin
                q_proj = apply_rope_to_tensor(q_proj, qc, qs)

        # 构建 SDPA 掩码（float 加法格式，避免 bool expand 的隐式内存分配）
        sdpa_mask = None
        if key_padding_mask is not None:
            # (B, Lk) → (B, 1, 1, Lk) float，-inf=mask, 0=attend
            sdpa_mask = torch.zeros(
                b, 1, 1, lk, dtype=query.dtype, device=query.device,
            )
            sdpa_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        if attn_mask is not None:
            # attn_mask: (Lq, Lk), 0=attend → float additive
            float_attn = torch.zeros(
                b, 1, lq, lk, dtype=query.dtype, device=query.device,
            )
            float_attn.masked_fill_(attn_mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
            sdpa_mask = sdpa_mask + float_attn if sdpa_mask is not None else float_attn

        dropout_p = self.dropout if self.training else 0.0
        # pylint: disable=not-callable
        out = functional.scaled_dot_product_attention(
            q_proj, k_proj, v_proj, attn_mask=sdpa_mask, dropout_p=dropout_p,
        )
        # pylint: enable=not-callable
        out = torch.nan_to_num(out, nan=0.0)
        out = out.transpose(1, 2).contiguous().view(b, lq, self.d_model)
        out = out * torch.sigmoid(self.W_g(query))
        out = self.W_o(out)
        return out, None


class CrossAttention(nn.Module):
    """交叉注意力模块。

    Q 来自查询令牌，KV 来自序列令牌。仅对 KV 侧应用 RoPE (rope_on_q=False)。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = "pre",
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode
        self.attn = RoPEMultiheadAttention(
            d_model=d_model, num_heads=num_heads,
            dropout=dropout, rope_on_q=False,
        )
        if ln_mode in ("pre", "post"):
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> Tensor:
        """交叉注意力前向。

        Args:
            query: (B, Nq, D)
            key_value: (B, L, D)
            key_padding_mask: (B, L), True = padding
            rope_cos/sin: KV 侧 RoPE
        Returns:
            (B, Nq, D)
        """
        residual = query
        if self.ln_mode == "pre":
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)
        out, _ = self.attn(
            query=query, key=key_value, value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos, rope_sin=rope_sin,
        )
        out = residual + out
        if self.ln_mode == "post":
            out = self.norm_q(out)
        return out


class RankMixerBlock(nn.Module):
    """HyFormer 查询增强块。

    三步：令牌混合（无参）→ 逐令牌 FFN → 残差连接。
    d_model 必须可被 T=Nq*S+Nns 整除 (full 模式)。
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = "full",
    ) -> None:
        super().__init__()
        self.t = n_total
        self.d = d_model
        self.mode = mode
        if mode == "none":
            return
        if mode == "full":
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, q_tensor: Tensor) -> Tensor:
        """无参令牌混合：切分通道 → 转置交换 → 展平。

        (B, T, D) → (B, T, T, d_sub) → transpose(1,2) → (B, T, D)
        """
        b, t, d = q_tensor.shape
        q_split = q_tensor.view(b, t, self.t, self.d_sub)
        q_rewired = q_split.transpose(1, 2).contiguous()
        return q_rewired.view(b, t, d)

    def forward(self, q_tensor: Tensor) -> Tensor:
        """查询增强前向。

        Args:
            q_tensor: (B, T, D), T = Nq*S + Nns
        Returns:
            (B, T, D)
        """
        if self.mode == "none":
            return q_tensor
        q_hat = self.token_mixing(q_tensor) if self.mode == "full" else q_tensor
        x = self.norm(q_hat)
        x = functional.gelu(self.fc1(x))  # pylint: disable=not-callable
        x = self.dropout(x)
        q_e = self.fc2(x)
        q_boost = q_tensor + q_e
        return self.post_norm(q_boost)


class MultiSeqQueryGenerator(nn.Module):
    """多序列查询令牌生成器。

    为每条序列独立生成查询令牌：
        GlobalInfo_i = Concat(NS_flat, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,Nq}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model
        global_info_dim = (num_ns + 1) * d_model
        self.global_info_norm = nn.LayerNorm(global_info_dim)
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
    ) -> list:
        """生成查询令牌。

        Args:
            ns_tokens: (B, M, D), 共享 NS 令牌
            seq_tokens_list: List[(B, L_i, D)], 长度 S
            seq_padding_masks: List[(B, L_i)], True = padding
        Returns:
            List[(B, Nq, D)], 长度 S
        """
        b = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(b, -1)
        q_tokens_list = []
        for i in range(self.num_sequences):
            valid_mask = (~seq_padding_masks[i]).unsqueeze(-1).float()
            seq_sum = (seq_tokens_list[i] * valid_mask).sum(dim=1)
            seq_count = valid_mask.sum(dim=1).clamp(min=1)
            seq_pooled = seq_sum / seq_count
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)
            global_info = self.global_info_norm(global_info)
            queries = [ffn(global_info) for ffn in cast(nn.ModuleList, self.query_ffns_per_seq[i])]
            q_tokens_list.append(torch.stack(queries, dim=1))
        return q_tokens_list


# ============================================================================
# 序列编码器
# ============================================================================


class SwiGLUEncoder(nn.Module):
    """轻量级无注意力编码器：x + Dropout(SwiGLU(LN(x)))。"""

    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> tuple:
        _ = kwargs  # SwiGLUEncoder ignores extra kwargs (e.g., rope_cos/rope_sin)
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        return residual + x, key_padding_mask


class TransformerEncoder(nn.Module):
    """标准 Transformer 编码器层（Pre-LN + RoPE 自注意力 + GELU FFN）。"""

    def __init__(
        self, d_model: int, num_heads: int,
        hidden_mult: int = 4, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model, num_heads=num_heads,
            dropout=dropout, rope_on_q=True,
        )
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> tuple:
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos, rope_sin=rope_sin,
        )
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return residual + x, key_padding_mask


class LongerEncoder(nn.Module):
    """Top-K 压缩序列编码器。

    自适应行为：
    - L > top_k: 交叉注意力。取最近 top_k 为 Q，全序列为 KV。
    - L <= top_k: 自注意力。Q=K=V=top_k 令牌。

    始终输出 (B, top_k, D)。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = RoPEMultiheadAttention(
            d_model=d_model, num_heads=num_heads,
            dropout=dropout, rope_on_q=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.Dropout(dropout),
        )

    def _gather_top_k(
        self, x: Tensor, key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """从每个样本选取最近 top_k 个有效位置。

        Returns:
            top_k_tokens: (B, top_k, D)
            new_mask: (B, top_k), True = padding
            pos_indices: (B, top_k), 原始位置索引
        """
        b, l_val, d = x.shape
        device = x.device
        valid_len = (~key_padding_mask).sum(dim=1)
        actual_k = torch.clamp(valid_len, max=self.top_k)
        start_pos = valid_len - actual_k
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(b, -1)
        indices = start_pos.unsqueeze(1) + offsets
        indices = torch.clamp(indices, min=0, max=l_val - 1)
        idx_exp = indices.unsqueeze(-1).expand(-1, -1, d)
        top_k_tokens = torch.gather(x, dim=1, index=idx_exp)
        new_pad_count = self.top_k - actual_k
        pos_idxs = torch.arange(self.top_k, device=device).unsqueeze(0)
        new_mask = pos_idxs < new_pad_count.unsqueeze(1)
        top_k_tokens = top_k_tokens * (~new_mask).unsqueeze(-1).float()
        return top_k_tokens, new_mask, indices

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """LongerEncoder 前向。

        Returns:
            output: (B, top_k, D)
            new_key_padding_mask: (B, top_k)
        """
        b, l_val, _d = x.shape
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(b, l_val, dtype=torch.bool, device=x.device)
        if l_val > self.top_k:
            q, new_mask, q_pos = self._gather_top_k(x, key_padding_mask)
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)
            q_rope_cos, q_rope_sin = None, None
            if rope_cos is not None and rope_sin is not None:
                head_dim = rope_cos.shape[2]
                cos_exp = rope_cos.expand(b, -1, -1)
                sin_exp = rope_sin.expand(b, -1, -1)
                idx = q_pos.unsqueeze(-1).expand(-1, -1, head_dim)
                q_rope_cos = torch.gather(cos_exp, 1, idx)
                q_rope_sin = torch.gather(sin_exp, 1, idx)
            attn_out, _ = self.attn(
                query=q_normed, key=kv_normed, value=kv_normed,
                key_padding_mask=key_padding_mask,
                rope_cos=rope_cos, rope_sin=rope_sin,
                q_rope_cos=q_rope_cos, q_rope_sin=q_rope_sin,
            )
            out = q + attn_out
        else:
            new_mask = key_padding_mask
            x_normed = self.norm_q(x)
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    l_val, device=x.device,
                )
            attn_out, _ = self.attn(
                query=x_normed, key=x_normed, value=x_normed,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                rope_cos=rope_cos, rope_sin=rope_sin,
            )
            out = x + attn_out
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        return residual + out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False,
) -> nn.Module:
    """序列编码器工厂。

    Args:
        encoder_type: "swiglu" | "transformer" | "longer"
    """
    if encoder_type == "swiglu":
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    if encoder_type == "transformer":
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    if encoder_type == "longer":
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    raise ValueError(f"Unknown encoder type: {encoder_type}")


# ============================================================================
# HyFormer 核心块
# ============================================================================


class MultiSeqHyFormerBlock(nn.Module):
    """多序列 HyFormer 块。

    每块执行三个子阶段：
    1. 独立序列演化（每序列独立编码）
    2. 独立查询解码（每序列查询与演化后序列交互）
    3. 联合查询增强（所有 Q + NS 令牌合并 → RankMixer → 拆分）
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = "swiglu",
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = "full",
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type, d_model=d_model,
                num_heads=num_heads, hidden_mult=hidden_mult,
                dropout=dropout, top_k=top_k, causal=causal,
            )
            for _ in range(num_sequences)
        ])
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model, num_heads=num_heads,
                dropout=dropout, ln_mode="pre",
            )
            for _ in range(num_sequences)
        ])
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model, n_total=n_total,
            hidden_mult=hidden_mult, dropout=dropout, mode=rank_mixer_mode,
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[Tensor]] = None,
        rope_sin_list: Optional[List[Tensor]] = None,
    ) -> Tuple[list, Tensor, list, list]:
        s = self.num_sequences
        nq = self.num_queries
        # 1. 序列演化
        next_seqs, next_masks = [], []
        for i in range(s):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            si, mi = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seqs.append(si)
            next_masks.append(mi)
        # 2. 查询解码
        decoded_qs = []
        for i in range(s):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            dq = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            decoded_qs.append(dq)
        # 3. 令牌融合 + RankMixer + 拆分
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)
        boosted = self.mixer(combined)
        next_q_list = []
        offset = 0
        for _ in range(s):
            next_q_list.append(boosted[:, offset:offset + nq, :])
            offset += nq
        next_ns = boosted[:, offset:, :]
        return next_q_list, next_ns, next_seqs, next_masks


# ============================================================================
# NS 令牌化器
# ============================================================================


class GroupNSTokenizer(nn.Module):
    """分组式 NS 令牌化器。

    按语义组（group）聚合特征：组内多值特征 mean pooling → 拼接 → SiLU(Linear)。
    每 group 输出一个 d_model 维令牌。
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold
        embs = []
        for vs, _offset, _length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            embs.append(
                None if skip else nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0)
            )
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: Tensor) -> Tensor:
        b_val = int_feats.shape[0]
        tokens = []
        group_projs: nn.ModuleList = self.group_projs
        for group, proj in zip(self.groups, group_projs, strict=False):
            fid_embs = []
            for fid_idx in group:
                _vs, offset, length = self.feature_specs[fid_idx]
                emb_real = self._emb_index[fid_idx]
                if emb_real == -1:
                    fid_emb = int_feats.new_zeros(b_val, self.emb_dim)
                else:
                    emb_layer = cast(nn.Embedding, self.embs[emb_real])
                    if length == 1:
                        ids = int_feats[:, offset].long().clamp(
                            0, int(emb_layer.num_embeddings) - 1,
                        )
                        fid_emb = emb_layer(ids)
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        vals = vals.clamp(0, int(emb_layer.num_embeddings) - 1)
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)
            tokens.append(functional.silu(proj(cat_emb)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


class RankMixerNSTokenizer(nn.Module):
    """RankMixer 风格 NS 令牌化器。

    所有特征 Embedding 全局拼接 → 等距切分为 num_ns_tokens 段 → 每段独立投影。
    令牌数量可自由选择，不受限于特征组数。
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold
        embs = []
        for vs, _offset, _length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            embs.append(
                None if skip else nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0)
            )
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim
        self.token_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(self.chunk_dim, d_model), nn.LayerNorm(d_model))
            for _ in range(num_ns_tokens)
        ])

    def forward(self, int_feats: Tensor) -> Tensor:
        b_val = int_feats.shape[0]
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                _vs, offset, length = self.feature_specs[fid_idx]
                emb_real = self._emb_index[fid_idx]
                if emb_real == -1:
                    all_embs.append(int_feats.new_zeros(b_val, self.emb_dim))
                else:
                    emb_layer = cast(nn.Embedding, self.embs[emb_real])
                    if length == 1:
                        ids = int_feats[:, offset].long().clamp(
                            0, int(emb_layer.num_embeddings) - 1,
                        )
                        all_embs.append(emb_layer(ids))
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        vals = vals.clamp(0, int(emb_layer.num_embeddings) - 1)
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        all_embs.append((emb_all * mask).sum(dim=1) / count)
        cat_emb = torch.cat(all_embs, dim=-1)
        if self._pad_size > 0:
            cat_emb = functional.pad(cat_emb, (0, self._pad_size))
        tokens = []
        token_projs: nn.ModuleList = self.token_projs
        for chunk, proj in zip(
            cat_emb.split(self.chunk_dim, dim=-1), token_projs, strict=False,
        ):
            tokens.append(functional.silu(proj(chunk)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


# ============================================================================
# PCVRHyFormer 纯 PyTorch 核心
# ============================================================================


class HyFormerCore(nn.Module):
    """HyFormer 纯 PyTorch 核心 — 不依赖 recsys 框架的任何类型。

    此模块只处理原始 Tensor 输入，由 HyFormerAdapter 负责 Batch 适配。
    支持有/无序列两种场景：num_sequences>0 时完整 HyFormer 管线，
    num_sequences=0 时 NS tokens mean pool → classifier。
    """

    def __init__(  # noqa: PLR0915
        self,
        # 数据 schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # 模型超参数
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = "transformer",
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = "full",
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        ns_tokenizer_type: str = "rankmixer",
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.num_hyformer_blocks = num_hyformer_blocks
        self.num_heads = num_heads
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.dropout_rate = dropout_rate

        # 安全保护：emb_skip_threshold=0 时，对超大词表特征自动跳过 Embedding 创建
        # 防止高基数特征（如原始 item_id）导致内存爆炸
        max_vocab = 0
        for dom_vs in seq_vocab_sizes.values():
            for vs in dom_vs:
                if int(vs) > max_vocab:
                    max_vocab = int(vs)
        for specs in [user_int_feature_specs, item_int_feature_specs]:
            for vs, _o, _l in specs:
                if int(vs) > max_vocab:
                    max_vocab = int(vs)
        if self.emb_skip_threshold <= 0 and max_vocab > 100_000:
            self.emb_skip_threshold = 100_000
            logger.info(
                f"检测到高基数特征 (max_vocab={max_vocab})，自动设置 "
                f"emb_skip_threshold={self.emb_skip_threshold} 防止内存溢出"
            )

        # --- NS Token 构建 ---
        if ns_tokenizer_type == "group":
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs, groups=user_ns_groups,
                emb_dim=emb_dim, d_model=d_model,
                emb_skip_threshold=self.emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)
            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs, groups=item_ns_groups,
                emb_dim=emb_dim, d_model=d_model,
                emb_skip_threshold=self.emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == "rankmixer":
            u_tokens = user_ns_tokens if user_ns_tokens > 0 else len(user_ns_groups)
            i_tokens = item_ns_tokens if item_ns_tokens > 0 else len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs, groups=user_ns_groups,
                emb_dim=emb_dim, d_model=d_model, num_ns_tokens=u_tokens,
                emb_skip_threshold=self.emb_skip_threshold,
            )
            num_user_ns = u_tokens
            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs, groups=item_ns_groups,
                emb_dim=emb_dim, d_model=d_model, num_ns_tokens=i_tokens,
                emb_skip_threshold=self.emb_skip_threshold,
            )
            num_item_ns = i_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, d_model), nn.LayerNorm(d_model),
            )
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model), nn.LayerNorm(d_model),
            )

        self.num_ns = (
            num_user_ns + (1 if self.has_user_dense else 0)
            + num_item_ns + (1 if self.has_item_dense else 0)
        )

        # --- 无序列模式：num_sequences=0 时跳过序列管线 ---
        if self.num_sequences == 0:
            # 代码路径：NS tokens → mean pool → output_proj → classifier
            total_tokens = self.num_ns
            if rank_mixer_mode == "full" and d_model % total_tokens != 0:
                logger.warning(
                    f"RankMixer full 模式要求 d_model={d_model} 能被 total_tokens={total_tokens} "
                    f"整除，自动回退为 ffn_only 模式"
                )
                rank_mixer_mode = "ffn_only"

            self._seq_embs = nn.ModuleDict()
            self._seq_emb_index = {}
            self._seq_is_id = {}
            self._seq_vocab_sizes = {}
            self._seq_proj = nn.ModuleDict()
            self.time_embedding = None
            self.query_generator = None
            self.blocks = None
            self.rotary_emb = None

            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            self.emb_dropout = nn.Dropout(dropout_rate)
            self.clsfier = nn.Sequential(
                nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
                nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(d_model, action_num),
            )
            self._init_params()
            return

        total_tokens = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == "full" and d_model % total_tokens != 0:
            logger.warning(
                f"RankMixer full 模式要求 d_model={d_model} 能被 total_tokens={total_tokens} "
                f"整除，自动回退为 ffn_only 模式"
            )
            rank_mixer_mode = "ffn_only"

        # --- 序列 Token 嵌入 ---
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (
                    self.emb_skip_threshold > 0
                    and int(vs) > self.emb_skip_threshold
                )
                embs_raw.append(
                    None if skip
                    else nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0)
                )
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            index_map = []
            real = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real)
                    real += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index: Dict[str, List[int]] = {}
        self._seq_is_id: Dict[str, List[bool]] = {}
        self._seq_vocab_sizes: Dict[str, List[int]] = {}
        self._seq_proj = nn.ModuleDict()
        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model), nn.LayerNorm(d_model),
            )

        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # --- HyFormer 核心组件 ---
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model, num_ns=self.num_ns,
            num_queries=num_queries, num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
        )
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model, num_heads=num_heads,
                num_queries=num_queries, num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult, dropout=dropout_rate,
                top_k=seq_top_k, causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
            )
            for _ in range(num_hyformer_blocks)
        ])

        if use_rope:
            self.rotary_emb = RotaryEmbedding(dim=d_model // num_heads, base=rope_base)
        else:
            self.rotary_emb = None

        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(d_model, action_num),
        )
        self._init_params()

    def _init_params(self) -> None:
        """Xavier 初始化所有 Embedding，padding_idx=0 置零。"""
        for domain in self.seq_domains:
            seq_embs = cast(nn.ModuleList, self._seq_embs[domain])
            for emb in seq_embs:
                emb_w = cast(nn.Embedding, emb).weight
                nn.init.xavier_normal_(emb_w.data)
                emb_w.data[0, :] = 0
        for tok in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tok.embs:
                emb_w = cast(nn.Embedding, emb).weight
                nn.init.xavier_normal_(emb_w.data)
                emb_w.data[0, :] = 0
        if self.num_time_buckets > 0 and self.time_embedding is not None:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000,
    ) -> set:
        """重新初始化高基数 Embedding，保留低基数与时间特征。

        Args:
            cardinality_threshold: 词表超过此值的 Embedding 被重置。
        Returns:
            被重置参数的 data_ptr() 集合。
        """
        reinit_ptrs: set = set()
        # 序列 Embedding
        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            embs = cast(nn.ModuleList, emb_list)
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    continue
                emb = cast(nn.Embedding, embs[real_idx])
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
        # NS tokenizer Embedding
        for tok in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for i, (vs, _offset, _length) in enumerate(tok.feature_specs):
                real_idx = tok._emb_index[i]  # pylint: disable=protected-access
                if real_idx == -1:
                    continue
                emb = cast(nn.Embedding, tok.embs[real_idx])
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
        return reinit_ptrs

    def _embed_seq_domain(
        self,
        seq: Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: Tensor,
    ) -> Tensor:
        """嵌入一个序列域：拼接 sideinfo Embedding 并投影到 d_model。"""
        b_val, s, l_val = seq.shape
        emb_list = []
        for i in range(s):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                emb_list.append(seq.new_zeros(b_val, l_val, self.emb_dim))
            else:
                emb = cast(nn.Embedding, sideinfo_embs[real_idx])
                seq_ids = seq[:, i, :].clamp(0, int(emb.num_embeddings) - 1)
                e = emb(seq_ids)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        # 安全守卫：该域所有特征 Embedding 均被跳过时，直接返回零张量
        if not emb_list:
            return seq.new_zeros(b_val, l_val, self.d_model)
        cat_emb = torch.cat(emb_list, dim=-1)
        token_emb = functional.gelu(proj(cat_emb))  # pylint: disable=not-callable
        if self.num_time_buckets > 0 and self.time_embedding is not None:
            token_emb = token_emb + self.time_embedding(
                time_bucket_ids.clamp(0, self.num_time_buckets - 1)
            )
        return token_emb

    def _make_padding_mask(self, seq_len: Tensor, max_len: int) -> Tensor:
        idx = torch.arange(max_len, device=seq_len.device).unsqueeze(0)
        return idx >= seq_len.unsqueeze(1)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True,
    ) -> Tensor:
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]
        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list
        for block in cast(nn.ModuleList, self.blocks):
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list, rope_sin_list = [], []
                device = curr_seqs[0].device
                for si in curr_seqs:
                    cos, sin = self.rotary_emb(si.shape[1], device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)
            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs, ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs, seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list, rope_sin_list=rope_sin_list,
            )
        b_val = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)
        output = all_q.view(b_val, -1)
        return self.output_proj(output)

    def forward(
        self,
        user_int_feats: Tensor,
        item_int_feats: Tensor,
        user_dense_feats: Optional[Tensor],
        item_dense_feats: Optional[Tensor],
        seq_data: Dict[str, Tensor],
        seq_lens: Dict[str, Tensor],
        seq_time_buckets: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """HyFormerCore 前向。

        Args:
            user_int_feats: (B, total_user_int_dim) long
            item_int_feats: (B, total_item_int_dim) long
            user_dense_feats: (B, user_dense_dim) float 或 None
            item_dense_feats: (B, item_dense_dim) float 或 None
            seq_data: {domain: (B, S, L)} long
            seq_lens: {domain: (B,)} long
            seq_time_buckets: {domain: (B, L)} long 或 None
        Returns:
            output tensor (B, action_num)
        """
        # 安全 clamp：防止超大特征值导致 Embedding 索引越界
        user_int_feats = user_int_feats.clamp(0, 10_000_000)
        item_int_feats = item_int_feats.clamp(0, 10_000_000)

        # 1. NS tokens
        user_ns = self.user_ns_tokenizer(user_int_feats)
        item_ns = self.item_ns_tokenizer(item_int_feats)
        ns_parts = [user_ns]
        if self.has_user_dense and user_dense_feats is not None:
            ns_parts.append(
                functional.silu(self.user_dense_proj(user_dense_feats)).unsqueeze(1)
            )
        ns_parts.append(item_ns)
        if self.has_item_dense and item_dense_feats is not None:
            ns_parts.append(
                functional.silu(self.item_dense_proj(item_dense_feats)).unsqueeze(1)
            )
        ns_tokens = torch.cat(ns_parts, dim=1)

        # 无序列模式：NS tokens → mean pool → output_proj → classifier
        if self.num_sequences == 0:
            output = ns_tokens.mean(dim=1)  # (B, D)
            if self.training:
                output = self.emb_dropout(output)
            output = self.output_proj(output)
            return self.clsfier(output)

        # 2. 序列嵌入
        if seq_time_buckets is None:
            seq_time_buckets = {}
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            if domain not in seq_data or domain not in seq_lens:
                # 缺失序列域：用零张量填充，保持序列数量一致
                logger.warning(
                    "序列域 '{}' 在 batch 中缺失，用零张量填充。"
                    "可用域: seq_data={}, seq_lens={}",
                    domain, list(seq_data.keys()), list(seq_lens.keys()),
                )
                # 从已有序列推断 batch_size 和 device
                if seq_tokens_list:
                    ref = seq_tokens_list[0]
                    b_val, _, l_ref = ref.shape
                    device = ref.device
                else:
                    b_val = ns_tokens.shape[0]
                    l_ref = 1
                    device = ns_tokens.device
                dummy_tokens = torch.zeros(b_val, l_ref, self.d_model, device=device)
                dummy_mask = torch.ones(b_val, l_ref, dtype=torch.bool, device=device)
                seq_tokens_list.append(dummy_tokens)
                seq_masks_list.append(dummy_mask)
                continue
            sdata = seq_data[domain]
            t_buckets = seq_time_buckets.get(
                domain,
                torch.zeros(
                    sdata.shape[0], sdata.shape[2], dtype=torch.long, device=sdata.device,
                ),
            )
            tokens = self._embed_seq_domain(
                sdata, cast(nn.ModuleList, self._seq_embs[domain]),
                self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain], t_buckets,
            )
            seq_tokens_list.append(tokens)
            seq_masks_list.append(
                self._make_padding_mask(seq_lens[domain], sdata.shape[2])
            )

        # 3. Query Generation
        q_tokens_list = cast(MultiSeqQueryGenerator, self.query_generator)(
            ns_tokens, seq_tokens_list, seq_masks_list,
        )

        # 4. HyFormer Blocks + Output Projection
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training,
        )

        # 5. Classifier
        return self.clsfier(output)


# ============================================================================
# HyFormerAdapter — recsys 框架适配器
# ============================================================================

# ---------------------------------------------------------------------------
# 任务类型映射表：将用户友好的 task 别名映射到框架严谨的 task_type/problem_type
# ---------------------------------------------------------------------------
TASK_MAPPING: Dict[str, Dict[str, Any]] = {
    "ctr": {
        "task_type": "pointwise",
        "problem_type": "binary",
        "default_metrics": ["roc_auc", "log_loss"],
        "default_action_num": 1,
    },
    "cvr": {
        "task_type": "pointwise",
        "problem_type": "binary",
        "default_metrics": ["roc_auc", "log_loss"],
        "default_action_num": 1,
    },
    "binary": {
        "task_type": "pointwise",
        "problem_type": "binary",
        "default_metrics": ["roc_auc", "log_loss"],
        "default_action_num": 1,
    },
    "multiclass": {
        "task_type": "pointwise",
        "problem_type": "multiclass",
        "default_metrics": ["accuracy", "f1_macro"],
        "default_action_num": None,
    },
    "regression": {
        "task_type": "pointwise",
        "problem_type": "regression",
        "default_metrics": ["mse", "mae"],
        "default_action_num": 1,
    },
    "multitask": {
        "task_type": "multitask",
        "problem_type": "binary",
        "default_metrics": ["roc_auc", "log_loss"],
        "default_action_num": None,
    },
    "ranking": {
        "task_type": "ranking",
        "problem_type": "implicit_ranking",
        "default_metrics": ["ndcg_at_k", "recall_at_k"],
        "default_action_num": 1,
    },
}

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
    """HyFormer 适配器 — 将 HyFormerCore 适配到 recsys 框架。

    设计原则：
    - 组合 HyFormerCore（唯一路径，无降级模式）
    - 实现 Batch → Tensor 转换
    - 通过 task 配置别名动态切换任务类型与输出格式
    - 支持稀疏/密集参数分离

    Parameters
    ----------
    config : dict, optional
        模型参数配置，核心新增字段：
        - task: 任务别名 (ctr/cvr/binary/multiclass/regression/multitask/ranking)
        - 其余超参数透传给 HyFormerCore。
    schema_metadata : dict, optional
        必须包含：
        - user_int_feature_specs, item_int_feature_specs
        - user_dense_dim, item_dense_dim, seq_vocab_sizes
        - user_ns_groups, item_ns_groups, seq_domains
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
        meta = schema_metadata or {}

        # ---- schema 必需校验 ----
        user_specs: List = meta.get("user_int_feature_specs", [])
        if not user_specs:
            raise ValueError(
                "HyFormer 需要结构化特征 schema。请确认数据集实现了 "
                "get_schema_metadata() 并返回 user_int_feature_specs。"
            )

        # ---- 任务解析 ----
        task_alias = cfg.get("task", "ctr")
        task_info = TASK_MAPPING.get(task_alias)
        if task_info is None:
            raise ValueError(
                f"未知 task 别名：{task_alias}。"
                f"支持的任务：{list(TASK_MAPPING.keys())}"
            )

        self.task = task_alias
        self.task_type = task_info["task_type"]
        self.problem_type = task_info["problem_type"]
        self.default_metrics = task_info["default_metrics"]

        # action_num 推导
        default_action_num = task_info["default_action_num"]
        if cfg.get("action_num") is not None:
            self.action_num = cfg["action_num"]
        elif default_action_num is not None:
            self.action_num = default_action_num
        elif task_alias == "multiclass":
            self.action_num = cfg.get("num_classes", 2)
        elif task_alias == "multitask":
            sub_tasks = cfg.get("sub_tasks", [])
            self.action_num = cfg.get("num_tasks", len(sub_tasks) if sub_tasks else 1)
        else:
            self.action_num = 1

        # ---- 通用参数 ----
        self.d_model = cfg.get("d_model", 64)
        self.emb_dim = cfg.get("emb_dim", 64)
        self.num_heads_cfg = cfg.get("num_heads", 4)
        self.num_hyformer_blocks_cfg = cfg.get("num_hyformer_blocks", 2)
        self.dropout_rate = cfg.get("dropout_rate", 0.1)

        # ---- 自动缩容：小数据集降低模型维度防止 OOM ----
        num_users = meta.get("num_users", meta.get("max_user_id", 0))
        num_items = meta.get("num_items", meta.get("max_item_id", 0))
        dataset_scale = max(num_users, num_items, 1)
        user_explicit_dims = "d_model" in cfg or "emb_dim" in cfg
        if num_users > 0 and num_items > 0 and dataset_scale < 10_000 and not user_explicit_dims:
            # 用户未显式指定模型维度时，小数据集自动缩容
            self.d_model = 32
            self.emb_dim = 32
            self.num_heads_cfg = 2
            self.num_hyformer_blocks_cfg = 1
            logger.info(
                "小数据集 (users=%d, items=%d)，自动缩容: d_model=%d, emb_dim=%d, "
                "heads=%d, blocks=%d。可通过显式设置 d_model / emb_dim 覆盖。",
                num_users, num_items, self.d_model, self.emb_dim,
                self.num_heads_cfg, self.num_hyformer_blocks_cfg,
            )

        # ---- 构造 HyFormerCore（唯一路径）----
        self.core = HyFormerCore(
            user_int_feature_specs=user_specs,
            item_int_feature_specs=meta.get("item_int_feature_specs", []),
            user_dense_dim=meta.get("user_dense_dim", 0),
            item_dense_dim=meta.get("item_dense_dim", 0),
            seq_vocab_sizes=meta.get("seq_vocab_sizes", {}),
            user_ns_groups=meta.get("user_ns_groups", []),
            item_ns_groups=meta.get("item_ns_groups", []),
            d_model=self.d_model,
            emb_dim=self.emb_dim,
            num_queries=cfg.get("num_queries", 1),
            num_hyformer_blocks=self.num_hyformer_blocks_cfg,
            num_heads=self.num_heads_cfg,
            seq_encoder_type=cfg.get("seq_encoder_type", "transformer"),
            hidden_mult=cfg.get("hidden_mult", 4),
            dropout_rate=self.dropout_rate,
            seq_top_k=cfg.get("seq_top_k", 50),
            seq_causal=cfg.get("seq_causal", False),
            action_num=self.action_num,
            num_time_buckets=cfg.get("num_time_buckets", 65),
            rank_mixer_mode=cfg.get("rank_mixer_mode", "full"),
            use_rope=cfg.get("use_rope", False),
            rope_base=cfg.get("rope_base", 10000.0),
            emb_skip_threshold=cfg.get("emb_skip_threshold", 0),
            seq_id_threshold=cfg.get("seq_id_threshold", 10000),
            ns_tokenizer_type=cfg.get("ns_tokenizer_type", "rankmixer"),
            user_ns_tokens=cfg.get("user_ns_tokens", 0),
            item_ns_tokens=cfg.get("item_ns_tokens", 0),
        )
        self.num_heads = self.core.num_heads
        self.num_hyformer_blocks = self.core.num_hyformer_blocks
        self.dropout_rate = self.core.dropout_rate
        self.core_attrs = self.core  # type: ignore[attr-defined]

        # ---- 模型大小守护 ----
        total_params = sum(p.numel() for p in self.core.parameters())
        total_params_m = total_params / 1_000_000
        if total_params_m > 50:
            logger.warning(
                "HyFormer 参数量较大 ({:.1f}M)，小数据集上可能 OOM。"
                "建议：减小 d_model/emb_dim，或增加 emb_skip_threshold。",
                total_params_m,
            )
        if total_params_m > 200:
            logger.error(
                "HyFormer 参数量过大 ({:.1f}M)，极易 OOM。"
                "请将 d_model 降至 32 以下或 emb_skip_threshold 升至 10000 以上。",
                total_params_m,
            )

        # 动态 required_features
        if self.task_type == "multitask":
            self.required_features = [
                "user_int_feats", "item_int_feats", "seq_data", "seq_lens",
                "task_labels",
            ]
        else:
            self.required_features = [
                "user_int_feats", "item_int_feats", "seq_data", "seq_lens",
            ]

        logger.info(
            "HyFormer 已启用 | task={}, task_type={}, problem_type={}, "
            "action_num={}, NS tokens={}, seq domains={}, d_model={}, "
            "heads={}, blocks={}, encoder={}, rank_mixer={}, rope={}, "
            "params={:.1f}M",
            self.task, self.task_type, self.problem_type, self.action_num,
            self.core.num_ns, self.core.num_sequences, self.d_model,
            self.num_heads, self.num_hyformer_blocks,
            cfg.get("seq_encoder_type", "transformer"),
            cfg.get("rank_mixer_mode", "full"),
            cfg.get("use_rope", False),
            total_params_m,
        )

    def forward(self, batch: Batch) -> ModelOutput:
        """前向传播 — 任务自适应输出格式。

        Returns:
            ModelOutput: 根据 task 类型产出不同字段组合。
        """
        user_int_feats = batch.get("user_int_feats")
        item_int_feats = batch.get("item_int_feats")
        if user_int_feats is None or item_int_feats is None:
            raise ValueError(
                "HyFormer 需要 batch 中包含 user_int_feats 和 item_int_feats。"
                f" 可用键: {list(batch.data.keys()) if hasattr(batch, 'data') else 'N/A'}"
            )

        user_dense: Optional[Tensor] = batch.get("user_dense_feats")  # type: ignore[assignment]
        item_dense: Optional[Tensor] = batch.get("item_dense_feats")  # type: ignore[assignment]

        # 序列数据：num_sequences=0 时空 dict
        seq_data_raw = batch.get("seq_data") or {}
        seq_lens_raw = batch.get("seq_lens") or {}
        seq_time_buckets = batch.get("seq_time_buckets")

        # 输入校验：确保 seq_data/seq_lens 为 dict 类型（collate 后可能为其他类型）
        if not isinstance(seq_data_raw, dict):
            logger.warning(
                "seq_data 类型异常 ({}), 已重置为空 dict", type(seq_data_raw).__name__,
            )
            seq_data_raw = {}
        if not isinstance(seq_lens_raw, dict):
            logger.warning(
                "seq_lens 类型异常 ({}), 已重置为空 dict", type(seq_lens_raw).__name__,
            )
            seq_lens_raw = {}

        logits = self.core(
            user_int_feats=user_int_feats.long(),
            item_int_feats=item_int_feats.long(),
            user_dense_feats=cast(Tensor, user_dense).float() if user_dense is not None else None,  # type: ignore[union-attr]
            item_dense_feats=cast(Tensor, item_dense).float() if item_dense is not None else None,  # type: ignore[union-attr]
            seq_data=seq_data_raw,
            seq_lens=seq_lens_raw,
            seq_time_buckets=seq_time_buckets,
        )  # (B, action_num)

        # NaN 守卫：检测前向输出异常，防止 BCE 在极端 logits 时静默 NaN
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()
            logger.warning(
                "HyFormer forward 输出异常: NaN=%d, Inf=%d, 回退为零 logits",
                nan_count, inf_count,
            )
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

        # 任务自适应后处理
        if self.task == "multitask":
            probs = torch.sigmoid(logits)
            task_outputs = {
                f"task_{i}": probs[:, i] for i in range(self.action_num)
            }
            return ModelOutput(
                scores=logits, probs=probs, task_outputs=task_outputs,
            )
        elif self.task == "multiclass":
            probs = functional.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            return ModelOutput(scores=logits, probs=probs, preds=preds)
        elif self.task == "regression":
            return ModelOutput(scores=logits.squeeze(-1))
        else:  # binary / ctr / cvr / ranking
            probs = torch.sigmoid(logits).squeeze(-1)
            return ModelOutput(scores=logits.squeeze(-1), probs=probs)

    def compute_loss(
        self, batch: Batch, output: ModelOutput,
    ) -> Dict[str, Tensor]:
        """计算任务自适应损失。

        Args:
            batch: 标准 batch 视图。
            output: 模型输出，包含 scores 字段。
        """
        scores = output.scores
        if scores is None:
            return {"loss": torch.tensor(0.0, requires_grad=True)}

        if self.task_type == "multitask":
            task_labels = cast(Optional[Tensor], batch.task_labels)  # type: ignore[arg-type]
            if task_labels is None:
                raise ValueError(
                    "multitask 模式需要 batch 中包含 task_labels。"
                )
            label = task_labels.float()
            per_task = functional.binary_cross_entropy_with_logits(
                scores, label, reduction="none",
            )
            task_masks = cast(Optional[Tensor], batch.task_masks)  # type: ignore[arg-type]
            if task_masks is not None:
                per_task = per_task * task_masks.float()
                loss = per_task.sum() / task_masks.float().sum().clamp(min=1)
            else:
                loss = per_task.mean()
            return {"loss": loss}

        elif self.problem_type == "multiclass":
            label = batch.label
            if label is None:
                label = batch.labels
            if label is None:
                raise ValueError("multiclass 模式需要 batch 中包含 label。")
            return {"loss": functional.cross_entropy(scores, label.long())}

        elif self.problem_type == "regression":
            label = batch.label
            if label is None:
                label = batch.labels
            if label is None:
                raise ValueError("regression 模式需要 batch 中包含 label。")
            return {"loss": functional.mse_loss(scores, label.float())}

        else:  # binary / ranking
            label = batch.label if batch.label is not None else batch.labels
            if label is None:
                raise ValueError(
                    "HyFormer compute_loss 需要 batch 中包含 label 或 labels 字段。"
                )
            loss = functional.binary_cross_entropy_with_logits(
                scores, label.float(),
            )
            return {"loss": loss}

    def get_sparse_params(self) -> List[nn.Parameter]:
        """返回所有 nn.Embedding 权重（供 Adagrad 优化）。"""
        sparse_ptrs = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_ptrs.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_ptrs]

    def get_dense_params(self) -> List[nn.Parameter]:
        """返回所有非 Embedding 参数（供 AdamW 优化）。"""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000,
    ) -> set:
        """选择性重新初始化高基数 Embedding。

        Args:
            cardinality_threshold: 词表超过此值的 Embedding 被重置。
        Returns:
            被重置参数的 data_ptr() 集合。
        """
        return self.core.reinit_high_cardinality_params(cardinality_threshold)
