"""Vector analysis — norm distribution, zero vectors, duplicate ratio, dimension variance.

Designed for multimodal embedding subsets (mm_emb_text/image/video) from
datasets like TAAC2025 TencentGR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorResult:
    """Vector analysis results for an embedding subset."""

    dim: int  # vector dimension
    n_vectors: int  # number of vectors (sampled)
    original_count: int  # original count before sampling
    sampled_at_load: bool  # whether pre-sampled at load time
    norm_stats: Dict[str, float]  # {mean, std, min, p25, p50, p75, p95, max}
    zero_vector_count: int  # count of all-zero vectors
    zero_vector_ratio: float  # fraction of zero vectors
    duplicate_ratio: float  # fraction of exact duplicate vectors
    dim_variance: Dict[str, float]  # {mean_var, min_var, max_var, mean_abs_mean}
    modality: str  # "text" | "image" | "video"
    skipped: bool = False
    skip_reason: Optional[str] = None


def analyze(vector_store) -> VectorResult:
    """Analyze a VectorStore: norm distribution, zero vectors, duplicate ratio, dimension variance.

    Parameters
    ----------
    vector_store : VectorStore
        Pre-sampled vector store (from taac2025.DatasetSchemaManifest subset loading).

    Returns
    -------
    VectorResult
    """
    # Lazy import to avoid circular dependency at module level
    from recsys.data.datasets.taac2025 import VectorStore  # noqa: F811

    if not isinstance(vector_store, VectorStore):
        return VectorResult(
            dim=0,
            n_vectors=0,
            original_count=0,
            sampled_at_load=False,
            norm_stats={},
            zero_vector_count=0,
            zero_vector_ratio=0.0,
            duplicate_ratio=0.0,
            dim_variance={},
            modality="unknown",
            skipped=True,
            skip_reason="Input is not a VectorStore instance.",
        )

    vectors = vector_store.vectors
    n = len(vectors)

    if n == 0 or vectors.shape[1] == 0:
        return VectorResult(
            dim=vector_store.dim,
            n_vectors=0,
            original_count=vector_store.original_count,
            sampled_at_load=vector_store.sampled_at_load,
            norm_stats={},
            zero_vector_count=0,
            zero_vector_ratio=0.0,
            duplicate_ratio=0.0,
            dim_variance={},
            modality=vector_store.modality,
            skipped=True,
            skip_reason="No vectors to analyze.",
        )

    dim = vectors.shape[1]

    # ---- L2 norm statistics ----
    norms = vector_store.norms()
    norm_stats = {
        "mean": round(float(norms.mean()), 6),
        "std": round(float(norms.std()), 6),
        "min": round(float(norms.min()), 6),
        "p25": round(float(np.percentile(norms, 25)), 6),
        "p50": round(float(np.percentile(norms, 50)), 6),
        "p75": round(float(np.percentile(norms, 75)), 6),
        "p95": round(float(np.percentile(norms, 95)), 6),
        "max": round(float(norms.max()), 6),
    }

    # ---- Zero vectors ----
    zero_mask = np.all(vectors == 0, axis=1)
    zero_count = int(zero_mask.sum())
    zero_ratio = round(zero_count / n, 6)

    # ---- Duplicate ratio ----
    dup_ratio = vector_store.duplicate_ratio()

    # ---- Per-dimension variance ----
    dim_vars = vectors.var(axis=0)
    dim_means = vectors.mean(axis=0)
    dim_variance = {
        "mean_var": round(float(dim_vars.mean()), 6),
        "min_var": round(float(dim_vars.min()), 6),
        "max_var": round(float(dim_vars.max()), 6),
        "mean_abs_mean": round(float(np.abs(dim_means).mean()), 6),
    }

    logger.info(
        "Vector analysis [%s]: %d vectors, dim=%d, "
        "norm=%.4f±%.4f, zero=%.2f%%, dup=%.2f%%",
        vector_store.modality,
        n,
        dim,
        norm_stats["mean"],
        norm_stats["std"],
        zero_ratio * 100,
        dup_ratio * 100,
    )

    return VectorResult(
        dim=dim,
        n_vectors=n,
        original_count=vector_store.original_count,
        sampled_at_load=vector_store.sampled_at_load,
        norm_stats=norm_stats,
        zero_vector_count=zero_count,
        zero_vector_ratio=zero_ratio,
        duplicate_ratio=dup_ratio,
        dim_variance=dim_variance,
        modality=vector_store.modality,
    )
