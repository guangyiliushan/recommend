"""Feature engineering utilities for large-scale offline datasets.

Capabilities:
    - Count / frequency encoding (with chunk-aware aggregation)
    - Target encoding (with offline aggregate caching)
    - Category dictionary encoding
    - Numeric normalization (min-max, z-score, log1p)
    - Feature crossing / hash crossing
    - Embedding dimension heuristics
    - Sequence cropping / padding utilities

Design:
    - All transforms support chunk/batch processing via fit_on_chunks / transform_chunks
    - Intermediate states (vocab, stats, aggregates) are serializable for caching
    - Compatible with preprocessor.MaterializedDatasetArtifact as input source
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# Config
# ============================================================================


@dataclass
class FeatureEngineeringConfig:
    """Configuration for offline feature engineering."""
    # Encode settings
    frequency_encode: bool = False
    target_encode: bool = False
    category_encode: bool = True
    hash_crossings: bool = False
    # Numeric normalization
    normalize: bool = False
    normalize_method: str = "minmax"  # minmax / zscore / log1p
    # Output
    output_dir: Optional[str] = None
    cache_features: bool = True
    # Hash crossing
    hash_bucket_size: int = 1_000_000
    crossing_pairs: Optional[List[Tuple[str, str]]] = None
    # Chunking
    chunk_size: int = 100_000
    # Seed
    seed: int = 42


# ============================================================================
# Feature state (serializable, cacheable)
# ============================================================================


@dataclass
class FrequencyMap:
    """Cached frequency map for a column."""
    column: str
    map: Dict[Any, int] = field(default_factory=dict)
    n_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"column": self.column, "map": self.map, "n_total": self.n_total}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FrequencyMap":
        return FrequencyMap(column=d["column"], map=d["map"], n_total=d["n_total"])


@dataclass
class TargetAggregates:
    """Cached target aggregates for a column (target encoding)."""
    column: str
    global_mean: float = 0.0
    category_means: Dict[Any, float] = field(default_factory=dict)
    category_counts: Dict[Any, int] = field(default_factory=dict)
    smoothing: float = 10.0

    def encode(self, value: Any) -> float:
        """Apply smoothed target encoding: (count*mean + smoothing*global) / (count + smoothing)."""
        cnt = self.category_counts.get(value, 0)
        mean = self.category_means.get(value, self.global_mean)
        return (cnt * mean + self.smoothing * self.global_mean) / (cnt + self.smoothing)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "global_mean": self.global_mean,
            "category_means": self.category_means,
            "category_counts": self.category_counts,
            "smoothing": self.smoothing,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TargetAggregates":
        return TargetAggregates(**d)


@dataclass
class NumericStats:
    """Cached normalization stats for a numeric column."""
    column: str
    min_val: float = 0.0
    max_val: float = 1.0
    mean_val: float = 0.0
    std_val: float = 1.0

    def normalize_minmax(self, value: float) -> float:
        rng = self.max_val - self.min_val
        if rng == 0:
            return 0.0
        return (value - self.min_val) / rng

    def normalize_zscore(self, value: float) -> float:
        if self.std_val == 0:
            return 0.0
        return (value - self.mean_val) / self.std_val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "mean_val": self.mean_val,
            "std_val": self.std_val,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NumericStats":
        return NumericStats(**d)


@dataclass
class CategoryVocab:
    """Cached category vocabulary."""
    column: str
    mapping: Dict[Any, int] = field(default_factory=dict)
    reverse_mapping: Dict[int, Any] = field(default_factory=dict)
    unk_id: int = 0

    def encode(self, value: Any) -> int:
        return self.mapping.get(value, self.unk_id)

    def decode(self, idx: int) -> Any:
        return self.reverse_mapping.get(idx, None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "mapping": self.mapping,
            "reverse_mapping": {str(k): v for k, v in self.reverse_mapping.items()},
            "unk_id": self.unk_id,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CategoryVocab":
        reverse = {int(k): v for k, v in d.get("reverse_mapping", {}).items()}
        return CategoryVocab(
            column=d["column"],
            mapping=d["mapping"],
            reverse_mapping=reverse,
            unk_id=d["unk_id"],
        )


@dataclass
class FeatureManifest:
    """Complete feature engineering state, serializable for caching."""
    frequency_maps: Dict[str, FrequencyMap] = field(default_factory=dict)
    target_aggregates: Dict[str, TargetAggregates] = field(default_factory=dict)
    numeric_stats: Dict[str, NumericStats] = field(default_factory=dict)
    category_vocabs: Dict[str, CategoryVocab] = field(default_factory=dict)
    # Metadata
    columns: List[str] = field(default_factory=list)
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frequency_maps": {k: v.to_dict() for k, v in self.frequency_maps.items()},
            "target_aggregates": {k: v.to_dict() for k, v in self.target_aggregates.items()},
            "numeric_stats": {k: v.to_dict() for k, v in self.numeric_stats.items()},
            "category_vocabs": {k: v.to_dict() for k, v in self.category_vocabs.items()},
            "columns": self.columns,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FeatureManifest":
        return FeatureManifest(
            frequency_maps={k: FrequencyMap.from_dict(v) for k, v in d.get("frequency_maps", {}).items()},
            target_aggregates={k: TargetAggregates.from_dict(v) for k, v in d.get("target_aggregates", {}).items()},
            numeric_stats={k: NumericStats.from_dict(v) for k, v in d.get("numeric_stats", {}).items()},
            category_vocabs={k: CategoryVocab.from_dict(v) for k, v in d.get("category_vocabs", {}).items()},
            columns=d.get("columns", []),
            numeric_cols=d.get("numeric_cols", []),
            categorical_cols=d.get("categorical_cols", []),
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @staticmethod
    def load(path: str) -> Optional["FeatureManifest"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            return FeatureManifest.from_dict(json.loads(p.read_text()))
        except Exception as e:
            logger.warning("Failed to load feature manifest from %s: %s", path, e)
            return None


# ============================================================================
# Fit-on-chunks: aggregate statistics over large data
# ============================================================================


class ChunkFeatureEngineer:
    """Feature engineer that fits on chunks and transforms incrementally.

    Usage:
        engineer = ChunkFeatureEngineer(config)
        engineer.fit_on_chunks(chunk_iterator)
        result = engineer.transform_chunk(single_chunk)
    """

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        self.config = config
        self.manifest = FeatureManifest()

    # ---- fit ----------------------------------------------------------------

    def fit_on_chunks(
        self,
        chunks: Any,  # iterable of pd.DataFrame
        label_col: Optional[str] = None,
    ) -> FeatureManifest:
        """Aggregate statistics over chunks without loading full data."""
        # Accumulators
        freq_acc: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        target_acc: Dict[str, Dict[Any, Tuple[float, int]]] = defaultdict(
            lambda: {"sum": defaultdict(float), "count": defaultdict(int)}
        )
        num_acc: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "sum_sq": 0.0, "count": 0}
        )
        cat_acc: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        global_label_sum = 0.0
        global_label_count = 0

        for chunk_idx, chunk in enumerate(chunks):
            logger.debug("Fitting chunk %d: %d rows", chunk_idx, len(chunk))

            for col in chunk.columns:
                series = chunk[col]

                # Frequency counts
                if self.config.frequency_encode:
                    vc = series.value_counts()
                    for val, cnt in vc.items():
                        freq_acc[col][val] += cnt

                # Target encoding
                if self.config.target_encode and label_col and label_col in chunk.columns:
                    labels = chunk[label_col]
                    for val in series.unique():
                        mask = series == val
                        target_acc[col]["sum"][val] += labels[mask].sum()
                        target_acc[col]["count"][val] += mask.sum()

                # Numeric stats
                if self.config.normalize and pd.api.types.is_numeric_dtype(series):
                    s = series.dropna()
                    if len(s) > 0:
                        na = num_acc[col]
                        na["min"] = min(na["min"], s.min())
                        na["max"] = max(na["max"], s.max())
                        na["sum"] += s.sum()
                        na["sum_sq"] += (s * s).sum()
                        na["count"] += len(s)

                # Category vocab
                if self.config.category_encode and not pd.api.types.is_numeric_dtype(series):
                    for val in series.dropna().unique():
                        cat_acc[col][val] += 1

            # Global label stats
            if label_col and label_col in chunk.columns:
                global_label_sum += chunk[label_col].sum()
                global_label_count += len(chunk)

        # Build manifest from accumulators
        global_mean = global_label_sum / max(global_label_count, 1)

        for col in freq_acc:
            self.manifest.frequency_maps[col] = FrequencyMap(
                column=col,
                map=dict(freq_acc[col]),
                n_total=sum(freq_acc[col].values()),
            )

        for col in target_acc:
            ta = target_acc[col]
            cat_means = {}
            cat_counts = {}
            for val in ta["sum"]:
                cnt = ta["count"][val]
                sm = ta["sum"][val]
                cat_means[val] = sm / max(cnt, 1)
                cat_counts[val] = cnt
            self.manifest.target_aggregates[col] = TargetAggregates(
                column=col,
                global_mean=global_mean,
                category_means=cat_means,
                category_counts=cat_counts,
            )

        for col in num_acc:
            na = num_acc[col]
            cnt = na["count"]
            mean = na["sum"] / max(cnt, 1)
            variance = (na["sum_sq"] / max(cnt, 1)) - (mean * mean)
            self.manifest.numeric_stats[col] = NumericStats(
                column=col,
                min_val=na["min"],
                max_val=na["max"],
                mean_val=mean,
                std_val=max(np.sqrt(abs(variance)), 1e-8),
            )
            self.manifest.numeric_cols.append(col)

        for col in cat_acc:
            vocab = sorted(cat_acc[col].items(), key=lambda x: -x[1])
            mapping: Dict[Any, int] = {}
            reverse: Dict[int, Any] = {}
            for idx, (val, _) in enumerate(vocab, start=1):  # 0 = UNK
                mapping[val] = idx
                reverse[idx] = val
            self.manifest.category_vocabs[col] = CategoryVocab(
                column=col,
                mapping=mapping,
                reverse_mapping=reverse,
            )
            self.manifest.categorical_cols.append(col)

        self.manifest.columns = list(
            set(list(freq_acc.keys()) + list(target_acc.keys()) +
                list(num_acc.keys()) + list(cat_acc.keys()))
        )

        return self.manifest

    # ---- transform ----------------------------------------------------------

    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Apply encoded transforms to a single chunk."""
        result = chunk.copy()

        for col in result.columns:
            if col not in self.manifest.columns:
                continue

            # Frequency encode
            if col in self.manifest.frequency_maps:
                fm = self.manifest.frequency_maps[col]
                result[f"{col}_freq"] = result[col].map(fm.map).fillna(0)

            # Target encode
            if col in self.manifest.target_aggregates:
                ta = self.manifest.target_aggregates[col]
                result[f"{col}_target"] = result[col].apply(ta.encode)

            # Category encode
            if col in self.manifest.category_vocabs:
                cv = self.manifest.category_vocabs[col]
                result[f"{col}_cat"] = result[col].apply(cv.encode)

            # Numeric normalize
            if col in self.manifest.numeric_stats and self.config.normalize:
                ns = self.manifest.numeric_stats[col]
                if self.config.normalize_method == "minmax":
                    result[col] = result[col].apply(ns.normalize_minmax)
                elif self.config.normalize_method == "zscore":
                    result[col] = result[col].apply(ns.normalize_zscore)
                elif self.config.normalize_method == "log1p":
                    result[col] = np.log1p(result[col].clip(lower=0))

        return result

    def fit_transform_materialized(
        self,
        materialized_path: str,
        label_col: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, FeatureManifest]:
        """Fit and transform a materialized columnar dataset in chunks.

        Returns (transformed DataFrame, feature manifest).
        """
        chunk_sz = chunk_size or self.config.chunk_size

        # Phase 1: fit on chunks
        logger.info("Fitting features on %s ...", materialized_path)
        if materialized_path.lower().endswith(".parquet"):
            chunks = pd.read_parquet(materialized_path,  # type: ignore[call-overload]
                columns=self.manifest.columns if self.manifest.columns else None,
            )
            # Re-chunk
            n_rows = len(chunks)
            chunk_iter = (chunks.iloc[i:i+chunk_sz] for i in range(0, n_rows, chunk_sz))
        else:
            chunk_iter = pd.read_csv(materialized_path, chunksize=chunk_sz)

        self.fit_on_chunks(chunk_iter, label_col=label_col)

        # Phase 2: transform
        logger.info("Transforming features on %s ...", materialized_path)
        if materialized_path.lower().endswith(".parquet"):
            all_data = pd.read_parquet(materialized_path)
            all_chunks = [all_data.iloc[i:i+chunk_sz].copy() for i in range(0, len(all_data), chunk_sz)]
        else:
            all_chunks = [c for c in pd.read_csv(materialized_path, chunksize=chunk_sz)]

        transformed = [self.transform_chunk(c) for c in all_chunks]
        result = pd.concat(transformed, ignore_index=True)

        if self.config.cache_features and self.config.output_dir:
            cm_path = Path(self.config.output_dir) / "feature_manifest.json"
            self.manifest.save(str(cm_path))
            logger.info("Feature manifest saved to %s", cm_path)

        return result, self.manifest


# ============================================================================
# Standalone transforms
# ============================================================================


def hash_crossing(
    series_a: pd.Series,
    series_b: pd.Series,
    bucket_size: int = 1_000_000,
) -> pd.Series:
    """Create hashed feature crossing between two columns."""
    combined = series_a.astype(str) + "_X_" + series_b.astype(str)
    hashes = combined.apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % bucket_size)
    return hashes


def embedding_dim_heuristic(
    n_categories: int,
    method: str = "google",
) -> int:
    """Compute a reasonable embedding dimension given vocabulary size.

    Args:
        n_categories: Number of unique categories.
        method: Heuristic method.
            - "google": 4th root rule, dim ≈ n^(1/4)*2
            - "fastai": max(1, int(n^0.25 * 1.6), min(50, n//2))
            - "rule_of_thumb": min(50, int(n**0.5))

    Returns:
        Suggested embedding dimension.
    """
    if n_categories <= 1:
        return 1
    if method == "google":
        return max(1, min(600, int(round(n_categories ** 0.25 * 2))))
    elif method == "fastai":
        return max(1, min(50, int(n_categories ** 0.25 * 1.6), n_categories // 2))
    elif method == "rule_of_thumb":
        return min(50, int(n_categories ** 0.5))
    return min(50, n_categories - 1)


def sequence_pad_truncate(
    sequence: List[Any],
    max_len: int,
    pad_value: Any = 0,
    truncate_from: str = "right",
) -> List[Any]:
    """Pad or truncate a sequence to max_len.

    Args:
        sequence: Input sequence.
        max_len: Target length.
        pad_value: Value used for padding.
        truncate_from: 'left' or 'right'.

    Returns:
        Padded/truncated sequence of length max_len.
    """
    if len(sequence) >= max_len:
        if truncate_from == "left":
            return sequence[-max_len:]
        return sequence[:max_len]
    pad_len = max_len - len(sequence)
    if truncate_from == "left":
        return [pad_value] * pad_len + list(sequence)
    return list(sequence) + [pad_value] * pad_len


# ============================================================================
# Vector Feature Engineering (for pgvector / embedding-based features)
# ============================================================================


@dataclass
class VectorFeatureConfig:
    """Configuration for vector feature processing."""
    vector_dim: int = 128
    normalize: bool = True
    distance_metric: str = "cosine"  # cosine, l2, inner_product
    # Similarity computation
    compute_similarity: bool = False
    similarity_top_k: int = 100
    # Dimensionality reduction
    reduce_dim: bool = False
    target_dim: int = 64
    reduction_method: str = "pca"  # pca, random_projection


@dataclass
class VectorStats:
    """Cached statistics for a vector column."""
    column: str
    dim: int = 128
    mean_vector: Optional[np.ndarray] = None
    std_vector: Optional[np.ndarray] = None
    n_vectors: int = 0
    # For PCA
    pca_components: Optional[np.ndarray] = None
    pca_mean: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "dim": self.dim,
            "mean_vector": self.mean_vector.tolist() if self.mean_vector is not None else None,
            "std_vector": self.std_vector.tolist() if self.std_vector is not None else None,
            "n_vectors": self.n_vectors,
            "pca_components": self.pca_components.tolist() if self.pca_components is not None else None,
            "pca_mean": self.pca_mean.tolist() if self.pca_mean is not None else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "VectorStats":
        return VectorStats(
            column=d["column"],
            dim=d["dim"],
            mean_vector=np.array(d["mean_vector"]) if d.get("mean_vector") else None,
            std_vector=np.array(d["std_vector"]) if d.get("std_vector") else None,
            n_vectors=d.get("n_vectors", 0),
            pca_components=np.array(d["pca_components"]) if d.get("pca_components") else None,
            pca_mean=np.array(d["pca_mean"]) if d.get("pca_mean") else None,
        )


class VectorFeatureEngineer:
    """Process vector features for recommendation systems.

    Capabilities:
        - L2 normalization for cosine similarity
        - Vector statistics computation (mean, std)
        - Dimensionality reduction (PCA, random projection)
        - Similarity matrix computation
        - Integration with pgvector output

    Usage:
        engineer = VectorFeatureEngineer(config)
        normalized = engineer.normalize_vectors(vectors)
        stats = engineer.compute_vector_stats(vectors, column="item_embedding")
    """

    def __init__(self, config: VectorFeatureConfig) -> None:
        self.config = config
        self._vector_stats: Dict[str, VectorStats] = {}

    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for cosine similarity.

        Args:
            vectors: Array of shape (n_vectors, dim)

        Returns:
            L2 normalized vectors of same shape
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        return vectors / norms

    def compute_vector_stats(
        self,
        vectors: np.ndarray,
        column: str = "embedding",
    ) -> VectorStats:
        """Compute statistics for a vector column.

        Args:
            vectors: Array of shape (n_vectors, dim)
            column: Column name for caching

        Returns:
            VectorStats with mean, std, and count
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        n_vectors, dim = vectors.shape

        stats = VectorStats(
            column=column,
            dim=dim,
            mean_vector=np.mean(vectors, axis=0),
            std_vector=np.std(vectors, axis=0),
            n_vectors=n_vectors,
        )

        self._vector_stats[column] = stats
        return stats

    def compute_similarity_matrix(
        self,
        query_vectors: np.ndarray,
        item_vectors: np.ndarray,
        metric: str = "cosine",
        top_k: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute similarity between query and item vectors.

        Args:
            query_vectors: Array of shape (n_queries, dim)
            item_vectors: Array of shape (n_items, dim)
            metric: Distance metric (cosine, l2, inner_product)
            top_k: Number of top similar items to return

        Returns:
            Tuple of (indices, scores) of shape (n_queries, top_k)
        """
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        if item_vectors.ndim == 1:
            item_vectors = item_vectors.reshape(1, -1)

        if metric == "cosine":
            # Normalize for cosine similarity
            q_norm = self.normalize_vectors(query_vectors)
            i_norm = self.normalize_vectors(item_vectors)
            similarity = np.dot(q_norm, i_norm.T)
        elif metric == "l2":
            # Negative L2 distance (higher = more similar)
            similarity = -np.linalg.norm(
                query_vectors[:, np.newaxis] - item_vectors[np.newaxis, :],
                axis=2
            )
        elif metric == "inner_product":
            similarity = np.dot(query_vectors, item_vectors.T)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Get top-k
        top_k = min(top_k, similarity.shape[1])
        indices = np.argsort(-similarity, axis=1)[:, :top_k]
        scores = np.take_along_axis(similarity, indices, axis=1)

        return indices, scores

    def reduce_dimensionality(
        self,
        vectors: np.ndarray,
        target_dim: int = 64,
        method: str = "pca",
        column: str = "embedding",
    ) -> np.ndarray:
        """Reduce vector dimensionality.

        Args:
            vectors: Array of shape (n_vectors, dim)
            target_dim: Target dimension
            method: Reduction method (pca, random_projection)
            column: Column name for caching PCA components

        Returns:
            Reduced vectors of shape (n_vectors, target_dim)
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        n_vectors, dim = vectors.shape
        target_dim = min(target_dim, dim)

        if method == "pca":
            try:
                from sklearn.decomposition import PCA

                pca = PCA(n_components=target_dim)
                reduced = pca.fit_transform(vectors)

                # Cache PCA components for later use
                if column in self._vector_stats:
                    self._vector_stats[column].pca_components = pca.components_
                    self._vector_stats[column].pca_mean = pca.mean_

                return reduced
            except ImportError:
                logger.warning("sklearn not available, falling back to random projection")
                method = "random_projection"

        if method == "random_projection":
            # Gaussian random projection
            np.random.seed(42)
            projection = np.random.randn(dim, target_dim) / np.sqrt(target_dim)
            return np.dot(vectors, projection)

        raise ValueError(f"Unknown reduction method: {method}")

    def apply_pca_transform(
        self,
        vectors: np.ndarray,
        column: str = "embedding",
    ) -> np.ndarray:
        """Apply cached PCA transform to new vectors.

        Args:
            vectors: Array of shape (n_vectors, dim)
            column: Column name with cached PCA components

        Returns:
            Transformed vectors
        """
        if column not in self._vector_stats:
            raise ValueError(f"No cached PCA for column: {column}")

        stats = self._vector_stats[column]
        if stats.pca_components is None:
            raise ValueError(f"PCA not fitted for column: {column}")

        centered = vectors - stats.pca_mean
        return np.dot(centered, stats.pca_components.T)

    def decode_base64_vectors(
        self,
        base64_strings: List[str],
        dim: int,
    ) -> np.ndarray:
        """Decode base64-encoded vectors from PostgreSQL export.

        Args:
            base64_strings: List of base64-encoded vector strings
            dim: Expected vector dimension

        Returns:
            Array of shape (n_vectors, dim)
        """
        import base64

        vectors = []
        for b64_str in base64_strings:
            try:
                raw_bytes = base64.b64decode(b64_str)
                vec = np.frombuffer(raw_bytes, dtype=np.float32)
                if len(vec) == dim:
                    vectors.append(vec)
                else:
                    # Pad or truncate
                    vec = np.pad(vec, (0, dim - len(vec))) if len(vec) < dim else vec[:dim]
                    vectors.append(vec)
            except Exception as e:
                logger.warning("Failed to decode vector: %s", e)
                vectors.append(np.zeros(dim, dtype=np.float32))

        return np.array(vectors, dtype=np.float32)

    def encode_vectors_base64(
        self,
        vectors: np.ndarray,
    ) -> List[str]:
        """Encode vectors as base64 strings for PostgreSQL import.

        Args:
            vectors: Array of shape (n_vectors, dim)

        Returns:
            List of base64-encoded strings
        """
        import base64

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        result = []
        for vec in vectors:
            vec_f32 = vec.astype(np.float32)
            b64_str = base64.b64encode(vec_f32.tobytes()).decode("ascii")
            result.append(b64_str)

        return result

    def get_vector_stats(self, column: str) -> Optional[VectorStats]:
        """Get cached vector statistics for a column."""
        return self._vector_stats.get(column)

    def save_vector_stats(self, path: str) -> None:
        """Save all vector statistics to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            col: stats.to_dict()
            for col, stats in self._vector_stats.items()
        }
        p.write_text(json.dumps(data, indent=2))

    def load_vector_stats(self, path: str) -> None:
        """Load vector statistics from a JSON file."""
        p = Path(path)
        if not p.exists():
            return

        data = json.loads(p.read_text())
        for col, stats_dict in data.items():
            self._vector_stats[col] = VectorStats.from_dict(stats_dict)


def compute_user_item_similarity(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    top_k: int = 100,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for user-item similarity computation.

    Args:
        user_vectors: User embeddings of shape (n_users, dim)
        item_vectors: Item embeddings of shape (n_items, dim)
        top_k: Number of top items per user
        normalize: Whether to L2 normalize vectors first

    Returns:
        Tuple of (item_indices, similarity_scores) of shape (n_users, top_k)
    """
    config = VectorFeatureConfig(
        normalize=normalize,
        similarity_top_k=top_k,
    )
    engineer = VectorFeatureEngineer(config)
    return engineer.compute_similarity_matrix(
        user_vectors, item_vectors,
        metric="cosine", top_k=top_k
    )
