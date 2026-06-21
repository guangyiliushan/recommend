"""Negative sampling strategies for implicit feedback.

Capabilities:
    - Uniform random sampling
    - Popularity-based sampling (weighted by item frequency)
    - In-batch negatives (use other samples in the same batch as negatives)
    - Hard negative mining (placeholder for external candidate pool)
    - Mixed sampling (combine multiple strategies with weights)

Design for large-scale datasets:
    - Item pool is loaded lazily or from cached columnar stats
    - Sampling does not require loading all candidates into memory
    - Reproducible with seeded RNG
    - Configurable per-split (train/val/test) strategies
    - Sampling state (item frequencies, pools) is cacheable for reuse
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SamplingStrategy(str, Enum):
    """Negative sampling strategy type."""
    UNIFORM = "uniform"
    POPULARITY = "popularity"
    IN_BATCH = "in_batch"
    HARD = "hard"
    MIXED = "mixed"


@dataclass
class NegativeSamplingConfig:
    """Configuration for negative sampling."""
    strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    # Number of negatives per positive sample
    num_negatives: int = 4
    # Mixed strategy weights (must sum to 1.0)
    mix_weights: Optional[Dict[SamplingStrategy, float]] = None
    # Item pool source: 'auto' = infer from data, 'cached' = load from cache file
    item_pool_source: str = "auto"
    # Cache path for item pool / popularity stats
    cache_path: Optional[str] = None
    # Max items to hold in memory for pool
    max_pool_size: int = 10_000_000
    # Seed for reproducibility
    seed: int = 42
    # Exclude positive items from negative sampling (True for most scenarios)
    exclude_positives: bool = True
    # For popularity-based: power to raise frequencies to (1.0 = proportional, 0.75 = tempered)
    popularity_power: float = 0.75


@dataclass
class ItemPoolStats:
    """Cached item pool statistics for efficient sampling."""
    item_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    frequencies: Optional[np.ndarray] = None  # popularity counts
    n_total_items: int = 0
    # Probability distribution for weighted sampling
    _sampling_probs: Optional[np.ndarray] = None

    @property
    def sampling_probs(self) -> np.ndarray:
        """Compute or return cached sampling probabilities."""
        if self._sampling_probs is not None:
            return self._sampling_probs
        if self.frequencies is not None and len(self.frequencies) > 0:
            raw = self.frequencies.astype(np.float64)
            raw /= raw.sum()
            self._sampling_probs = raw
        else:
            n = len(self.item_ids)
            self._sampling_probs = np.ones(n) / max(n, 1)
        return self._sampling_probs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_ids": self.item_ids.tolist() if len(self.item_ids) < 10_000 else [],
            "n_total_items": self.n_total_items,
            "frequencies": self.frequencies.tolist() if self.frequencies is not None and len(self.frequencies) < 10_000 else [],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ItemPoolStats":
        item_ids = np.array(d.get("item_ids", []), dtype=np.int64)
        freqs = np.array(d.get("frequencies", []), dtype=np.float64) if d.get("frequencies") else None
        return ItemPoolStats(
            item_ids=item_ids,
            frequencies=freqs,
            n_total_items=d.get("n_total_items", len(item_ids)),
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(p),
            item_ids=self.item_ids,
            frequencies=self.frequencies if self.frequencies is not None else np.array([]),
            n_total_items=self.n_total_items,
        )

    @staticmethod
    def load(path: str) -> Optional["ItemPoolStats"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = np.load(str(p), allow_pickle=True)
            freqs_arr = data.get("frequencies")
            freqs = freqs_arr if freqs_arr is not None and len(freqs_arr) > 0 else None
            return ItemPoolStats(
                item_ids=data["item_ids"],
                frequencies=freqs,
                n_total_items=int(data.get("n_total_items", len(data["item_ids"]))),
            )
        except Exception as e:
            logger.warning("Failed to load item pool stats from %s: %s", path, e)
            return None


class NegativeSampler:
    """Memory-efficient negative sampler for large-scale implicit feedback.

    Usage:
        sampler = NegativeSampler(config)
        sampler.fit(item_df, item_id_col="item_id", freq_col=None)
        negatives = sampler.sample(positive_items, n_per_positive=4)
    """

    def __init__(self, config: NegativeSamplingConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._pool: Optional[ItemPoolStats] = None

    # ---- fit ----------------------------------------------------------------

    def fit(
        self,
        item_ids: Optional[np.ndarray] = None,
        frequencies: Optional[np.ndarray] = None,
        item_df: Optional[Any] = None,  # pd.DataFrame
        item_id_col: str = "item_id",
        freq_col: Optional[str] = None,
    ) -> "NegativeSampler":
        """Build the item pool for sampling.

        Can be called with:
        - item_ids + frequencies arrays directly
        - A pandas DataFrame with item_id_col and optional freq_col
        - Neither → will lazily infer from .sample() calls
        """
        # Try loading from cache first
        if self.config.cache_path:
            cached = ItemPoolStats.load(self.config.cache_path)
            if cached is not None:
                self._pool = cached
                logger.info("Loaded item pool from cache: %d items", self._pool.n_total_items)
                return self

        if item_ids is not None:
            self._pool = ItemPoolStats(
                item_ids=np.asarray(item_ids, dtype=np.int64),
                frequencies=np.asarray(frequencies, dtype=np.float64) if frequencies is not None else None,
                n_total_items=len(item_ids),
            )
        elif item_df is not None:
            ids = item_df[item_id_col].unique()
            freqs = None
            if freq_col and freq_col in item_df.columns:
                freqs = item_df.groupby(item_id_col)[freq_col].sum().reindex(ids, fill_value=1).values
            self._pool = ItemPoolStats(
                item_ids=np.asarray(ids, dtype=np.int64),
                frequencies=np.asarray(freqs, dtype=np.float64) if freqs is not None else None,
                n_total_items=len(ids),
            )

        if self._pool is not None and self.config.cache_path:
            self._pool.save(self.config.cache_path)
            logger.info("Saved item pool to cache: %d items", self._pool.n_total_items)

        return self

    @property
    def pool(self) -> ItemPoolStats:
        if self._pool is None:
            raise RuntimeError(
                "Item pool not initialized. Call .fit() first with item_ids or item_df."
            )
        return self._pool

    @property
    def n_items(self) -> int:
        return self.pool.n_total_items if self._pool else 0

    # ---- sample -------------------------------------------------------------

    def sample(
        self,
        positive_items: np.ndarray,
        n_per_positive: Optional[int] = None,
        exclude: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sample negative items.

        Args:
            positive_items: Array of positive item IDs, shape (n_samples,).
            n_per_positive: Number of negatives per positive. Defaults to config.
            exclude: Items to exclude from sampling (e.g., positive items per user).

        Returns:
            Array of negative item IDs, shape (n_samples * n_per_positive,).
        """
        n_neg = n_per_positive or self.config.num_negatives
        n_pos = len(positive_items)
        total_needed = n_pos * n_neg

        strategy = self.config.strategy

        if strategy == SamplingStrategy.UNIFORM:
            sampled = self._sample_uniform(total_needed, exclude)
        elif strategy == SamplingStrategy.POPULARITY:
            sampled = self._sample_popularity(total_needed, exclude)
        elif strategy == SamplingStrategy.IN_BATCH:
            sampled = self._sample_in_batch(positive_items, n_neg)
        elif strategy == SamplingStrategy.MIXED:
            sampled = self._sample_mixed(total_needed, exclude)
        else:
            # Default to uniform
            sampled = self._sample_uniform(total_needed, exclude)

        return sampled

    def sample_per_user(
        self,
        user_positive_items: Dict[int, List[int]],
        n_per_positive: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        """Sample negatives per user, excluding their own positives.

        Args:
            user_positive_items: Dict of user_id → list of positive item_ids.
            n_per_positive: Negatives per positive.

        Returns:
            Dict of user_id → array of negative item_ids.
        """
        n_neg = n_per_positive or self.config.num_negatives
        result: Dict[int, np.ndarray] = {}

        for uid, pos_items in user_positive_items.items():
            exclude_set = set(pos_items) if self.config.exclude_positives else set()
            exclude_arr = np.array(list(exclude_set), dtype=np.int64) if exclude_set else None
            result[uid] = self.sample(
                np.array(pos_items, dtype=np.int64),
                n_per_positive=n_neg,
                exclude=exclude_arr,
            )

        return result

    # ---- internal strategies -----------------------------------------------

    def _sample_uniform(self, n: int, exclude: Optional[np.ndarray] = None) -> np.ndarray:
        """Uniform random sampling from the item pool."""
        pool_ids = self.pool.item_ids
        if len(pool_ids) == 0:
            return np.array([], dtype=np.int64)

        # Exclude items if needed
        if exclude is not None and len(exclude) > 0:
            mask = ~np.isin(pool_ids, exclude)
            candidates = pool_ids[mask]
            if len(candidates) == 0:
                candidates = pool_ids  # fallback
        else:
            candidates = pool_ids

        indices = self._rng.integers(0, len(candidates), size=n)
        return candidates[indices]

    def _sample_popularity(self, n: int, exclude: Optional[np.ndarray] = None) -> np.ndarray:
        """Popularity-weighted sampling."""
        pool_ids = self.pool.item_ids
        probs = self.pool.sampling_probs.copy()

        if self.config.popularity_power != 1.0:
            probs = probs ** self.config.popularity_power
            probs /= probs.sum()

        if exclude is not None and len(exclude) > 0:
            mask = ~np.isin(pool_ids, exclude)
            pool_ids = pool_ids[mask]
            probs = probs[mask]
            if len(pool_ids) == 0:
                pool_ids = self.pool.item_ids
                probs = self.pool.sampling_probs.copy()
            probs /= probs.sum()

        indices = self._rng.choice(len(pool_ids), size=n, p=probs, replace=True)
        return pool_ids[indices]

    def _sample_in_batch(self, positive_items: np.ndarray, n_per_positive: int) -> np.ndarray:
        """Use other samples in the batch as negatives (in-batch negatives).

        For each positive item, randomly sample other items in the batch as negatives.
        This is efficient for large-scale training where the pool is naturally large.
        """
        n_pos = len(positive_items)
        total = n_pos * n_per_positive
        if n_pos < 2:
            return np.array([], dtype=np.int64)

        # For each position, pick random other items from the batch
        all_indices = np.arange(n_pos)
        neg_indices = np.empty(total, dtype=np.int64)
        pos = 0
        for i in range(n_pos):
            others = all_indices[all_indices != i]
            if len(others) == 0:
                others = all_indices
            chosen = self._rng.choice(others, size=n_per_positive, replace=len(others) < n_per_positive)
            neg_indices[pos:pos + n_per_positive] = positive_items[chosen]
            pos += n_per_positive

        return neg_indices

    def _sample_mixed(self, n: int, exclude: Optional[np.ndarray] = None) -> np.ndarray:
        """Mix multiple strategies according to weights."""
        if self.config.mix_weights is None:
            # Default: 50% uniform, 50% popularity
            weights = {SamplingStrategy.UNIFORM: 0.5, SamplingStrategy.POPULARITY: 0.5}
        else:
            weights = self.config.mix_weights

        total_weight = sum(weights.values())
        normalized = {k: v / total_weight for k, v in weights.items()}

        result_parts: List[np.ndarray] = []
        remaining = n

        strategies = list(normalized.keys())
        for i, strat in enumerate(strategies):
            count = remaining if i == len(strategies) - 1 else int(n * normalized[strat])
            remaining -= count

            if count <= 0:
                continue

            if strat == SamplingStrategy.UNIFORM:
                result_parts.append(self._sample_uniform(count, exclude))
            elif strat == SamplingStrategy.POPULARITY:
                result_parts.append(self._sample_popularity(count, exclude))
            elif strat == SamplingStrategy.IN_BATCH:
                result_parts.append(self._sample_uniform(count, exclude))  # fallback
            else:
                result_parts.append(self._sample_uniform(count, exclude))

        if not result_parts:
            return np.array([], dtype=np.int64)

        result = np.concatenate(result_parts)
        self._rng.shuffle(result)  # mix strategies
        return result

    # ---- utility ------------------------------------------------------------

    def get_item_frequencies(self) -> Optional[np.ndarray]:
        """Return item frequency array (useful for popularity-based approaches)."""
        return self.pool.frequencies

    def get_item_universe(self) -> np.ndarray:
        """Return all unique item IDs in the pool."""
        return self.pool.item_ids


# ============================================================================
# Factory helpers
# ============================================================================


def create_sampler(
    strategy: str = "uniform",
    num_negatives: int = 4,
    seed: int = 42,
    **kwargs: Any,
) -> NegativeSampler:
    """Create a negative sampler from a strategy name and config.

    Args:
        strategy: One of 'uniform', 'popularity', 'in_batch', 'hard', 'mixed'.
        num_negatives: Number of negatives per positive.
        seed: Random seed.
        **kwargs: Additional config passed to NegativeSamplingConfig.

    Returns:
        Configured NegativeSampler.
    """
    try:
        strat = SamplingStrategy(strategy.lower())
    except ValueError:
        logger.warning("Unknown strategy '%s', falling back to uniform", strategy)
        strat = SamplingStrategy.UNIFORM

    config = NegativeSamplingConfig(
        strategy=strat,
        num_negatives=num_negatives,
        seed=seed,
        **kwargs,
    )
    return NegativeSampler(config)


# ============================================================================
# PostgreSQL-based negative sampling (TABLESAMPLE)
# ============================================================================


class PostgresNegativeSampler:
    """Negative sampler that samples directly from PostgreSQL.

    Uses PostgreSQL's TABLESAMPLE for efficient random sampling without
    loading the entire item pool into memory.

    Features:
        - TABLESAMPLE BERNOULLI: True random sampling
        - TABLESAMPLE SYSTEM: Block-level sampling (faster, less random)
        - Popularity-weighted sampling via ORDER BY RANDOM() with weights
        - Vector-based hard negative mining via pgvector similarity

    Usage:
        sampler = PostgresNegativeSampler(connection_string, config)
        negatives = sampler.sample(n_samples=1000)
    """

    def __init__(
        self,
        connection_string: str,
        config: NegativeSamplingConfig,
        table: str = "items",
        item_id_col: str = "item_id",
        popularity_col: Optional[str] = None,
    ) -> None:
        self.connection_string = connection_string
        self.config = config
        self.table = table
        self.item_id_col = item_id_col
        self.popularity_col = popularity_col
        self._rng = np.random.default_rng(config.seed)
        self._total_items: Optional[int] = None

    def _get_connection(self) -> Any:
        """Get PostgreSQL connection."""
        try:
            import psycopg
            return psycopg.connect(self.connection_string)
        except ImportError as err:
            raise ImportError(
                "psycopg is required. Install with: pip install psycopg[binary]  or  uv sync --extra db"
            ) from err

    def get_total_items(self) -> int:
        """Get total number of items in the table."""
        if self._total_items is not None:
            return self._total_items

        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            self._total_items = cur.fetchone()[0]  # type: ignore[index]
        return self._total_items

    def sample_tablesample(
        self,
        n_samples: int,
        method: str = "bernoulli",
        exclude: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sample using PostgreSQL TABLESAMPLE.

        Args:
            n_samples: Number of samples to return
            method: 'bernoulli' (true random) or 'system' (block-level, faster)
            exclude: Items to exclude from sampling

        Returns:
            Array of sampled item IDs
        """
        total = self.get_total_items()
        if total == 0:
            return np.array([], dtype=np.int64)

        # Calculate percentage for TABLESAMPLE
        sample_pct = min(100.0, (n_samples / total) * 100 * 2)  # 2x buffer

        with self._get_connection() as conn, conn.cursor() as cur:
            # Build query
            if exclude is not None and len(exclude) > 0:
                exclude_list = ",".join(str(x) for x in exclude[:10000])  # Limit for SQL
                query = f"""
                        SELECT {self.item_id_col} FROM {self.table}
                        WHERE {self.item_id_col} NOT IN ({exclude_list})
                        TABLESAMPLE {method.upper()}({sample_pct:.4f})
                        LIMIT {n_samples}
                    """
            else:
                query = f"""
                        SELECT {self.item_id_col} FROM {self.table}
                        TABLESAMPLE {method.upper()}({sample_pct:.4f})
                        LIMIT {n_samples}
                    """

            cur.execute(query)
            rows = cur.fetchall()
            return np.array([r[0] for r in rows], dtype=np.int64)

    def sample_popularity(
        self,
        n_samples: int,
        exclude: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sample items weighted by popularity.

        Uses ORDER BY with weighted random selection.

        Args:
            n_samples: Number of samples
            exclude: Items to exclude

        Returns:
            Array of sampled item IDs
        """
        if self.popularity_col is None:
            logger.warning("No popularity column set, falling back to uniform sampling")
            return self.sample_tablesample(n_samples, exclude=exclude)

        power = self.config.popularity_power

        with self._get_connection() as conn, conn.cursor() as cur:
            # Weighted random sampling using POWER(popularity, power) * RANDOM()
            if exclude is not None and len(exclude) > 0:
                exclude_list = ",".join(str(x) for x in exclude[:10000])
                query = f"""
                        SELECT {self.item_id_col}
                        FROM {self.table}
                        WHERE {self.item_id_col} NOT IN ({exclude_list})
                        ORDER BY POWER({self.popularity_col}, {power}) * RANDOM() DESC
                        LIMIT {n_samples}
                    """
            else:
                query = f"""
                        SELECT {self.item_id_col}
                        FROM {self.table}
                        ORDER BY POWER({self.popularity_col}, {power}) * RANDOM() DESC
                        LIMIT {n_samples}
                    """

            cur.execute(query)
            rows = cur.fetchall()
            return np.array([r[0] for r in rows], dtype=np.int64)

    def sample_hard_negatives(
        self,
        query_item_id: int,
        vector_column: str = "embedding",
        n_samples: int = 100,
        distance_metric: str = "cosine",
        similarity_range: Tuple[float, float] = (0.3, 0.7),
    ) -> np.ndarray:
        """Sample hard negatives using vector similarity.

        Hard negatives are items that are similar but not too similar to the query item.

        Args:
            query_item_id: The item to find hard negatives for
            vector_column: Name of the vector column
            n_samples: Number of hard negatives to sample
            distance_metric: Distance metric (cosine, l2, inner_product)
            similarity_range: (min_sim, max_sim) range for hard negatives

        Returns:
            Array of hard negative item IDs
        """
        # Distance operators for pgvector
        distance_ops = {
            "cosine": "<=>",
            "l2": "<->",
            "inner_product": "<#>",
        }
        op = distance_ops.get(distance_metric, "<=>")

        with self._get_connection() as conn, conn.cursor() as cur:
            # Get query vector first
            cur.execute(
                f"SELECT {vector_column} FROM {self.table} WHERE {self.item_id_col} = %s",
                (query_item_id,)
            )
            result = cur.fetchone()
            if result is None:
                return np.array([], dtype=np.int64)

            query_vector = result[0]

            # Find items with similarity in the hard negative range
            # For cosine distance: lower = more similar
            # We want items with distance in (1-max_sim, 1-min_sim)
            if distance_metric == "cosine":
                min_dist = 1.0 - similarity_range[1]
                max_dist = 1.0 - similarity_range[0]
            else:
                min_dist = similarity_range[0]
                max_dist = similarity_range[1]

            query = f"""
                    SELECT {self.item_id_col}
                    FROM {self.table}
                    WHERE {self.item_id_col} != %s
                      AND {vector_column} {op} %s BETWEEN %s AND %s
                    ORDER BY RANDOM()
                    LIMIT %s
                """

            cur.execute(query, (query_item_id, query_vector, min_dist, max_dist, n_samples))
            rows = cur.fetchall()
            return np.array([r[0] for r in rows], dtype=np.int64)

    def sample(
        self,
        n_samples: int,
        strategy: Optional[SamplingStrategy] = None,
        exclude: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sample negative items from PostgreSQL.

        Args:
            n_samples: Number of samples
            strategy: Sampling strategy (defaults to config.strategy)
            exclude: Items to exclude

        Returns:
            Array of sampled item IDs
        """
        strat = strategy or self.config.strategy

        if strat == SamplingStrategy.UNIFORM:
            return self.sample_tablesample(n_samples, method="bernoulli", exclude=exclude)
        elif strat == SamplingStrategy.POPULARITY:
            return self.sample_popularity(n_samples, exclude=exclude)
        else:
            return self.sample_tablesample(n_samples, method="bernoulli", exclude=exclude)

    def sample_for_users(
        self,
        user_positive_items: Dict[int, List[int]],
        n_per_positive: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        """Sample negatives per user from PostgreSQL.

        Args:
            user_positive_items: Dict of user_id → list of positive item_ids
            n_per_positive: Negatives per positive

        Returns:
            Dict of user_id → array of negative item_ids
        """
        n_neg = n_per_positive or self.config.num_negatives
        result: Dict[int, np.ndarray] = {}

        for uid, pos_items in user_positive_items.items():
            total_needed = len(pos_items) * n_neg
            exclude_arr = np.array(pos_items, dtype=np.int64) if self.config.exclude_positives else None
            result[uid] = self.sample(total_needed, exclude=exclude_arr)

        return result


def create_postgres_sampler(
    connection_string: str,
    table: str = "items",
    item_id_col: str = "item_id",
    popularity_col: Optional[str] = None,
    strategy: str = "uniform",
    num_negatives: int = 4,
    seed: int = 42,
    **kwargs: Any,
) -> PostgresNegativeSampler:
    """Create a PostgreSQL-based negative sampler.

    Args:
        connection_string: PostgreSQL connection string
        table: Table containing items
        item_id_col: Column name for item IDs
        popularity_col: Optional column for popularity-weighted sampling
        strategy: Sampling strategy
        num_negatives: Number of negatives per positive
        seed: Random seed
        **kwargs: Additional config options

    Returns:
        Configured PostgresNegativeSampler
    """
    try:
        strat = SamplingStrategy(strategy.lower())
    except ValueError:
        logger.warning("Unknown strategy '%s', falling back to uniform", strategy)
        strat = SamplingStrategy.UNIFORM

    config = NegativeSamplingConfig(
        strategy=strat,
        num_negatives=num_negatives,
        seed=seed,
        **kwargs,
    )

    return PostgresNegativeSampler(
        connection_string=connection_string,
        config=config,
        table=table,
        item_id_col=item_id_col,
        popularity_col=popularity_col,
    )
