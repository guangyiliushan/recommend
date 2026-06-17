"""Data pipeline: datasets, preprocessing, feature engineering, negative sampling.

Public API:
    - Dataset adapters (TAAC 2025/2026)
    - Preprocessing pipeline (OfflinePreprocessingPipeline, PreprocessingConfig)
    - Feature engineering (ChunkFeatureEngineer, VectorFeatureEngineer, FeatureManifest)
    - Negative sampling (NegativeSampler, PostgresNegativeSampler)
    - Database backends (PostgresReader, PostgresVectorReader)
    - Backend/format/codec discovery
"""

# ---- Dataset adapters ---------------------------------------------------
# ---- Registry / Discovery -----------------------------------------------
from recsys.data.dataset_registry import (  # noqa: F401
    get_compression_capabilities,
    get_data_backend,
    get_database_backend,
    get_database_reader_class,
    has_database_capability,
    list_compression_codecs,
    list_data_backends,
    list_database_backends,
    list_feature_primitives,
    list_sampling_strategies,
    list_storage_formats,
)
from recsys.data.datasets.taac2025 import (  # noqa: F401
    TAAC2025Dataset,
    TAAC2025Dataset1M,
    TAAC2025Dataset10M,
)
from recsys.data.datasets.taac2026 import (  # noqa: F401
    TAAC2026DataSample,
    TAAC2026Dataset,
    TAAC2026SecondRound,
)

# ---- Feature engineering ------------------------------------------------
from recsys.data.feature_engineering import (  # noqa: F401
    CategoryVocab,
    ChunkFeatureEngineer,
    FeatureEngineeringConfig,
    FeatureManifest,
    FrequencyMap,
    NumericStats,
    TargetAggregates,
    VectorFeatureConfig,
    VectorFeatureEngineer,
    VectorStats,
    compute_user_item_similarity,
    embedding_dim_heuristic,
    hash_crossing,
    sequence_pad_truncate,
)

# ---- Negative sampling --------------------------------------------------
from recsys.data.negative_sampling import (  # noqa: F401
    ItemPoolStats,
    NegativeSampler,
    NegativeSamplingConfig,
    PostgresNegativeSampler,
    SamplingStrategy,
    create_postgres_sampler,
    create_sampler,
)

# ---- Preprocessing pipeline ---------------------------------------------
from recsys.data.preprocessor import (  # noqa: F401
    CachePolicy,
    CacheTier,
    ColumnStatsCache,
    CompressionCodec,
    # Data structures
    DatasetFingerprint,
    ExecutionBackend,
    IncrementalPolicy,
    MaterializedDatasetArtifact,
    # Pipeline
    OfflinePreprocessingPipeline,
    PipelinePhase,
    # Database readers
    PostgresReader,
    PostgresVectorReader,
    PreprocessCheckpoint,
    PreprocessingConfig,
    ResourceLimits,
    StorageConfig,
    # Config / Enums
    StorageFormat,
    # Capability helpers
    is_backend_available,
    is_format_supported,
    list_available_backends,
    # Materialization
    materialize_to_columnar,
    # Readers
    read_chunked_pandas,
    read_chunked_pyarrow,
    read_dask_dataframe,
    read_polars_lazy,
)
