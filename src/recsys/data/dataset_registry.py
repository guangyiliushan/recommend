"""Dataset registration, data backend discovery, and format/codec capabilities.

Registers:
    - all dataset adapters with DATASET_REGISTRY
    - execution backend capability map
    - storage format capability map
    - compression codec capability map
    - database backend capability map (PostgreSQL + pgvector)
    - negative sampling strategy registry
    - feature engineering primitive registry

All registries gracefully handle missing optional dependencies.
"""

# Import registries so that side-effect registration happens.
# Trigger auto-import of all dataset adapters to register them.
# ---- Data backend discovery -------------------------------------------
from typing import Any, Dict, List, Optional

import recsys.data.datasets.taac2025  # noqa: F401
import recsys.data.datasets.taac2026  # noqa: F401
from recsys.core.registry import DATASET_REGISTRY  # noqa: F401
from recsys.data.preprocessor import (
    CompressionCodec,
    ExecutionBackend,
    StorageFormat,
    is_format_supported,
)

# Backend capability registry
_BACKEND_CAPABILITIES: Dict[str, Dict] = {
    ExecutionBackend.PANDAS.value: {
        "chunked_read": True,
        "mmap": True,
        "dtype_downcast": True,
        "category_encode": True,
        "io_formats": [StorageFormat.CSV.value, StorageFormat.PARQUET.value,
                       StorageFormat.FEATHER.value, StorageFormat.ORC.value],
    },
    ExecutionBackend.PYARROW.value: {
        "chunked_read": True,
        "mmap": True,
        "dtype_downcast": False,
        "category_encode": False,
        "io_formats": [StorageFormat.CSV.value, StorageFormat.PARQUET.value,
                       StorageFormat.FEATHER.value, StorageFormat.ORC.value],
    },
    ExecutionBackend.POLARS.value: {
        "chunked_read": True,
        "mmap": False,
        "dtype_downcast": True,
        "category_encode": True,
        "pushdown": True,
        "lazy": True,
        "io_formats": [StorageFormat.CSV.value, StorageFormat.PARQUET.value],
    },
    ExecutionBackend.DASK.value: {
        "chunked_read": True,
        "mmap": False,
        "dtype_downcast": False,
        "distributed": True,
        "io_formats": [StorageFormat.CSV.value, StorageFormat.PARQUET.value],
    },
    ExecutionBackend.MODIN.value: {
        "chunked_read": True,
        "mmap": False,
        "dtype_downcast": True,
        "io_formats": [StorageFormat.CSV.value, StorageFormat.PARQUET.value],
    },
    ExecutionBackend.VAEX.value: {
        "chunked_read": True,
        "mmap": True,
        "dtype_downcast": False,
        "virtual_columns": True,
        "io_formats": [StorageFormat.CSV.value, StorageFormat.PARQUET.value],
    },
}

# Compression codec capability map
_COMPRESSION_CAPABILITIES = {
    CompressionCodec.NONE.value: {"speed_rank": 1, "ratio_rank": 5},
    CompressionCodec.SNAPPY.value: {"speed_rank": 2, "ratio_rank": 3},
    CompressionCodec.ZSTD.value: {"speed_rank": 4, "ratio_rank": 1},
    CompressionCodec.LZ4.value: {"speed_rank": 3, "ratio_rank": 4},
    CompressionCodec.GZIP.value: {"speed_rank": 5, "ratio_rank": 2},
}

# Negative sampling strategy registry
_SAMPLING_STRATEGIES = {
    "uniform": "Uniform random negative sampling",
    "popularity": "Popularity-weighted negative sampling",
    "in_batch": "In-batch negative sampling",
    "hard": "Hard negative mining (requires candidate pool)",
    "mixed": "Mixed strategy with weighted combination",
}

# Feature engineering primitive registry
_FEATURE_PRIMITIVES = {
    "frequency_encode": "Count/frequency encoding",
    "target_encode": "Target encoding with smoothing",
    "category_encode": "Category dictionary encoding",
    "minmax_normalize": "Min-max normalization",
    "zscore_normalize": "Z-score normalization",
    "log1p_normalize": "Log(1+x) normalization",
    "hash_crossing": "Hashed feature crossing",
    "embedding_dim_heuristic": "Embedding dimension heuristic",
    "sequence_pad_truncate": "Sequence pad/truncate",
    "vector_normalize": "L2 vector normalization",
    "vector_similarity": "Vector similarity computation",
    "vector_reduce_dim": "Vector dimensionality reduction (PCA)",
}

# Database backend capability map (PostgreSQL + pgvector)
_DATABASE_BACKENDS = {
    "postgresql": {
        "reader_class": "PostgresReader",
        "capabilities": [
            "copy_export",       # COPY TO STDOUT for fast export
            "server_cursor",     # Server-side cursor for pagination
            "copy_binary",       # Binary format export
            "parallel_export",   # Parallel partition export
            "partition_aware",   # Partition table support
        ],
        "description": "PostgreSQL reader with COPY and server-side cursor",
    },
    "postgres_vector": {
        "reader_class": "PostgresVectorReader",
        "capabilities": [
            "copy_export",
            "server_cursor",
            "copy_binary",
            "parallel_export",
            "partition_aware",
            "vector_search",     # pgvector similarity search
            "vector_index",      # HNSW/IVFFlat index support
            "hard_negatives",    # Vector-based hard negative mining
        ],
        "description": "PostgreSQL + pgvector for vector similarity search",
    },
}


def list_data_backends() -> List[str]:
    """List all available execution backends."""
    from recsys.data.preprocessor import list_available_backends
    return list_available_backends()


def get_data_backend(name: str) -> Optional[Dict]:
    """Get capability info for a data backend."""
    return _BACKEND_CAPABILITIES.get(name)


def list_storage_formats() -> List[str]:
    """List all supported storage formats."""
    return [f.value for f in StorageFormat if is_format_supported(f)]


def list_compression_codecs() -> List[str]:
    """List all compression codecs."""
    return [c.value for c in CompressionCodec]


def get_compression_capabilities(name: str) -> Optional[Dict]:
    """Get capability info for a compression codec."""
    return _COMPRESSION_CAPABILITIES.get(name)


def list_sampling_strategies() -> Dict[str, str]:
    """List available negative sampling strategies."""
    return dict(_SAMPLING_STRATEGIES)


def list_feature_primitives() -> Dict[str, str]:
    """List available feature engineering primitives."""
    return dict(_FEATURE_PRIMITIVES)


def list_database_backends() -> List[str]:
    """List available database backends."""
    return list(_DATABASE_BACKENDS.keys())


def get_database_backend(name: str) -> Optional[Dict]:
    """Get capability info for a database backend."""
    return _DATABASE_BACKENDS.get(name)


def get_database_reader_class(name: str) -> Optional[str]:
    """Get the reader class name for a database backend."""
    backend = _DATABASE_BACKENDS.get(name)
    return backend.get("reader_class") if backend else None


def has_database_capability(name: str, capability: str) -> bool:
    """Check if a database backend has a specific capability."""
    backend = _DATABASE_BACKENDS.get(name)
    if backend is None:
        return False
    return capability in backend.get("capabilities", [])


def get_dataset_capabilities(name: str) -> Dict[str, Any]:
    """Get capability metadata for a registered dataset.

    Returns a dict with keys like:
        supports_multi_subset, supports_candidates,
        supports_vector_embeddings, default_eda_subset.
    Unknown datasets return all keys as False/None.
    """
    default = {
        "supports_multi_subset": False,
        "supports_candidates": False,
        "supports_vector_embeddings": False,
        "default_eda_subset": None,
    }
    try:
        meta = DATASET_REGISTRY.get_metadata(name)
    except KeyError:
        return default
    return {
        "supports_multi_subset": bool(meta.get("supports_multi_subset")),
        "supports_candidates": bool(meta.get("supports_candidates")),
        "supports_vector_embeddings": bool(meta.get("supports_vector_embeddings")),
        "default_eda_subset": meta.get("default_eda_subset"),
    }
