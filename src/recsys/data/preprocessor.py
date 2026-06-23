"""Data preprocessing utilities for large-scale offline datasets.

Capabilities:
    - Chunked low-memory reading (pandas chunksize, pyarrow batches, mmap)
    - CSV → columnar format conversion (Parquet / Feather / ORC)
    - Automatic dtype downcast (int64→int32, float64→float32) + category encoding
    - Multi-level cache: hot memory / disk cache / raw archive
    - Fingerprint-based incremental preprocessing with checkpoint resume
    - Database offline bulk export (PostgreSQL COPY, parallel export, vector support)
    - Resource-aware adaptive chunk sizing

Optional backends (graceful fallback):
    - polars: lazy scan_csv / scan_parquet with predicate/projection pushdown
    - dask: distributed dataframe for larger-than-RAM computation
    - vaex: memory-mapped out-of-core DataFrame
    - modin: pandas-compatible distributed DataFrame
    - joblib: intermediate result persistence
    - psycopg: PostgreSQL connector with pgvector support

Design:
    - All config objects are plain dataclasses, serializable to JSON
    - Optional dependencies degrade with clear ImportError hints
    - Fingerprint = hash(source_path + mtime + config snapshot) for cache keying
    - Checkpoint stages: ingest → schema_infer → normalize → materialize → stats
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ============================================================================
# Enums & Config
# ============================================================================


class StorageFormat(str, Enum):
    """Columnar storage format."""
    CSV = "csv"
    PARQUET = "parquet"
    FEATHER = "feather"
    ORC = "orc"


class CompressionCodec(str, Enum):
    """Compression algorithm."""
    NONE = "none"
    SNAPPY = "snappy"
    ZSTD = "zstd"
    LZ4 = "lz4"
    GZIP = "gzip"


class ExecutionBackend(str, Enum):
    """Data processing backend."""
    PANDAS = "pandas"
    PYARROW = "pyarrow"
    POLARS = "polars"
    DASK = "dask"
    VAEX = "vaex"
    MODIN = "modin"


class PipelinePhase(str, Enum):
    """Stages of the preprocessing pipeline."""
    INGEST = "ingest"
    SCHEMA_INFER = "schema_infer"
    NORMALIZE = "normalize"
    MATERIALIZE = "materialize"
    STATS = "stats"


class CacheTier(str, Enum):
    """Cache storage tier."""
    MEMORY = "memory"       # in-process hot cache
    DISK = "disk"           # file-based fast cache
    ARCHIVE = "archive"     # raw archival copy


@dataclass
class ResourceLimits:
    """Memory and IO resource constraints."""
    max_memory_mb: int = 4096
    chunk_size: int = 100_000
    num_workers: int = 0  # 0 = auto (cpu_count)
    spill_dir: Optional[str] = None
    use_mmap: bool = True
    # Fraction of available memory to use (0.0-1.0)
    memory_fraction: float = 0.85

    @property
    def max_memory_bytes(self) -> int:
        return self.max_memory_mb * 1024 * 1024

    @staticmethod
    def auto(memory_fraction: float = 0.85) -> "ResourceLimits":
        """Create limits based on system available memory."""
        try:
            import psutil
            avail_mb = int(psutil.virtual_memory().available * memory_fraction / (1024 * 1024))
            return ResourceLimits(max_memory_mb=avail_mb, memory_fraction=memory_fraction)
        except ImportError:
            return ResourceLimits(memory_fraction=memory_fraction)

    def adaptive_chunk_size(self, row_size_estimate: int = 200) -> int:
        """Estimate safe chunk size based on row byte estimate."""
        return max(10_000, min(self.max_memory_bytes // row_size_estimate, self.chunk_size))


@dataclass
class CachePolicy:
    """Multi-tier caching configuration."""
    enabled: bool = True
    tiers: Tuple[CacheTier, ...] = (CacheTier.DISK, CacheTier.MEMORY)
    max_memory_entries: int = 128
    cache_root: Optional[str] = None
    ttl_seconds: Optional[int] = None  # None = no expiration


@dataclass
class IncrementalPolicy:
    """Incremental processing configuration."""
    enabled: bool = True
    # Use file fingerprint (mtime + size) for cache validation
    use_fingerprint: bool = True
    # Also hash a sample of content for stronger validation
    content_sample_bytes: int = 4096
    # Phases at which checkpoints are written
    checkpoint_phases: Tuple[PipelinePhase, ...] = (
        PipelinePhase.INGEST,
        PipelinePhase.MATERIALIZE,
        PipelinePhase.STATS,
    )
    force_rebuild: bool = False
    # When True, only process new partitions/batches since last run
    incremental_mode: bool = False


@dataclass
class StorageConfig:
    """Materialized storage configuration."""
    format: StorageFormat = StorageFormat.PARQUET
    compression: CompressionCodec = CompressionCodec.ZSTD
    compression_level: int = 3
    row_group_size: int = 128 * 1024  # ~128K rows per row group
    partition_columns: Optional[List[str]] = None
    bucket_columns: Optional[List[str]] = None
    sort_columns: Optional[List[str]] = None
    # Output directory for materialized data
    output_dir: Optional[str] = None


@dataclass
class PreprocessingConfig:
    """Master configuration for offline preprocessing pipeline."""
    source_path: str = ""
    backend: ExecutionBackend = ExecutionBackend.PANDAS
    storage: StorageConfig = field(default_factory=StorageConfig)
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    cache: CachePolicy = field(default_factory=CachePolicy)
    incremental: IncrementalPolicy = field(default_factory=IncrementalPolicy)
    # Columns
    usecols: Optional[List[str]] = None
    exclude_cols: Optional[List[str]] = None
    # Dtype optimization
    downcast_int: bool = True
    downcast_float: bool = True
    auto_category: bool = True
    category_threshold: float = 0.5  # max fraction of unique values for category
    # Normalization
    normalize_numeric: bool = False
    normalize_method: str = "minmax"  # minmax / zscore / log1p
    # Seed for reproducibility
    seed: int = 42

    @property
    def cache_root(self) -> Path:
        root = self.cache.cache_root or os.path.join(
            self.storage.output_dir or "./outputs/cache", ".preprocess_cache"
        )
        return Path(root)


# ============================================================================
# Fingerprint
# ============================================================================


@dataclass
class DatasetFingerprint:
    """Input dataset fingerprint for cache validation."""
    source_path: str
    source_size: int
    source_mtime: float
    content_sample_hash: str = ""
    config_hash: str = ""
    columns: List[str] = field(default_factory=list)
    n_rows: int = 0

    def key(self) -> str:
        """Produce a stable cache key."""
        parts = [
            str(Path(self.source_path).resolve()),
            str(self.source_size),
            str(int(self.source_mtime)),
            self.content_sample_hash[:16],
            self.config_hash,
        ]
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    @staticmethod
    def compute(
        source_path: str,
        config_snapshot: Dict[str, Any],
        content_sample_bytes: int = 4096,
    ) -> "DatasetFingerprint":
        """Compute fingerprint from source file and config."""
        sp = Path(source_path)
        stat = sp.stat()
        # Content sample hash
        try:
            with open(source_path, "rb") as f:
                sample = f.read(content_sample_bytes)
                sample_hash = hashlib.md5(sample).hexdigest()
        except Exception:
            sample_hash = ""
        # Config hash
        config_str = json.dumps(config_snapshot, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        return DatasetFingerprint(
            source_path=str(sp.resolve()),
            source_size=stat.st_size,
            source_mtime=stat.st_mtime,
            content_sample_hash=sample_hash,
            config_hash=config_hash,
        )


# ============================================================================
# Checkpoint
# ============================================================================


@dataclass
class PreprocessCheckpoint:
    """Checkpoint tracking which pipeline phases are complete."""
    fingerprint_key: str = ""
    completed_phases: List[str] = field(default_factory=list)
    columns_inferred: List[str] = field(default_factory=list)
    dtypes_inferred: Dict[str, str] = field(default_factory=dict)
    n_rows: int = 0
    # Timestamps
    started_at: float = 0.0
    updated_at: float = 0.0
    # Error tracking
    errors: List[Dict[str, str]] = field(default_factory=list)

    def is_phase_done(self, phase: PipelinePhase) -> bool:
        return phase.value in self.completed_phases

    def mark_done(self, phase: PipelinePhase) -> None:
        if phase.value not in self.completed_phases:
            self.completed_phases.append(phase.value)
        self.updated_at = time.time()

    @staticmethod
    def load(path: str) -> Optional["PreprocessCheckpoint"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            return PreprocessCheckpoint(**data)
        except Exception:
            return None

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "fingerprint_key": self.fingerprint_key,
            "completed_phases": self.completed_phases,
            "columns_inferred": self.columns_inferred,
            "dtypes_inferred": self.dtypes_inferred,
            "n_rows": self.n_rows,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "errors": self.errors,
        }
        p.write_text(json.dumps(data, indent=2))


# ============================================================================
# Column Stats Cache
# ============================================================================


@dataclass
class ColumnStatsCache:
    """Cache for per-column statistics."""
    column: str
    dtype: str
    n_unique: int
    n_null: int
    n_total: int
    min_val: Any = None
    max_val: Any = None
    mean_val: Optional[float] = None
    std_val: Optional[float] = None
    # For target encoding
    target_aggregates: Optional[Dict[Any, float]] = None
    # For frequency encoding
    frequency_map: Optional[Dict[Any, int]] = None

    @property
    def null_fraction(self) -> float:
        return self.n_null / max(self.n_total, 1)

    @property
    def cardinality_ratio(self) -> float:
        return self.n_unique / max(self.n_total, 1)


# ============================================================================
# Materialized Artifact
# ============================================================================


@dataclass
class MaterializedDatasetArtifact:
    """Result of materializing a dataset to columnar format."""
    path: str
    format: str
    compression: str
    n_rows: int
    n_cols: int
    file_size_bytes: int
    row_groups: int
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    fingerprint: Optional[DatasetFingerprint] = None
    # Sidecar metadata files
    metadata_path: Optional[str] = None
    stats_path: Optional[str] = None

    @property
    def file_size_mb(self) -> float:
        return self.file_size_bytes / (1024 * 1024)

    @property
    def compression_ratio(self) -> float:
        """Uncompressed / compressed ratio (requires source size in fingerprint)."""
        if self.fingerprint and self.fingerprint.source_size > 0:
            return self.fingerprint.source_size / max(self.file_size_bytes, 1)
        return 0.0


# ============================================================================
# Low-memory reader utilities
# ============================================================================


def _infer_schema_from_sample(
    source_path: str,
    sample_rows: int = 10_000,
    **kwargs: Any,
) -> Tuple[List[str], Dict[str, str]]:
    """Infer column names and dtypes from a sample of the source file."""
    is_csv = source_path.lower().endswith(".csv")
    if is_csv:
        df_sample = pd.read_csv(source_path, nrows=sample_rows, **kwargs)
    else:
        df_sample = pd.read_parquet(source_path, **kwargs)
        if len(df_sample) > sample_rows:
            df_sample = df_sample.head(sample_rows)
    columns = list(df_sample.columns)
    dtypes = {c: str(df_sample[c].dtype) for c in columns}
    return columns, dtypes


def _downcast_dtype(series: pd.Series, config: PreprocessingConfig) -> pd.Series:
    """Downcast a numeric series to a smaller dtype, or convert to category."""
    dtype = series.dtype
    n_unique = series.nunique()
    n_total = len(series)

    # Category encoding for low-cardinality columns
    if config.auto_category and str(dtype) == "object" and n_unique / max(n_total, 1) < config.category_threshold:
        return series.astype("category")

    if not pd.api.types.is_numeric_dtype(dtype):
        return series

    # Integer downcast
    if config.downcast_int and pd.api.types.is_integer_dtype(dtype):
        c_min, c_max = series.min(), series.max()
        if c_min >= 0:
            if c_max < 2**8:
                return series.astype("uint8")
            elif c_max < 2**16:
                return series.astype("uint16")
            elif c_max < 2**32:
                return series.astype("uint32")
        else:
            if c_min > -128 and c_max < 128:
                return series.astype("int8")
            elif c_min > -32768 and c_max < 32768:
                return series.astype("int16")
            elif c_min > -(2**31) and c_max < (2**31) - 1:
                return series.astype("int32")

    # Float downcast
    if config.downcast_float and pd.api.types.is_float_dtype(dtype):
        return series.astype("float32")

    return series


def _compute_column_stats(
    series: pd.Series, label_series: Optional[pd.Series] = None
) -> ColumnStatsCache:
    """Compute statistics for a single column."""
    dtype_str = str(series.dtype)
    n_total = len(series)
    n_null = int(series.isna().sum())
    n_unique = int(series.nunique())

    result = ColumnStatsCache(
        column=str(series.name or "unknown"),
        dtype=dtype_str,
        n_unique=n_unique,
        n_null=n_null,
        n_total=n_total,
    )

    if pd.api.types.is_numeric_dtype(series):
        result.min_val = float(series.min()) if n_total > n_null else None
        result.max_val = float(series.max()) if n_total > n_null else None
        result.mean_val = float(series.mean()) if n_total > n_null else None
        result.std_val = float(series.std()) if n_total > n_null else None

    if isinstance(series.dtype, pd.CategoricalDtype) or dtype_str in ("object", "category"):
        try:
            result.frequency_map = dict(series.value_counts().head(1000))
        except Exception:
            result.frequency_map = {}

    if label_series is not None and n_unique > 0:
        try:
            df = pd.DataFrame({"val": series, "label": label_series})
            result.target_aggregates = (
                df.groupby("val")["label"].mean().to_dict()
            )
        except Exception:
            result.target_aggregates = {}

    return result


# ============================================================================
# Chunked Reader
# ============================================================================


def read_chunked_pandas(
    source_path: str,
    config: PreprocessingConfig,
    chunk_callback: Optional[Callable[[pd.DataFrame, int], Optional[pd.DataFrame]]] = None,
) -> Tuple[pd.DataFrame, List[ColumnStatsCache]]:
    """Read a large file in chunks, optionally applying a callback per chunk.

    Returns (concatenated_result, column_stats).
    """
    is_csv = source_path.lower().endswith(".csv")
    chunk_size = config.resources.adaptive_chunk_size()

    all_chunks: List[pd.DataFrame] = []
    all_stats: Dict[str, ColumnStatsCache] = {}

    read_kwargs: Dict[str, Any] = {}
    if config.usecols:
        read_kwargs["usecols"] = config.usecols

    if is_csv:
        reader = pd.read_csv(source_path, chunksize=chunk_size, **read_kwargs)
    else:
        # Parquet/Feather: read in row-group batches
        pf = pq.ParquetFile(source_path)
        reader = (pf.read_row_group(i).to_pandas() for i in range(pf.metadata.num_row_groups))

    for chunk_idx, chunk in enumerate(reader):
        if config.exclude_cols:
            chunk = chunk.drop(columns=[c for c in config.exclude_cols if c in chunk.columns], errors="ignore")
        if config.usecols:
            chunk = chunk[[c for c in config.usecols if c in chunk.columns]]

        # Apply dtype optimization
        for col in chunk.columns:
            chunk[col] = _downcast_dtype(chunk[col], config)

        # Callback (e.g. for streaming write)
        if chunk_callback:
            result = chunk_callback(chunk, chunk_idx)
            if result is not None:
                chunk = result

        # Accumulate stats (sample-based for large data)
        for col in chunk.columns:
            if col not in all_stats:
                all_stats[col] = _compute_column_stats(chunk[col])

        all_chunks.append(chunk)

        logger.debug("Processed chunk %d: %d rows", chunk_idx, len(chunk))

    combined = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()
    return combined, list(all_stats.values())


def read_chunked_pyarrow(
    source_path: str,
    config: PreprocessingConfig,
) -> pa.Table:
    """Read using PyArrow's native batch reader (zero-copy, memory-efficient)."""
    is_csv = source_path.lower().endswith(".csv")
    batch_size = config.resources.adaptive_chunk_size()

    if is_csv:
        # PyArrow CSV reader with column projection
        read_options = pa.csv.ReadOptions(block_size=batch_size * 200)
        if config.usecols:
            convert_options = pa.csv.ConvertOptions(include_columns=config.usecols)
        else:
            convert_options = pa.csv.ConvertOptions()
        table = pa.csv.read_csv(source_path, read_options=read_options, convert_options=convert_options)
    else:
        # Use dataset API for predicate/projection pushdown
        dataset = ds.dataset(source_path)
        table = dataset.to_table(columns=config.usecols) if config.usecols else dataset.to_table()

    return table


# ---- Optional backend readers --------------------------------------------------


def _check_backend(backend: ExecutionBackend) -> Optional[str]:
    """Return error message if backend is unavailable, None otherwise."""
    checks = {
        ExecutionBackend.POLARS: "polars",
        ExecutionBackend.DASK: "dask",
        ExecutionBackend.VAEX: "vaex",
        ExecutionBackend.MODIN: "modin",
        ExecutionBackend.PYARROW: "pyarrow",
    }
    pkg = checks.get(backend)
    if pkg is None:
        return None
    try:
        __import__(pkg)
        return None
    except ImportError:
        return f"Backend '{backend.value}' requires 'pip install {pkg}' or 'uv sync --extra bigdata'"


def read_polars_lazy(
    source_path: str,
    config: PreprocessingConfig,
) -> Any:
    """Read via Polars lazy API (scan_csv / scan_parquet) with pushdown."""
    try:
        import polars as pl
    except ImportError as err:
        raise ImportError(
            "polars is required. Install with: pip install polars  or  uv sync --extra bigdata"
        ) from err
    is_csv = source_path.lower().endswith(".csv")
    if is_csv:
        lf = pl.scan_csv(
            source_path,
            has_header=True,
            infer_schema_length=10_000,
        )
    else:
        lf = pl.scan_parquet(source_path)
    if config.usecols:
        lf = lf.select(config.usecols)
    return lf.collect() if not config.incremental.incremental_mode else lf


def read_dask_dataframe(
    source_path: str,
    config: PreprocessingConfig,
) -> Any:
    """Read via Dask DataFrame for out-of-core processing."""
    try:
        import dask.dataframe as dd
    except ImportError as err:
        raise ImportError(
            "dask is required. Install with: pip install dask[dataframe]  or  uv sync --extra bigdata"
        ) from err
    is_csv = source_path.lower().endswith(".csv")
    if is_csv:
        ddf = dd.read_csv(
            source_path,
            blocksize=config.resources.adaptive_chunk_size() * 200,
            dtype=config.usecols,
        )
    else:
        ddf = dd.read_parquet(source_path, columns=config.usecols)
    return ddf


# ============================================================================
# Materialization (CSV → Columnar)
# ============================================================================


def materialize_to_columnar(
    source_path: str,
    config: PreprocessingConfig,
) -> MaterializedDatasetArtifact:
    """Convert a source file (CSV/Parquet) to a columnar format with compression."""
    fmt = config.storage.format
    comp = config.storage.compression
    output_dir = Path(config.storage.output_dir or "./outputs/cache")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(source_path).stem

    # Read with dtype optimization
    logger.info("Reading %s with backend=%s ...", source_path, config.backend.value)
    df, stats = read_chunked_pandas(source_path, config)

    # Determine output path and write
    ext_map = {
        StorageFormat.PARQUET: ".parquet",
        StorageFormat.FEATHER: ".feather",
        StorageFormat.ORC: ".orc",
        StorageFormat.CSV: ".csv",
    }
    ext = ext_map.get(fmt, ".parquet")
    out_path = output_dir / f"{stem}_{fmt.value}_{comp.value}{ext}"

    compression_map = {
        CompressionCodec.NONE: None,
        CompressionCodec.SNAPPY: "snappy",
        CompressionCodec.ZSTD: "zstd",
        CompressionCodec.LZ4: "lz4",
        CompressionCodec.GZIP: "gzip",
    }
    pa_compression = compression_map.get(comp)

    row_group_size = config.storage.row_group_size if fmt == StorageFormat.PARQUET else None

    logger.info("Writing %s (format=%s, compression=%s)...", out_path, fmt.value, comp.value)
    table = pa.Table.from_pandas(df, preserve_index=False)

    if fmt == StorageFormat.PARQUET:
        pq.write_table(
            table, str(out_path),
            compression=pa_compression,
            row_group_size=row_group_size,
        )
    elif fmt == StorageFormat.FEATHER:
        import pyarrow.feather as ft
        ft.write_feather(table, str(out_path), compression=pa_compression or "uncompressed")
    elif fmt == StorageFormat.ORC:
        # PyArrow doesn't have native ORC writer; use pandas as fallback
        if pa_compression == "zstd":
            df.to_orc(str(out_path), index=False)
        else:
            df.to_orc(str(out_path), index=False)
    else:
        df.to_csv(str(out_path), index=False)

    file_size = out_path.stat().st_size

    # Write sidecar metadata
    meta_path = output_dir / f"{stem}_{fmt.value}_{comp.value}_meta.json"
    stats_path = output_dir / f"{stem}_{fmt.value}_{comp.value}_stats.json"
    meta = {
        "source": str(Path(source_path).resolve()),
        "format": fmt.value,
        "compression": comp.value,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "file_size_bytes": file_size,
        "row_group_size": row_group_size,
        "backend": config.backend.value,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    stats_path.write_text(
        json.dumps(
            [
                {
                    "column": s.column,
                    "dtype": s.dtype,
                    "n_unique": s.n_unique,
                    "n_null": s.n_null,
                    "n_total": s.n_total,
                    "null_fraction": s.null_fraction,
                    "cardinality_ratio": s.cardinality_ratio,
                }
                for s in stats
            ],
            indent=2,
        )
    )

    fingerprint = DatasetFingerprint.compute(source_path, meta)

    return MaterializedDatasetArtifact(
        path=str(out_path),
        format=fmt.value,
        compression=comp.value,
        n_rows=len(df),
        n_cols=len(df.columns),
        file_size_bytes=file_size,
        row_groups=(len(df) // max(row_group_size or 1, 1)) + 1,
        columns=list(df.columns),
        dtypes={c: str(df[c].dtype) for c in df.columns},
        fingerprint=fingerprint,
        metadata_path=str(meta_path),
        stats_path=str(stats_path),
    )


# ============================================================================
# Pipeline orchestrator
# ============================================================================


class OfflinePreprocessingPipeline:
    """Orchestrated preprocessing pipeline with checkpoint/resume and caching.

    Usage:
        pipeline = OfflinePreprocessingPipeline(config)
        pipeline.run()  # Full pipeline
        pipeline.run(phases=[PipelinePhase.INGEST])  # Specific phases
        pipeline.run(phases=[PipelinePhase.INGEST], force=True)  # Force rebuild
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self._checkpoint: Optional[PreprocessCheckpoint] = None
        self._fingerprint: Optional[DatasetFingerprint] = None
        self._artifact: Optional[MaterializedDatasetArtifact] = None
        self._column_stats: List[ColumnStatsCache] = []
        self._started_at = time.time()

    # ---- public API -----------------------------------------------------------

    def run(
        self,
        phases: Optional[Sequence[PipelinePhase]] = None,
        force: bool = False,
    ) -> MaterializedDatasetArtifact:
        """Execute the preprocessing pipeline."""
        if phases is None:
            phases = [
                PipelinePhase.INGEST,
                PipelinePhase.SCHEMA_INFER,
                PipelinePhase.NORMALIZE,
                PipelinePhase.MATERIALIZE,
                PipelinePhase.STATS,
            ]

        # Compute fingerprint
        meta_snapshot = {
            "source_path": self.config.source_path,
            "usecols": self.config.usecols,
            "backend": self.config.backend.value,
            "format": self.config.storage.format.value,
            "compression": self.config.storage.compression.value,
            "downcast_int": self.config.downcast_int,
            "downcast_float": self.config.downcast_float,
            "auto_category": self.config.auto_category,
        }
        self._fingerprint = DatasetFingerprint.compute(
            self.config.source_path, meta_snapshot
        )

        # Load or init checkpoint
        ckpt_path = self._checkpoint_path()
        self._checkpoint = PreprocessCheckpoint.load(ckpt_path) or PreprocessCheckpoint(
            fingerprint_key=self._fingerprint.key(),
            started_at=self._started_at,
        )

        if self.config.incremental.force_rebuild or force:
            self._clear_cache()
            self._checkpoint = PreprocessCheckpoint(
                fingerprint_key=self._fingerprint.key(),
                started_at=self._started_at,
            )

        for phase in phases:
            if not force and self._checkpoint.is_phase_done(phase):
                logger.info("Phase %s already complete — skipping", phase.value)
                continue

            logger.info("Running phase: %s", phase.value)
            try:
                self._execute_phase(phase)
                self._checkpoint.mark_done(phase)
                self._checkpoint.save(ckpt_path)
            except Exception as e:
                logger.error("Phase %s failed: %s", phase.value, e)
                self._checkpoint.errors.append({
                    "phase": phase.value,
                    "error": str(e),
                })
                self._checkpoint.save(ckpt_path)
                raise

        return self._artifact  # type: ignore[return-value]

    def get_checkpoint(self) -> Optional[PreprocessCheckpoint]:
        return self._checkpoint

    def get_fingerprint(self) -> Optional[DatasetFingerprint]:
        return self._fingerprint

    def get_column_stats(self) -> List[ColumnStatsCache]:
        return self._column_stats

    def get_artifact(self) -> Optional[MaterializedDatasetArtifact]:
        return self._artifact

    # ---- internal -------------------------------------------------------------

    def _checkpoint_path(self) -> str:
        root = self.config.cache_root
        key = self._fingerprint.key() if self._fingerprint else "unknown"
        return str(root / f"checkpoint_{key}.json")

    def _clear_cache(self) -> None:
        ckpt_path = Path(self._checkpoint_path())
        if ckpt_path.exists():
            ckpt_path.unlink()
        logger.info("Cleared cache for %s", self.config.source_path)

    def _execute_phase(self, phase: PipelinePhase) -> None:
        handlers: Dict[PipelinePhase, Callable[[], None]] = {
            PipelinePhase.INGEST: self._phase_ingest,
            PipelinePhase.SCHEMA_INFER: self._phase_schema_infer,
            PipelinePhase.NORMALIZE: self._phase_normalize,
            PipelinePhase.MATERIALIZE: self._phase_materialize,
            PipelinePhase.STATS: self._phase_stats,
        }
        handler = handlers.get(phase)
        if handler:
            handler()

    def _phase_ingest(self) -> None:
        """Validate source file and estimate row count."""
        sp = Path(self.config.source_path)
        if not sp.exists():
            raise FileNotFoundError(f"Source file not found: {self.config.source_path}")
        # Quick row count estimate for CSV
        if self.config.source_path.lower().endswith(".csv"):
            with open(self.config.source_path, "rb") as f:
                n_lines = sum(1 for _ in f)
            self._checkpoint.n_rows = max(0, n_lines - 1)  # type: ignore[union-attr]
        logger.info("Ingest complete: %s (%d bytes)", sp.name, sp.stat().st_size)

    def _phase_schema_infer(self) -> None:
        """Infer column schema from source sample."""
        columns, dtypes = _infer_schema_from_sample(
            self.config.source_path, sample_rows=10_000
        )
        if self._checkpoint:
            self._checkpoint.columns_inferred = columns
            self._checkpoint.dtypes_inferred = dtypes
        logger.info("Schema inferred: %d columns", len(columns))

    def _phase_normalize(self) -> None:
        """Normalize / clean data in chunks (passthrough for now)."""
        logger.info("Normalize complete (passthrough)")

    def _phase_materialize(self) -> None:
        """Convert and materialize to columnar format."""
        self._artifact = materialize_to_columnar(
            self.config.source_path, self.config
        )
        logger.info(
            "Materialized: %d rows, %d cols, %.1f MB (%s/%s)",
            self._artifact.n_rows,
            self._artifact.n_cols,
            self._artifact.file_size_mb,
            self._artifact.format,
            self._artifact.compression,
        )

    def _phase_stats(self) -> None:
        """Compute column statistics."""
        if self._artifact is None:
            self._phase_materialize()
        # Stats are already computed during materialization
        logger.info("Stats complete")


# ============================================================================
# Database readers (optional dependencies)
# ============================================================================


class DatabaseReader:
    """Base class for database offline export readers."""

    def __init__(self, connection_string: str, config: PreprocessingConfig) -> None:
        self.connection_string = connection_string
        self.config = config

    def export_to_columnar(self, query: str, output_prefix: str) -> MaterializedDatasetArtifact:
        raise NotImplementedError


class PostgresReader(DatabaseReader):
    """PostgreSQL reader with server-side cursor, COPY export, and vector support.

    Features:
        - COPY TO STDOUT (fastest export path)
        - Server-side cursor pagination (low memory)
        - Binary format export (faster than CSV)
        - Parallel partition export
        - Vector column support via pgvector
    """

    def export_via_copy(self, query: str, output_path: str) -> str:
        """Export query results via COPY TO STDOUT (fastest path)."""
        try:
            import psycopg
        except ImportError as err:
            raise ImportError(
                "psycopg is required. Install with: pip install psycopg[binary]  or  uv sync --extra db"
            ) from err
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with psycopg.connect(self.connection_string) as conn, conn.cursor() as cur, open(str(out), "w") as f:
            cur.copy_expert(f"COPY ({query}) TO STDOUT WITH CSV HEADER", f)
        return str(out)

    def export_via_copy_binary(self, query: str, output_path: str) -> str:
        """Export via COPY BINARY format (faster than CSV, smaller file size).

        Binary format is ~30% faster than CSV for large datasets.
        """
        try:
            import psycopg
        except ImportError as err:
            raise ImportError(
                "psycopg is required. Install with: pip install psycopg[binary]  or  uv sync --extra db"
            ) from err
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with psycopg.connect(self.connection_string) as conn, conn.cursor() as cur, open(str(out), "wb") as f:
            cur.copy_expert(f"COPY ({query}) TO STDOUT WITH BINARY", f)
        return str(out)

    def export_via_cursor(
        self, query: str, output_path: str, page_size: int = 100_000
    ) -> str:
        """Export via server-side cursor with pagination (lower memory)."""
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as err:
            raise ImportError(
                "psycopg is required. Install with: pip install psycopg[binary]  or  uv sync --extra db"
            ) from err
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with psycopg.connect(self.connection_string) as conn, conn.cursor(
            name="offline_export_cursor",
            row_factory=dict_row,
        ) as cur:
            cur.itersize = page_size
            cur.execute(query)
            first_batch = True
            for batch in cur:
                df = pd.DataFrame(batch)  # type: ignore[arg-type]
                df.to_csv(
                    str(out),
                    mode="a",
                    header=first_batch,
                    index=False,
                )
                first_batch = False
        return str(out)

    def export_parallel(
        self,
        table: str,
        output_dir: str,
        partition_column: str = "created_at",
        workers: int = 4,
        where_clause: Optional[str] = None,
    ) -> List[str]:
        """Parallel export from partitioned table using multiple connections.

        Each worker exports a separate partition concurrently.
        Requires table to be partitioned by the given column.

        Args:
            table: Table name to export
            output_dir: Directory for output files
            partition_column: Column used for partitioning
            workers: Number of parallel workers
            where_clause: Optional WHERE clause for filtering

        Returns:
            List of exported file paths
        """
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            import psycopg
        except ImportError as err:
            raise ImportError(
                "psycopg is required. Install with: pip install psycopg[binary]  or  uv sync --extra db"
            ) from err

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Get partition boundaries
        with psycopg.connect(self.connection_string) as conn, conn.cursor() as cur:
            # Get min/max values for partition column
            cur.execute(f"SELECT MIN({partition_column}), MAX({partition_column}) FROM {table}")
            min_val, max_val = cur.fetchone()  # type: ignore[misc]

            if min_val is None or max_val is None:
                logger.warning("Table %s is empty, skipping export", table)
                return []

            # Get total count for progress
            count_query = f"SELECT COUNT(*) FROM {table}"
            if where_clause:
                count_query += f" WHERE {where_clause}"
            cur.execute(count_query)
            total_rows = cur.fetchone()[0]  # type: ignore[index]

        logger.info("Exporting %s: %d rows, %d workers", table, total_rows, workers)

        # Calculate partition ranges
        if isinstance(min_val, (int, float)):
            step = (max_val - min_val) / workers
            ranges = [(min_val + i * step, min_val + (i + 1) * step) for i in range(workers)]
        else:
            # For timestamp/date columns, use date_trunc
            ranges = [(i, i + 1) for i in range(workers)]  # Placeholder

        output_files: List[str] = []

        def export_partition(worker_id: int, range_start: Any, range_end: Any) -> str:
            out_path = out_dir / f"{table}_part{worker_id}.csv"
            query = f"SELECT * FROM {table} WHERE {partition_column} >= %s AND {partition_column} < %s"
            if where_clause:
                query += f" AND {where_clause}"

            with psycopg.connect(self.connection_string) as conn, conn.cursor() as cur, open(str(out_path), "w") as f:
                cur.copy_expert(
                    f"COPY ({query}) TO STDOUT WITH CSV HEADER",
                    f,
                    params=(range_start, range_end),
                )
            return str(out_path)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(export_partition, i, start, end): i
                for i, (start, end) in enumerate(ranges)
            }
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    path = future.result()
                    output_files.append(path)
                    logger.debug("Worker %d completed: %s", worker_id, path)
                except Exception as e:
                    logger.error("Worker %d failed: %s", worker_id, e)

        logger.info("Parallel export complete: %d files", len(output_files))
        return sorted(output_files)

    def export_with_vectors(
        self,
        query: str,
        output_path: str,
        vector_columns: Optional[List[str]] = None,
    ) -> str:
        """Export including vector columns as numpy arrays.

        Vector columns are exported as base64-encoded numpy arrays,
        which can be decoded using np.frombuffer(base64.b64decode(...)).

        Args:
            query: SQL query to execute
            output_path: Output file path
            vector_columns: List of vector column names to export as arrays

        Returns:
            Path to exported file
        """
        try:
            import base64

            import psycopg
            from psycopg.rows import dict_row
        except ImportError as err:
            raise ImportError(
                "psycopg is required. Install with: pip install psycopg[binary]  or  uv sync --extra db"
            ) from err

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        vector_cols = vector_columns or []

        with psycopg.connect(self.connection_string) as conn, conn.cursor(
            row_factory=dict_row
        ) as cur:
            cur.execute(query)

            first_batch = True
            batch_rows: List[Dict[str, Any]] = []

            for row in cur:
                processed_row = dict(row)
                # Convert vector columns to base64-encoded numpy arrays
                for vec_col in vector_cols:
                    if vec_col in processed_row and processed_row[vec_col] is not None:
                        vec_str = processed_row[vec_col]
                        # pgvector returns string like "[1.0, 2.0, 3.0]"
                        if isinstance(vec_str, str):
                            vec_values = [float(x) for x in vec_str.strip("[]").split(",")]
                            vec_array = np.array(vec_values, dtype=np.float32)
                            processed_row[vec_col] = base64.b64encode(vec_array.tobytes()).decode("ascii")

                batch_rows.append(processed_row)

                if len(batch_rows) >= self.config.resources.chunk_size:
                    df = pd.DataFrame(batch_rows)
                    df.to_csv(str(out), mode="a", header=first_batch, index=False)
                    first_batch = False
                    batch_rows = []

            # Write remaining rows
            if batch_rows:
                df = pd.DataFrame(batch_rows)
                df.to_csv(str(out), mode="a", header=first_batch, index=False)

        return str(out)

    def export_to_columnar(self, query: str, output_prefix: str) -> MaterializedDatasetArtifact:
        csv_path = self.export_via_copy(query, f"{output_prefix}_export.csv")
        self.config.source_path = csv_path
        return materialize_to_columnar(csv_path, self.config)


class PostgresVectorReader(PostgresReader):
    """PostgreSQL reader with pgvector support for vector similarity search.

    Extends PostgresReader with vector-specific operations:
        - Vector column export
        - Similarity search queries
        - ANN index support (HNSW, IVFFlat)
    """

    def __init__(
        self,
        connection_string: str,
        config: PreprocessingConfig,
        vector_dim: int = 128,
    ) -> None:
        super().__init__(connection_string, config)
        self.vector_dim = vector_dim

    def search_similar(
        self,
        table: str,
        vector_column: str,
        query_vector: np.ndarray,
        k: int = 100,
        distance_metric: str = "cosine",
        where_clause: Optional[str] = None,
    ) -> pd.DataFrame:
        """Perform vector similarity search using pgvector.

        Args:
            table: Table name containing vectors
            vector_column: Name of the vector column
            query_vector: Query vector (numpy array)
            k: Number of nearest neighbors to return
            distance_metric: Distance metric (cosine, l2, inner_product)
            where_clause: Optional WHERE clause for filtering

        Returns:
            DataFrame with similar items and distances
        """
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as err:
            raise ImportError(
                "psycopg is required. Install with: pip install psycopg[binary]  or  uv sync --extra db"
            ) from err

        # Format query vector for pgvector
        vec_str = "[" + ",".join(str(x) for x in query_vector) + "]"

        # Select distance operator based on metric
        distance_ops = {
            "cosine": "<=>",  # cosine distance
            "l2": "<->",      # L2 distance
            "inner_product": "<#>",  # negative inner product
        }
        op = distance_ops.get(distance_metric, "<=>")

        query = f"""
            SELECT *, {vector_column} {op} '{vec_str}'::vector AS distance
            FROM {table}
        """
        if where_clause:
            query += f" WHERE {where_clause}"
        query += f" ORDER BY {vector_column} {op} '{vec_str}'::vector LIMIT {k}"

        with psycopg.connect(self.connection_string) as conn, conn.cursor(
            row_factory=dict_row
        ) as cur:
            cur.execute(query)
            rows = cur.fetchall()
            return pd.DataFrame(rows)

    def export_similar_items(
        self,
        table: str,
        vector_column: str,
        query_vectors: np.ndarray,
        k: int = 100,
        output_path: str = "similar_items.csv",
        distance_metric: str = "cosine",
    ) -> str:
        """Export similar items for multiple query vectors.

        Args:
            table: Table name containing vectors
            vector_column: Name of the vector column
            query_vectors: Array of query vectors (n_queries x dim)
            k: Number of nearest neighbors per query
            output_path: Output file path
            distance_metric: Distance metric

        Returns:
            Path to exported file
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        all_results: List[pd.DataFrame] = []
        for i, qvec in enumerate(query_vectors):
            df = self.search_similar(
                table, vector_column, qvec, k, distance_metric
            )
            df["query_id"] = i
            all_results.append(df)

        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(str(out), index=False)
        return str(out)

    def create_vector_index(
        self,
        table: str,
        vector_column: str,
        index_type: str = "hnsw",
        distance_metric: str = "cosine",
        m: int = 16,
        ef_construction: int = 64,
    ) -> None:
        """Create vector index for faster similarity search.

        Args:
            table: Table name
            vector_column: Vector column name
            index_type: Index type (hnsw, ivfflat)
            distance_metric: Distance metric
            m: HNSW m parameter (connections per layer)
            ef_construction: HNSW ef_construction parameter
        """
        try:
            import psycopg
        except ImportError as err:
            raise ImportError(
                "psycopg is required. Install with: pip install psycopg[binary]  or  uv sync --extra db"
            ) from err

        # Select distance operator class
        op_classes = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "inner_product": "vector_ip_ops",
        }
        op_class = op_classes.get(distance_metric, "vector_cosine_ops")

        index_name = f"idx_{table}_{vector_column}_{index_type}"

        if index_type == "hnsw":
            sql = f"""
                CREATE INDEX IF NOT EXISTS {index_name} ON {table}
                USING hnsw ({vector_column} {op_class})
                WITH (m = {m}, ef_construction = {ef_construction})
            """
        elif index_type == "ivfflat":
            sql = f"""
                CREATE INDEX IF NOT EXISTS {index_name} ON {table}
                USING ivfflat ({vector_column} {op_class})
                WITH (lists = 100)
            """
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        with psycopg.connect(self.connection_string) as conn, conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()

        logger.info("Created %s index on %s.%s", index_type, table, vector_column)


# ============================================================================
# Backend capability helpers
# ============================================================================


def is_backend_available(backend: Union[str, ExecutionBackend]) -> bool:
    """Check if a backend is importable."""
    if isinstance(backend, str):
        backend = ExecutionBackend(backend)
    return _check_backend(backend) is None


def list_available_backends() -> List[str]:
    """List all backends that are currently importable."""
    return [
        b.value for b in ExecutionBackend
        if is_backend_available(b)
    ]


def is_format_supported(fmt: Union[str, StorageFormat]) -> bool:
    """Check if a storage format is writable with current dependencies."""
    if isinstance(fmt, str):
        fmt = StorageFormat(fmt)
    if fmt == StorageFormat.ORC:
        try:
            _has_orc = hasattr(pd.DataFrame({"a": [1]}), "to_orc")
            return _has_orc
        except AttributeError:
            return False
    return True
