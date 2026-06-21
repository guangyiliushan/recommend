"""Preprocessor 单元测试：config、fingerprint、checkpoint、materialization。"""

from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data.preprocessor import (
    CompressionCodec,
    DatasetFingerprint,
    ExecutionBackend,
    OfflinePreprocessingPipeline,
    PipelinePhase,
    PreprocessCheckpoint,
    PreprocessingConfig,
    ResourceLimits,
    StorageConfig,
    StorageFormat,
    is_backend_available,
    is_format_supported,
    list_available_backends,
    materialize_to_columnar,
    read_chunked_pandas,
)


class TestEnums:
    def test_storage_format_values(self):
        assert StorageFormat.PARQUET.value == "parquet"
        assert StorageFormat.FEATHER.value == "feather"
        assert StorageFormat.ORC.value == "orc"

    def test_compression_codec_values(self):
        assert CompressionCodec.SNAPPY.value == "snappy"
        assert CompressionCodec.ZSTD.value == "zstd"

    def test_execution_backend_values(self):
        assert ExecutionBackend.PANDAS.value == "pandas"


class TestResourceLimits:
    def test_defaults(self):
        limits = ResourceLimits()
        assert limits.max_memory_mb == 4096
        assert limits.chunk_size == 100_000

    def test_adaptive_chunk_size(self):
        limits = ResourceLimits(max_memory_mb=1024)
        chunk = limits.adaptive_chunk_size(row_size_estimate=200)
        assert 10000 <= chunk <= 100_000


class TestFingerprint:
    def test_compute_and_key(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("a,b\n1,2\n3,4\n")
        fp = DatasetFingerprint.compute(str(csv), {"format": "parquet"})
        assert fp.source_size > 0
        assert len(fp.key()) == 32  # MD5 hex

    def test_different_config_different_key(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("a,b\n1,2\n3,4\n")
        fp1 = DatasetFingerprint.compute(str(csv), {"format": "parquet"})
        fp2 = DatasetFingerprint.compute(str(csv), {"format": "orc"})
        assert fp1.key() != fp2.key()


class TestCheckpoint:
    def test_save_load(self, tmp_path):
        ckpt = PreprocessCheckpoint(fingerprint_key="abc123")
        ckpt.mark_done(PipelinePhase.INGEST)
        ckpt.mark_done(PipelinePhase.MATERIALIZE)

        path = str(tmp_path / "ckpt.json")
        ckpt.save(path)

        loaded = PreprocessCheckpoint.load(path)
        assert loaded is not None
        assert loaded.fingerprint_key == "abc123"
        assert loaded.is_phase_done(PipelinePhase.INGEST)
        assert loaded.is_phase_done(PipelinePhase.MATERIALIZE)
        assert not loaded.is_phase_done(PipelinePhase.STATS)

    def test_load_nonexistent(self, tmp_path):
        assert PreprocessCheckpoint.load(str(tmp_path / "nonexistent.json")) is None


class TestPreprocessingConfig:
    def test_default_cache_root(self):
        config = PreprocessingConfig(
            storage=StorageConfig(output_dir="./outputs/test_cache")
        )
        assert ".preprocess_cache" in str(config.cache_root)


class TestChunkedReader:
    def test_read_csv_with_downcast(self, tmp_path):
        csv = tmp_path / "data.csv"
        n = 1000
        df = pd.DataFrame({
            "int_col": np.arange(n, dtype=np.int64),
            "float_col": np.random.randn(n).astype(np.float64),
            "cat_col": np.random.choice(["A", "B", "C"], size=n),
        })
        df.to_csv(str(csv), index=False)

        config = PreprocessingConfig(
            source_path=str(csv),
            downcast_int=True,
            downcast_float=True,
            auto_category=True,
        )
        result, stats = read_chunked_pandas(str(csv), config)
        assert len(result) == n
        # int should be downcasted to uint16 (n=1000 < 65536)
        assert result["int_col"].dtype in ("uint16", "int16", "uint32", "int32")
        # float should be downcasted to float32
        assert result["float_col"].dtype == "float32"


class TestMaterialize:
    def test_csv_to_parquet(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        csv = tmp_path / "data.csv"
        n = 500
        df = pd.DataFrame({
            "user_id": np.arange(n, dtype=np.int64),
            "item_id": np.random.randint(1, 100, size=n),
            "label": np.random.randint(0, 2, size=n).astype(np.float32),
        })
        df.to_csv(str(csv), index=False)

        config = PreprocessingConfig(
            source_path=str(csv),
            storage=StorageConfig(
                format=StorageFormat.PARQUET,
                compression=CompressionCodec.SNAPPY,
                output_dir=str(output_dir),
                row_group_size=256,
            ),
        )
        artifact = materialize_to_columnar(str(csv), config)
        assert artifact.format == "parquet"
        assert artifact.n_rows == n
        assert Path(artifact.path).exists()
        assert artifact.metadata_path and Path(artifact.metadata_path).exists()

    def test_csv_to_feather(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        csv = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        df.to_csv(str(csv), index=False)

        config = PreprocessingConfig(
            source_path=str(csv),
            storage=StorageConfig(
                format=StorageFormat.FEATHER,
                compression=CompressionCodec.ZSTD,
                output_dir=str(output_dir),
            ),
        )
        artifact = materialize_to_columnar(str(csv), config)
        assert artifact.format == "feather"
        assert Path(artifact.path).exists()


class TestPipeline:
    def test_full_pipeline(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        csv = tmp_path / "data.csv"
        df = pd.DataFrame({
            "user_id": np.arange(200, dtype=np.int64),
            "item_id": np.random.randint(1, 50, size=200),
            "label": np.random.randint(0, 2, size=200).astype(np.float32),
        })
        df.to_csv(str(csv), index=False)

        config = PreprocessingConfig(
            source_path=str(csv),
            storage=StorageConfig(
                format=StorageFormat.PARQUET,
                compression=CompressionCodec.SNAPPY,
                output_dir=str(output_dir),
            ),
            downcast_int=True,
            downcast_float=True,
        )
        pipeline = OfflinePreprocessingPipeline(config)
        artifact = pipeline.run()
        assert artifact is not None
        assert artifact.n_rows == 200

    def test_pipeline_resume(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        csv = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_csv(str(csv), index=False)

        config = PreprocessingConfig(
            source_path=str(csv),
            storage=StorageConfig(
                format=StorageFormat.PARQUET,
                compression=CompressionCodec.SNAPPY,
                output_dir=str(output_dir),
            ),
        )
        pipeline = OfflinePreprocessingPipeline(config)

        # First run: INGEST only
        pipeline.run(phases=[PipelinePhase.INGEST])
        ckpt = pipeline.get_checkpoint()
        assert ckpt is not None
        assert ckpt.is_phase_done(PipelinePhase.INGEST)

        # Second run: remaining phases should resume from checkpoint
        artifact = pipeline.run()
        assert artifact is not None


class TestBackendHelpers:
    def test_list_available_backends(self):
        backends = list_available_backends()
        assert "pandas" in backends
        assert "pyarrow" in backends

    def test_is_backend_available(self):
        assert is_backend_available(ExecutionBackend.PANDAS) is True
        assert is_backend_available(ExecutionBackend.PYARROW) is True

    def test_is_format_supported(self):
        assert is_format_supported(StorageFormat.PARQUET) is True
        assert is_format_supported(StorageFormat.CSV) is True
