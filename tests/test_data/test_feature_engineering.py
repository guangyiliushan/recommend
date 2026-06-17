"""Feature engineering 单元测试。"""

import numpy as np
import pandas as pd

from recsys.data.feature_engineering import (
    ChunkFeatureEngineer,
    FeatureEngineeringConfig,
    FeatureManifest,
    embedding_dim_heuristic,
    hash_crossing,
    sequence_pad_truncate,
)


class TestEmbeddingDimHeuristic:
    def test_google_rule(self):
        dim = embedding_dim_heuristic(1000, method="google")
        assert 5 <= dim <= 20

    def test_small_vocab(self):
        assert embedding_dim_heuristic(1) == 1
        # 5 categories → 5^0.25 * 2 ≈ 3
        assert embedding_dim_heuristic(5) >= 1

    def test_fastai_rule(self):
        dim = embedding_dim_heuristic(10000, method="fastai")
        assert 1 <= dim <= 50


class TestSequencePadTruncate:
    def test_pad_right(self):
        result = sequence_pad_truncate([1, 2, 3], 5, pad_value=0)
        assert result == [1, 2, 3, 0, 0]

    def test_truncate_right(self):
        result = sequence_pad_truncate([1, 2, 3, 4, 5], 3)
        assert result == [1, 2, 3]

    def test_truncate_left(self):
        result = sequence_pad_truncate([1, 2, 3, 4, 5], 3, truncate_from="left")
        assert result == [3, 4, 5]


class TestHashCrossing:
    def test_basic(self):
        a = pd.Series(["A", "B", "A"])
        b = pd.Series(["X", "Y", "X"])
        result = hash_crossing(a, b, bucket_size=1000)
        assert len(result) == 3
        assert result.nunique() == 2  # ("A_X", "B_Y", "A_X")


class TestChunkFeatureEngineer:
    def test_fit_on_chunks(self):
        chunks = [
            pd.DataFrame({
                "cat": ["A", "B", "A"],
                "num": [1.0, 2.0, 3.0],
                "label": [0, 1, 1],
            }),
            pd.DataFrame({
                "cat": ["B", "C", "A"],
                "num": [4.0, 5.0, 6.0],
                "label": [0, 0, 1],
            }),
        ]
        config = FeatureEngineeringConfig(
            frequency_encode=False,
            target_encode=True,
            category_encode=True,
            normalize=True,
        )
        engineer = ChunkFeatureEngineer(config)
        manifest = engineer.fit_on_chunks(chunks, label_col="label")

        assert "cat" in manifest.category_vocabs
        assert "num" in manifest.numeric_stats
        assert "target_aggregates" in dir(manifest)

    def test_transform_chunk(self):
        chunks = [
            pd.DataFrame({
                "cat": ["A", "B", "A", "B", "C"],
                "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            }),
        ]
        config = FeatureEngineeringConfig(
            category_encode=True,
            normalize=True,
        )
        engineer = ChunkFeatureEngineer(config)
        engineer.fit_on_chunks(chunks)
        result = engineer.transform_chunk(chunks[0].copy())

        assert "cat_cat" in result.columns
        assert result["cat_cat"].dtype in (np.int32, np.int64)

    def test_feature_manifest_save_load(self, tmp_path):
        config = FeatureEngineeringConfig(category_encode=True)
        engineer = ChunkFeatureEngineer(config)
        engineer.fit_on_chunks([
            pd.DataFrame({"cat": ["A", "B", "C"], "num": [1.0, 2.0, 3.0]})
        ])

        path = str(tmp_path / "manifest.json")
        engineer.manifest.save(path)

        loaded = FeatureManifest.load(path)
        assert loaded is not None
        assert len(loaded.category_vocabs) == 1
        assert "cat" in loaded.category_vocabs
