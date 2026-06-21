"""Negative sampling 单元测试。"""

import numpy as np

from recsys.data.negative_sampling import (
    ItemPoolStats,
    NegativeSampler,
    NegativeSamplingConfig,
    SamplingStrategy,
    create_sampler,
)


class TestItemPoolStats:
    def test_save_load(self, tmp_path):
        item_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        freqs = np.array([10.0, 20.0, 5.0, 3.0, 2.0], dtype=np.float64)
        pool = ItemPoolStats(item_ids=item_ids, frequencies=freqs, n_total_items=5)

        path = str(tmp_path / "pool.npz")
        pool.save(path)

        loaded = ItemPoolStats.load(path)
        assert loaded is not None
        assert loaded.n_total_items == 5
        np.testing.assert_array_equal(loaded.item_ids, item_ids)

    def test_sampling_probs(self):
        pool = ItemPoolStats(
            item_ids=np.array([1, 2, 3]),
            frequencies=np.array([5.0, 3.0, 2.0]),
        )
        probs = pool.sampling_probs
        assert probs[0] > probs[1] > probs[2]


class TestNegativeSampler:
    def test_uniform_sampling(self):
        config = NegativeSamplingConfig(
            strategy=SamplingStrategy.UNIFORM,
            num_negatives=4,
            seed=42,
        )
        sampler = NegativeSampler(config)
        sampler.fit(item_ids=np.arange(1, 101))

        positives = np.array([1, 2, 3, 4, 5])
        negatives = sampler.sample(positives)
        assert len(negatives) == 20  # 5 * 4
        assert all(n in sampler.get_item_universe() for n in negatives)

    def test_popularity_sampling(self):
        config = NegativeSamplingConfig(
            strategy=SamplingStrategy.POPULARITY,
            num_negatives=4,
            seed=42,
        )
        sampler = NegativeSampler(config)
        # Item 1 is very popular, item 100 is rare
        freqs = np.array([100.0] + [1.0] * 99)
        sampler.fit(item_ids=np.arange(1, 101), frequencies=freqs)

        positives = np.array([1, 2])
        negatives = sampler.sample(positives, n_per_positive=100)
        # Item 1 should appear more often than item 100
        unique, counts = np.unique(negatives, return_counts=True)
        assert counts[0] > counts[-1]  # item 1 should be sampled most

    def test_in_batch_sampling(self):
        config = NegativeSamplingConfig(
            strategy=SamplingStrategy.IN_BATCH,
            num_negatives=2,
            seed=42,
        )
        sampler = NegativeSampler(config)
        sampler.fit(item_ids=np.arange(1, 101))

        positives = np.array([10, 20, 30, 40, 50])
        negatives = sampler.sample(positives)
        assert len(negatives) == 10  # 5 * 2

    def test_exclude_positives(self):
        config = NegativeSamplingConfig(
            strategy=SamplingStrategy.UNIFORM,
            num_negatives=4,
            exclude_positives=True,
            seed=42,
        )
        sampler = NegativeSampler(config)
        sampler.fit(item_ids=np.arange(1, 6))  # pool = {1,2,3,4,5}

        positives = np.array([1, 2, 3])
        exclude = np.array([1, 2, 3])
        negatives = sampler.sample(positives, n_per_positive=10, exclude=exclude)
        # Only items 4 and 5 should appear
        assert set(negatives) <= {4, 5}

    def test_sample_per_user(self):
        config = NegativeSamplingConfig(
            strategy=SamplingStrategy.UNIFORM,
            num_negatives=2,
            seed=42,
        )
        sampler = NegativeSampler(config)
        sampler.fit(item_ids=np.arange(1, 11))

        user_pos = {1: [1, 2], 2: [3, 4]}
        result = sampler.sample_per_user(user_pos)
        assert 1 in result
        assert 2 in result
        assert len(result[1]) == 4  # 2 * 2
        assert len(result[2]) == 4

    def test_mixed_sampling(self):
        config = NegativeSamplingConfig(
            strategy=SamplingStrategy.MIXED,
            num_negatives=4,
            seed=42,
        )
        sampler = NegativeSampler(config)
        sampler.fit(item_ids=np.arange(1, 101))
        negatives = sampler.sample(np.array([1, 2, 3]))
        assert len(negatives) == 12

    def test_pool_cache(self, tmp_path):
        cache_path = str(tmp_path / "pool_cache.npz")
        config = NegativeSamplingConfig(
            strategy=SamplingStrategy.UNIFORM,
            cache_path=cache_path,
            seed=42,
        )
        sampler1 = NegativeSampler(config)
        sampler1.fit(item_ids=np.arange(1, 101))

        # Second sampler should load from cache
        sampler2 = NegativeSampler(config)
        sampler2.fit()  # no args, should load from cache
        assert sampler2.n_items == 100


class TestCreateSampler:
    def test_factory(self):
        sampler = create_sampler(strategy="uniform", num_negatives=8, seed=123)
        assert sampler.config.strategy == SamplingStrategy.UNIFORM
        assert sampler.config.num_negatives == 8

    def test_factory_unknown_strategy_falls_back(self):
        sampler = create_sampler(strategy="unknown_strategy_x", num_negatives=4)
        assert sampler.config.strategy == SamplingStrategy.UNIFORM
