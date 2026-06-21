"""Classical era models (2001–2015): ItemCF.

已实现：
- itemcf (ItemBasedCF): 非训练式协同过滤基线
- itemcf_backends (分布式计算后端): numpy/dask/modin 后端

预留（占位/未实现）：
- matrix_factorization.py (MF)
- factorization_machine.py (FM)
- gru4rec.py (GRU4Rec)
- model_based_cf.py
- user_based_cf.py
"""

from recsys.models.classical.item_based_cf import ItemBasedCF  # noqa: F401 — 触发注册
from recsys.models.classical.itemcf_backends import (  # noqa: F401
    ComputeBackend,
    DaskBackend,
    ModinBackend,
    NumpyBackend,
    get_compute_backend,
)

__all__ = [
    "ItemBasedCF",
    "ComputeBackend",
    "NumpyBackend",
    "DaskBackend",
    "ModinBackend",
    "get_compute_backend",
]
