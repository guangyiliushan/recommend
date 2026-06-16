"""Classical era models (2001–2015): ItemCF.

已实现：
- itemcf (ItemBasedCF): 非训练式协同过滤基线

预留（占位/未实现）：
- matrix_factorization.py (MF)
- factorization_machine.py (FM)
- gru4rec.py (GRU4Rec)
- model_based_cf.py
- user_based_cf.py
"""

from recsys.models.classical.item_based_cf import ItemBasedCF  # noqa: F401 — 触发注册

__all__ = ["ItemBasedCF"]
