"""Unified architectures & Scaling Law (2024–2026): HyFormer, HSTU, InterFormer, OneTrans, HoMer, MTmixAtt.

已实现：
- hyformer (HyFormerAdapter): 通用任务适配 HyFormer 架构，支持 5 种核心任务类型

预留（占位/未实现）：
- hstu.py (HSTU)
- interformer.py (InterFormer)
- onetrans.py (OneTrans)
- homer.py (HoMer)
- mtmixatt.py (MTmixAtt)
- longer.py (Longer)
- wukong.py (WuKong)
"""

from recsys.models.unified.hyformer import HyFormerAdapter  # noqa: F401 — 触发注册

__all__ = ["HyFormerAdapter"]
