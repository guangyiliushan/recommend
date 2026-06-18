"""分层级进度追踪工具（环境变量控制，benchmark 友好）。

设计原则：
- RECSYS_PROGRESS 环境变量: 0=静默, 1=进度条, 2=DEBUG日志
- RECSYS_BENCHMARK_MODE 环境变量: 1=强制静默（benchmark 并发场景）
- 轻量级工厂函数，无 tqdm 类继承，benchmark 并发时自动禁用进度条
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


def _progress_level() -> int:
    """0=静默, 1=进度条, 2=DEBUG日志。"""
    return int(os.environ.get("RECSYS_PROGRESS", "0"))


def _is_benchmark() -> bool:
    return os.environ.get("RECSYS_BENCHMARK_MODE", "0") == "1"


class _NullProgress:
    """空进度条（静默/benchmark 模式）。"""

    def update(self, n: int = 1) -> None: ...
    def close(self) -> None: ...


@contextmanager
def progress_phase(name: str, total: Optional[int] = None):
    """阶段级进度上下文管理器。

    在 RECSYS_PROGRESS=1 时显示 tqdm 进度条，
    在 RECSYS_PROGRESS=0 或 benchmark 模式时静默。

    Parameters
    ----------
    name : str
        阶段名称 / 进度条标题。
    total : int, optional
        总步数。为 None 时仅打印开始/完成日志。

    Yields
    ------
    pbar : tqdm-like or _NullProgress
        带 update(n) 和 close() 方法的进度条对象。
    """
    from loguru import logger

    level = _progress_level()
    silent = _is_benchmark() or level == 0
    pbar: Any = _NullProgress()

    if not silent:
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
            if total is not None:
                pbar = tqdm(total=total, desc=name, unit="it", leave=False)
            elif level >= 2:
                logger.debug(f"[开始] {name}")
        except ImportError:
            if level >= 2:
                logger.debug(f"[开始] {name}")

    try:
        yield pbar
    finally:
        if isinstance(pbar, _NullProgress):
            if not silent and level >= 2:
                logger.debug(f"[完成] {name}")
        else:
            pbar.close()


@contextmanager
def phase_timer(name: str, durations: Dict[str, float]):
    """阶段计时器（仅记录耗时，无进度条）。

    Parameters
    ----------
    name : str
        阶段名称（作为 durations 字典的 key）。
    durations : Dict[str, float]
        耗时记录字典（原地修改）。
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        durations[name] = time.perf_counter() - start
