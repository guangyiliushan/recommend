"""Deterministic training settings.

- Set random seed (Python, NumPy, PyTorch, CUDA)
- Enable/disable cuDNN benchmark
- Deterministic mode with warn_only fallback
- Returns structured info for artifact writing
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 错误定义
# ---------------------------------------------------------------------------

class ReproducibilityError(Exception):
    """可复现性相关错误。"""

    def __init__(
        self,
        message: str,
        code: str = "REPRODUCIBILITY_ERROR",
        phase: str = "runtime",
        hint: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.phase = phase
        self.hint = hint


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SeedInfo:
    """种子设置结果摘要。"""

    seed: int
    python_random: bool = True
    numpy: bool = True
    torch: bool = True
    cuda: bool = False


@dataclass
class DeterministicInfo:
    """确定性模式设置结果摘要。"""

    enabled: bool
    warn_only: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = False
    torch_deterministic: bool = False


# ---------------------------------------------------------------------------
# 状态管理
# ---------------------------------------------------------------------------

_seed_info: Optional[SeedInfo] = None
_deterministic_info: Optional[DeterministicInfo] = None


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> SeedInfo:
    """设置全局随机种子。

    覆盖 Python random、NumPy、PyTorch（含 CUDA）。

    Parameters
    ----------
    seed : int
        随机种子，必须为非负整数。

    Returns
    -------
    SeedInfo
        实际生效配置摘要。

    Raises
    ------
    ReproducibilityError
        seed 不合法。
    """
    global _seed_info

    if not isinstance(seed, int) or seed < 0:
        raise ReproducibilityError(
            f"seed 必须为非负整数，当前: {seed}",
            code="SEED_INVALID",
            hint="seed 应为 0 或正整数",
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_seeded = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True

    _seed_info = SeedInfo(
        seed=seed,
        python_random=True,
        numpy=True,
        torch=True,
        cuda=cuda_seeded,
    )

    return _seed_info


def deterministic_mode(
    enabled: bool = True,
    warn_only: bool = False,
) -> DeterministicInfo:
    """启用或禁用确定性模式。

    统一管理：
    - torch.use_deterministic_algorithms
    - torch.backends.cudnn.deterministic
    - torch.backends.cudnn.benchmark

    Parameters
    ----------
    enabled : bool
        是否启用确定性模式，默认 True。
    warn_only : bool
        当算子不支持确定性算法时，是仅警告 (True) 还是抛出异常 (False)。

    Returns
    -------
    DeterministicInfo

    Raises
    ------
    ReproducibilityError
        启用确定性模式但算子不支持且 warn_only=False。
    """
    global _deterministic_info

    if not enabled:
        _deterministic_info = DeterministicInfo(enabled=False)
        return _deterministic_info

    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except RuntimeError as e:
        raise ReproducibilityError(
            f"无法启用确定性模式: {e}",
            code="DETERMINISTIC_NOT_SUPPORTED",
            hint="设置 warn_only=True 跳过不支持的操作，或更换模型结构",
        ) from e

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 额外：禁用 CUDA 卷积自动调优
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "matmul") and hasattr(
        torch.backends.cuda.matmul, "allow_tf32"
    ):
        torch.backends.cuda.matmul.allow_tf32 = False

    _deterministic_info = DeterministicInfo(
        enabled=True,
        warn_only=warn_only,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        torch_deterministic=True,
    )

    return _deterministic_info


def get_reproducibility_summary() -> Dict[str, Any]:
    """返回当前可复现配置摘要。

    用于写入 status.json 和日志。

    Returns
    -------
    Dict[str, Any]
    """
    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
    }

    if _seed_info is not None:
        info["seed"] = _seed_info.seed
        info["seed_cuda"] = _seed_info.cuda

    if _deterministic_info is not None:
        info["deterministic"] = _deterministic_info.enabled
        info["deterministic_warn_only"] = _deterministic_info.warn_only

    # CUDA 版本信息
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = _get_cudnn_version()

    return info


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _get_cudnn_version() -> Optional[str]:
    """获取 cuDNN 版本。"""
    try:
        return str(torch.backends.cudnn.version())
    except Exception:
        return None
