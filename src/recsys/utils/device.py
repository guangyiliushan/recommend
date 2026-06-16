"""Device management.

- Auto-detect GPU/CPU
- CUDA availability check
- Mixed precision support detection (fp16/bf16)
- MPS (Apple Silicon) support
- Returns structured DeviceInfo, not just torch.device
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# 错误定义
# ---------------------------------------------------------------------------

class DeviceError(Exception):
    """设备相关错误。"""

    def __init__(
        self,
        message: str,
        code: str = "DEVICE_NOT_AVAILABLE",
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
class DeviceInfo:
    """设备信息，包含 torch.device 与能力探测结果。"""

    device: torch.device
    device_type: str  # "cpu" / "cuda" / "mps"
    cuda_available: bool = False
    num_devices: int = 1
    supports_amp: bool = False
    supports_bf16: bool = False
    gpu_name: Optional[str] = None
    compute_capability: Optional[str] = None
    memory_total_gb: Optional[float] = None

    @property
    def is_cuda(self) -> bool:
        """是否使用 CUDA。"""
        return self.device_type == "cuda"

    @property
    def is_mps(self) -> bool:
        """是否使用 MPS。"""
        return self.device_type == "mps"

    @property
    def is_cpu(self) -> bool:
        """是否使用 CPU。"""
        return self.device_type == "cpu"


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def get_device(device_str: Optional[str] = None) -> DeviceInfo:
    """设备选择与能力探测。

    优先级：
    1. 函数参数 device_str
    2. 环境变量 RECBENCH_DEVICE
    3. 自动探测 ("auto")

    Parameters
    ----------
    device_str : str, optional
        设备字符串：``"cpu"`` / ``"cuda"`` / ``"mps"`` / ``"auto"``。
        为 None 时取环境变量 RECBENCH_DEVICE，仍为 None 时默认 "auto"。

    Returns
    -------
    DeviceInfo

    Raises
    ------
    DeviceError
        设备不合法或请求的设备不可用。
    """
    if device_str is None:
        device_str = os.environ.get("RECBENCH_DEVICE", "auto")

    device_str = device_str.lower().strip()

    if device_str not in {"cpu", "cuda", "mps", "auto"}:
        raise DeviceError(
            f"不支持的设备类型: '{device_str}'",
            code="DEVICE_INVALID",
            hint="支持的值: cpu, cuda, mps, auto",
        )

    # "auto" 模式：优先 CUDA、其次 MPS、最后 CPU
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif _mps_available():
            device_str = "mps"
        else:
            device_str = "cpu"

    return _build_device_info(device_str)


def get_device_info_summary(info: DeviceInfo) -> str:
    """返回人类可读的设备信息摘要。

    Parameters
    ----------
    info : DeviceInfo

    Returns
    -------
    str
    """
    lines = [
        f"Device: {info.device_type.upper()}",
        f"  torch device: {info.device}",
    ]

    if info.is_cuda:
        lines.append(f"  GPU: {info.gpu_name or 'Unknown'}")
        lines.append(f"  Compute Capability: {info.compute_capability or 'Unknown'}")
        lines.append(f"  Memory: {info.memory_total_gb:.1f} GB" if info.memory_total_gb else "  Memory: Unknown")
        lines.append(f"  AMP (fp16): {'Yes' if info.supports_amp else 'No'}")
        lines.append(f"  BF16: {'Yes' if info.supports_bf16 else 'No'}")
        lines.append(f"  Num Devices: {info.num_devices}")
    elif info.is_mps:
        lines.append("  Apple Silicon MPS")
    else:
        lines.append("  CPU only")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _mps_available() -> bool:
    """检查 MPS 是否可用。"""
    if not hasattr(torch.backends, "mps"):
        return False
    try:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except Exception:
        return False


def _build_device_info(device_str: str) -> DeviceInfo:
    """根据设备字符串构建 DeviceInfo。"""
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise DeviceError(
                "请求 CUDA 设备但 CUDA 不可用",
                code="DEVICE_NOT_AVAILABLE",
                hint="设置 device='auto' 或 runtime.device=auto 自动回退到 CPU，"
                     "或检查 CUDA 安装",
            )
        return _build_cuda_info()

    if device_str == "mps":
        if not _mps_available():
            raise DeviceError(
                "请求 MPS 设备但 MPS 不可用",
                code="DEVICE_NOT_AVAILABLE",
                hint="MPS 仅在 Apple Silicon (M1/M2/M3) 上可用，"
                     "设置 device='auto' 自动回退",
            )
        return DeviceInfo(
            device=torch.device("mps"),
            device_type="mps",
            cuda_available=False,
            num_devices=1,
        )

    # CPU
    return DeviceInfo(
        device=torch.device("cpu"),
        device_type="cpu",
        cuda_available=torch.cuda.is_available(),  # 报告 CUDA 可用性即使选了 CPU
        num_devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
    )


def _build_cuda_info() -> DeviceInfo:
    """构建 CUDA 设备的 DeviceInfo。"""
    device = torch.device("cuda")
    num_devices = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if num_devices > 0 else None
    capability = _get_compute_capability()
    memory_gb = _get_gpu_memory_gb()
    supports_amp = _check_amp_support()
    supports_bf16 = _check_bf16_support()

    return DeviceInfo(
        device=device,
        device_type="cuda",
        cuda_available=True,
        num_devices=num_devices,
        supports_amp=supports_amp,
        supports_bf16=supports_bf16,
        gpu_name=gpu_name,
        compute_capability=capability,
        memory_total_gb=memory_gb,
    )


def _get_compute_capability() -> Optional[str]:
    """获取 CUDA 计算能力。"""
    try:
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}.{minor}"
    except Exception:
        return None


def _get_gpu_memory_gb() -> Optional[float]:
    """获取 GPU 总显存（GB）。"""
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_mem / (1024 ** 3)
    except Exception:
        return None


def _check_amp_support() -> bool:
    """检查是否支持 AMP (fp16)。"""
    # CUDA 7.0+ 均支持 fp16 AMP
    try:
        cap = _get_compute_capability()
        if cap is None:
            return False
        major = float(cap.split(".")[0])
        return major >= 7.0
    except Exception:
        return False


def _check_bf16_support() -> bool:
    """检查是否支持 bf16。"""
    # Ampere (SM 8.0+) 支持 bf16
    try:
        cap = _get_compute_capability()
        if cap is None:
            return False
        major = float(cap.split(".")[0])
        return major >= 8.0
    except Exception:
        return False
