"""Performance profiling utilities.

Capabilities (progressively optional):
    - Parameter count (total, trainable, per-module) — always available
    - Inference latency measurement (warmup + benchmark) — always available
    - Throughput measurement — always available
    - GPU memory usage tracking — CUDA only
    - FLOPs estimation — requires thop or fvcore (optional)

Design:
    - Optional dependencies degrade gracefully with clear messages
    - All results are structured for JSON artifact writing
    - Profiling is non-destructive: runs on a sample batch, not full training
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# 错误定义
# ---------------------------------------------------------------------------

class ProfilingError(Exception):
    """性能画像错误。"""

    def __init__(
        self,
        message: str,
        code: str = "PROFILING_ERROR",
        phase: str = "profiling",
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
class ParameterInfo:
    """模型参数统计。"""

    total: int = 0
    trainable: int = 0
    non_trainable: int = 0
    by_module: Dict[str, int] = field(default_factory=dict)


@dataclass
class LatencyInfo:
    """推理延迟统计。"""

    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    throughput: float = 0.0  # samples/sec
    num_iters: int = 0
    warmup: int = 0
    batch_size: int = 0


@dataclass
class MemoryInfo:
    """GPU 显存使用。"""

    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    peak_mb: float = 0.0


@dataclass
class ProfilingConfig:
    """性能画像配置。"""

    enabled: bool = True
    measure_flops: bool = True
    measure_memory: bool = True
    num_iters: int = 100
    warmup: int = 10


@dataclass
class ProfilingResult:
    """性能画像完整结果。"""

    parameter_info: ParameterInfo = field(default_factory=ParameterInfo)
    latency_info: Optional[LatencyInfo] = None
    memory_info: Optional[MemoryInfo] = None
    flops: Optional[int] = None
    flops_method: Optional[str] = None
    config: ProfilingConfig = field(default_factory=ProfilingConfig)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> ParameterInfo:
    """统计模型参数数量（按模块汇总）。

    Parameters
    ----------
    model : torch.nn.Module
        要统计的模型。

    Returns
    -------
    ParameterInfo
    """
    total = 0
    trainable = 0
    by_module: Dict[str, int] = {}

    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count

        # 按一级模块汇总
        module_name = name.split(".")[0] if "." in name else "root"
        by_module[module_name] = by_module.get(module_name, 0) + count

    return ParameterInfo(
        total=total,
        trainable=trainable,
        non_trainable=total - trainable,
        by_module=by_module,
    )


def measure_inference_latency(
    model: torch.nn.Module,
    input_sample: Dict[str, Tensor],
    device: torch.device,
    num_iters: int = 100,
    warmup: int = 10,
) -> LatencyInfo:
    """测量模型推理延迟和吞吐。

    Parameters
    ----------
    model : torch.nn.Module
        待测模型（需处于 eval 模式）。
    input_sample : Dict[str, Tensor]
        输入 batch（已移至目标设备）。
    device : torch.device
        目标设备。
    num_iters : int
        测量迭代次数。
    warmup : int
        预热迭代次数。

    Returns
    -------
    LatencyInfo
    """
    model.eval()
    batch_size = _get_batch_size(input_sample)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**input_sample)

    # 同步 CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings: List[float] = []
    with torch.no_grad():
        for _ in range(num_iters):
            start = time.perf_counter()
            _ = model(**input_sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            timings.append(elapsed)

    timings_t = torch.tensor(timings)
    mean_ms = float(timings_t.mean().item())
    std_ms = float(timings_t.std().item())
    throughput = (batch_size * 1000) / mean_ms if mean_ms > 0 else 0.0

    return LatencyInfo(
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=float(timings_t.min().item()),
        max_ms=float(timings_t.max().item()),
        throughput=throughput,
        num_iters=num_iters,
        warmup=warmup,
        batch_size=batch_size,
    )


def get_memory_usage(device: torch.device) -> Optional[MemoryInfo]:
    """获取当前 GPU 显存使用情况。

    仅 CUDA 设备有效。

    Parameters
    ----------
    device : torch.device

    Returns
    -------
    MemoryInfo or None
        非 CUDA 设备返回 None。
    """
    if device.type != "cuda":
        return None

    try:
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    except Exception:
        return None

    return MemoryInfo(
        allocated_mb=round(allocated, 2),
        reserved_mb=round(reserved, 2),
        peak_mb=round(peak, 2),
    )


def profile_model(
    model: torch.nn.Module,
    input_sample: Dict[str, Tensor],
    device: torch.device,
    config: Optional[ProfilingConfig] = None,
) -> ProfilingResult:
    """模型性能画像主入口。

    Parameters
    ----------
    model : torch.nn.Module
        待测模型（需处于 eval 模式）。
    input_sample : Dict[str, Tensor]
        输入 batch（已移至目标设备）。
    device : torch.device
        目标设备。
    config : ProfilingConfig, optional
        画像配置。

    Returns
    -------
    ProfilingResult
    """
    if config is None:
        config = ProfilingConfig()

    result = ProfilingResult(config=config)

    # 参数量统计（始终可用）
    result.parameter_info = count_parameters(model)

    # 推理延迟
    try:
        result.latency_info = measure_inference_latency(
            model, input_sample, device,
            num_iters=config.num_iters,
            warmup=config.warmup,
        )
    except Exception as e:
        result.warnings.append(f"延迟测量失败: {e}")

    # 显存
    if config.measure_memory:
        try:
            mem = get_memory_usage(device)
            if mem is not None:
                result.memory_info = mem
            else:
                result.warnings.append("显存测量仅支持 CUDA 设备")
        except Exception as e:
            result.warnings.append(f"显存测量失败: {e}")

    # FLOPs（可选依赖）
    if config.measure_flops:
        try:
            flops, method = _estimate_flops(model, input_sample)
            result.flops = flops
            result.flops_method = method
        except ProfilingError as e:
            result.warnings.append(str(e))
        except Exception as e:
            result.warnings.append(f"FLOPs 估算失败: {e}")

    _reset_memory_stats(device)

    return result


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _get_batch_size(input_sample: Dict[str, Tensor]) -> int:
    """从输入字典中推断 batch size。"""
    for key in ["user_id", "item_id", "sample_id", "label"]:
        tensor = input_sample.get(key)
        if tensor is not None and tensor.ndim >= 1:
            return tensor.shape[0]
    # 后备：取任意 tensor 的第一维
    for tensor in input_sample.values():
        if isinstance(tensor, Tensor) and tensor.ndim >= 1:
            return tensor.shape[0]
    return 1


def _estimate_flops(
    model: torch.nn.Module,
    input_sample: Dict[str, Tensor],
) -> tuple:
    """估算模型 FLOPs。

    Returns (flops, method_name)。
    """
    # 优先使用 thop
    try:
        from thop import profile  # type: ignore[import-untyped]

        # thop 需要 tuple/list 形式的输入而非 dict
        inputs = tuple(input_sample.values())
        flops, _ = profile(model, inputs=inputs, verbose=False)
        return int(flops), "thop"
    except ImportError:
        pass
    except Exception:
        pass

    # 回退 fvcore
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore[import-untyped]
        flops_analyzer = FlopCountAnalysis(model, input_sample)
        return int(flops_analyzer.total()), "fvcore"
    except ImportError as e:
        raise ProfilingError(
            "FLOPs 估算需要 thop 或 fvcore，均未安装",
            code="PROFILING_DEPENDENCY_MISSING",
            hint="pip install thop 或 pip install fvcore，或设置 profiling.measure_flops=False",
        ) from e
    except Exception as e:
        raise ProfilingError(
            f"FLOPs 估算失败: {e}",
            code="PROFILING_INPUT_INVALID",
            hint="检查 input_sample 是否匹配模型输入",
        ) from e


def _reset_memory_stats(device: torch.device) -> None:
    """重置 CUDA 显存统计。"""
    if device.type == "cuda":
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats(device)
