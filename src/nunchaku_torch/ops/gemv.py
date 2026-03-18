import torch

try:
    from .._C import ops as _C_ops
except ImportError:
    _C_ops = None

from .cpu_ops import awq_gemv_w4a16_cpu
from .xpu_ops import is_available as _xpu_kernels_available, awq_gemv_w4a16_xpu


def awq_gemv_w4a16_cuda(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> torch.Tensor:
    if in_feats.device.type == "xpu" and _xpu_kernels_available():
        return awq_gemv_w4a16_xpu(
            in_feats, kernel, scaling_factors, zeros, m, n, k, group_size
        )
    if in_feats.device.type == "cpu" or _C_ops is None:
        return awq_gemv_w4a16_cpu(
            in_feats, kernel, scaling_factors, zeros, m, n, k, group_size
        )
    return _C_ops.gemv_awq(
        in_feats, kernel, scaling_factors, zeros, m, n, k, group_size
    )
