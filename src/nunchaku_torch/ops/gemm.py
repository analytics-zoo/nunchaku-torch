import math

import torch

try:
    from .._C import ops as _C_ops
except ImportError:
    _C_ops = None

from .cpu_ops import svdq_gemm_w4a4_cpu
from .xpu_ops import is_available as _xpu_kernels_available, svdq_gemm_w4a4_xpu


def svdq_gemm_w4a4_cuda(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: torch.Tensor | None = None,
    qout: torch.Tensor | None = None,
    ascales: torch.Tensor | None = None,
    wscales: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    poolout: torch.Tensor | None = None,
    lora_act_in: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    norm_q: torch.Tensor | None = None,
    norm_k: torch.Tensor | None = None,
    rotary_emb: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    smooth_factor: torch.Tensor | None = None,
    out_vk: torch.Tensor | None = None,
    out_linearattn: torch.Tensor | None = None,
    act_unsigned: bool = False,
    lora_scales: list[float] | None = None,
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: float | None = 1.0,
    wcscales: torch.Tensor | None = None,
    out_q: torch.Tensor | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    attn_tokens: int = 0,
    lora_mode: str = "naive",
    lora_up_effective: torch.Tensor | None = None,
    bias_effective: torch.Tensor | None = None,
):
    if lora_scales is None:
        if lora_up is not None:
            rank = lora_up.shape[1]
            lora_scales = [1.0] * math.ceil(rank / 16)
        elif lora_up_effective is not None:
            rank = lora_up_effective.shape[0]
            lora_scales = [1.0] * math.ceil(rank / 16)
        else:
            lora_scales = []
    if alpha is None:
        alpha = 1.0

    if act.device.type == "xpu" and _xpu_kernels_available():
        svdq_gemm_w4a4_xpu(
            act=act,
            wgt=wgt,
            out=out,
            qout=qout,
            ascales=ascales,
            wscales=wscales,
            oscales=oscales,
            poolout=poolout,
            lora_act_in=lora_act_in,
            lora_up=lora_up,
            lora_down=lora_down,
            lora_act_out=lora_act_out,
            norm_q=norm_q,
            norm_k=norm_k,
            rotary_emb=rotary_emb,
            bias=bias,
            smooth_factor=smooth_factor,
            out_vk=out_vk,
            out_linearattn=out_linearattn,
            act_unsigned=act_unsigned,
            lora_scales=lora_scales,
            fuse_silu=fuse_silu,
            fp4=fp4,
            alpha=alpha,
            wcscales=wcscales,
            out_q=out_q,
            out_k=out_k,
            out_v=out_v,
            attn_tokens=attn_tokens,
            lora_mode=lora_mode,
            lora_up_effective=lora_up_effective,
            bias_effective=bias_effective,
        )
        return

    if act.device.type == "cpu" or _C_ops is None:
        svdq_gemm_w4a4_cpu(
            act=act,
            wgt=wgt,
            out=out,
            qout=qout,
            ascales=ascales,
            wscales=wscales,
            oscales=oscales,
            poolout=poolout,
            lora_act_in=lora_act_in,
            lora_up=lora_up,
            lora_down=lora_down,
            lora_act_out=lora_act_out,
            norm_q=norm_q,
            norm_k=norm_k,
            rotary_emb=rotary_emb,
            bias=bias,
            smooth_factor=smooth_factor,
            out_vk=out_vk,
            out_linearattn=out_linearattn,
            act_unsigned=act_unsigned,
            lora_scales=lora_scales,
            fuse_silu=fuse_silu,
            fp4=fp4,
            alpha=alpha,
            wcscales=wcscales,
            out_q=out_q,
            out_k=out_k,
            out_v=out_v,
            attn_tokens=attn_tokens,
            lora_mode=lora_mode,
            lora_up_effective=lora_up_effective,
            bias_effective=bias_effective,
        )
        return

    _C_ops.gemm_w4a4(
        act,
        wgt,
        out,
        qout,
        ascales,
        wscales,
        oscales,
        poolout,
        lora_act_in,
        lora_up,
        lora_down,
        lora_act_out,
        norm_q,
        norm_k,
        rotary_emb,
        bias,
        smooth_factor,
        out_vk,
        out_linearattn,
        act_unsigned,
        lora_scales,
        fuse_silu,
        fp4,
        alpha,
        wcscales,
        out_q,
        out_k,
        out_v,
        attn_tokens,
    )
