import torch

from ..utils import ceil_divide


def un_pack_int4(packed: torch.Tensor, signed: bool = True) -> torch.Tensor:
    p = packed.view(torch.uint8).to(torch.int16)
    low = p & 0x0F
    high = (p >> 4) & 0x0F
    if signed:
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.stack([low, high], dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    return unpacked.float()


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    K = values.shape[-1]
    assert K % 2 == 0
    v = values.to(torch.int16)
    even = v[..., 0::2] & 0x0F
    odd = (v[..., 1::2] & 0x0F) << 4
    return (even | odd).to(torch.uint8)


def _dequantize_w4a4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    signed: bool = True,
) -> torch.Tensor:
    unpacked = un_pack_int4(packed, signed=signed)
    N, K = unpacked.shape
    num_groups = K // group_size
    unpacked = unpacked.view(N, num_groups, group_size)
    sc = scales.float().T.unsqueeze(-1)
    return (unpacked * sc).view(N, K)


def _svdq_groupwise_intdot_lowp_accum(
    act_packed: torch.Tensor,
    act_scales: torch.Tensor,
    wgt_packed: torch.Tensor,
    wgt_scales: torch.Tensor,
    act_unsigned: bool = False,
    alpha: float = 1.0,
    wcscales: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    group_size_act = (act_packed.shape[1] * 2) // act_scales.shape[0]
    group_size_wgt = (wgt_packed.shape[1] * 2) // wgt_scales.shape[0]
    if group_size_act != group_size_wgt:
        raise ValueError(
            f"Mismatched group sizes: act={group_size_act}, wgt={group_size_wgt}"
        )

    group_size = group_size_act
    act_i = un_pack_int4(act_packed, signed=not act_unsigned).to(torch.float32)
    wgt_i = un_pack_int4(wgt_packed, signed=True).to(torch.float32)

    M, K = act_i.shape
    N = wgt_i.shape[0]
    num_groups = K // group_size

    # Vectorized grouped dot product — no Python for-loop.
    #
    # Pre-scale each group's values by its per-group scale, then reshape back
    # to [M, K] / [N, K] and do ONE large matmul.  This avoids materializing
    # a [G, M, N] intermediate tensor (which would be huge for large M/N).
    #
    # Mathematically equivalent to:
    #   result[m,n] = Σ_g ( Σ_k act[m,g,k]*wgt[n,g,k] ) * ascale[g,m]*wscale[g,n]
    # Rewritten as:
    #   result[m,n] = Σ_g Σ_k (act[m,g,k]*ascale[g,m]) * (wgt[n,g,k]*wscale[g,n])
    #              = (scaled_act @ scaled_wgt.T)  where scaled_act = act * ascale
    act_3d = act_i.view(M, num_groups, group_size)      # [M, G, gs]
    wgt_3d = wgt_i.view(N, num_groups, group_size)      # [N, G, gs]

    act_scales_f = act_scales.float()   # [G, M]
    wgt_scales_f = wgt_scales.float()   # [G, N]

    # Pre-scale: broadcast [G, M, 1] * [M, G, gs] -> [M, G, gs]
    scaled_act = act_3d * act_scales_f.T.unsqueeze(2)   # act_scales.T is [M, G]
    scaled_wgt = wgt_3d * wgt_scales_f.T.unsqueeze(2)   # wgt_scales.T is [N, G]

    # Single large matmul: [M, K] @ [K, N] -> [M, N]
    result = scaled_act.reshape(M, K) @ scaled_wgt.reshape(N, K).T

    if alpha != 1.0:
        result = result * float(alpha)
    if wcscales is not None:
        result = result * wcscales.float().view(1, -1)
    if bias is not None:
        result = result + bias.float().view(1, -1)

    return result.to(out_dtype)


def compute_lora_bias_residual_cpu(
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor | None,
    bias: torch.Tensor | None,
    lora_mode: str = "naive",
    lora_up_effective: torch.Tensor | None = None,
    bias_effective: torch.Tensor | None = None,
    lora_scales: list[float] | None = None,
) -> torch.Tensor:
    if lora_mode not in ("naive", "effective_linear"):
        raise ValueError(f"Unsupported lora_mode: {lora_mode}")

    lora_act = lora_act_in.float()
    if lora_scales is not None and len(lora_scales) > 0:
        rank = lora_act.shape[1]
        scale_t = torch.ones(rank, dtype=torch.float32, device=lora_act.device)
        for g, s in enumerate(lora_scales):
            start = g * 16
            end = min(start + 16, rank)
            if start >= rank:
                break
            scale_t[start:end] = float(s)
        lora_act = lora_act * scale_t.view(1, -1)

    if lora_mode == "naive":
        if lora_up is None:
            raise ValueError("lora_up is required for lora_mode='naive'")
        residual = lora_act @ lora_up.float().T
        if bias is not None:
            residual = residual + bias.float().view(1, -1)
        return residual

    if lora_up_effective is None:
        raise ValueError(
            "lora_up_effective is required for lora_mode='effective_linear'"
        )

    residual = lora_act @ lora_up_effective.float()
    if bias_effective is not None:
        residual = residual + bias_effective.float().view(1, -1)
    return residual


def svdq_quantize_w4a4_act_fuse_lora_cpu(
    input: torch.Tensor,
    output: torch.Tensor,
    oscales: torch.Tensor,
    lora_down: torch.Tensor | None,
    lora_act_out: torch.Tensor | None,
    smooth: torch.Tensor | None,
    fuse_glu: bool = False,
    fp4: bool = False,
) -> None:
    if fp4:
        raise NotImplementedError(
            "CPU fallback does not support NVFP4 (fp4) quantization"
        )
    if fuse_glu:
        raise NotImplementedError(
            "CPU fallback does not support fused GLU (Gated Linear Unit)"
        )

    M, K = input.shape
    M_pad = output.shape[0]
    group_size = 64

    x = input.float()

    if lora_down is not None and lora_act_out is not None:
        lora_result = x @ lora_down.float()
        lora_act_out[:M].copy_(lora_result)
        if M_pad > M:
            lora_act_out[M:].zero_()

    if smooth is not None:
        x = x / smooth.float()

    num_groups = K // group_size
    x_grouped = x.view(M, num_groups, group_size)

    group_max = x_grouped.abs().amax(dim=-1)
    scales = group_max / 7.0
    scales = scales.clamp(min=1e-10)

    rscale = 7.0 / group_max.clamp(min=1e-10)
    x_q = torch.round(x_grouped * rscale.unsqueeze(-1))
    x_q = x_q.clamp(-8, 7).to(torch.int8).view(M, K)

    packed = _pack_int4(x_q)
    output[:M].copy_(packed)
    if M_pad > M:
        output[M:].zero_()

    oscales[:, :M].copy_(scales.T.to(oscales.dtype))
    if M_pad > M:
        oscales[:, M:].zero_()


def svdq_gemm_w4a4_cpu(
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
) -> None:
    if fp4:
        raise NotImplementedError(
            "CPU fallback does not support NVFP4 (fp4) quantization"
        )
    if ascales is None or wscales is None:
        raise ValueError("ascales and wscales are required")

    result = _svdq_groupwise_intdot_lowp_accum(
        act_packed=act,
        act_scales=ascales,
        wgt_packed=wgt,
        wgt_scales=wscales,
        act_unsigned=act_unsigned,
        alpha=1.0 if alpha is None else alpha,
        wcscales=wcscales,
        bias=None,
        out_dtype=torch.float32,
    )

    if lora_act_in is not None:
        lora_contrib = compute_lora_bias_residual_cpu(
            lora_act_in=lora_act_in,
            lora_up=lora_up,
            bias=bias,
            lora_mode=lora_mode,
            lora_up_effective=lora_up_effective,
            bias_effective=bias_effective,
            lora_scales=lora_scales,
        )
        result = result + lora_contrib
    elif bias is not None:
        result = result + bias.float()

    if fuse_silu:
        result = result * torch.sigmoid(result)

    if out is not None:
        M_out = out.shape[0]
        N_out = out.shape[1]
        out.copy_(result[:M_out, :N_out].to(out.dtype))

    if qout is not None and oscales is not None:
        _quantize_output_for_next_layer(
            result, qout, oscales, lora_down, lora_act_out, smooth_factor
        )


def _quantize_output_for_next_layer(
    result, qout, oscales, lora_down, lora_act_out, smooth_factor
):
    M, N = result.shape
    M_pad = qout.shape[0]
    group_size = 64

    x = result.float()

    if lora_down is not None and lora_act_out is not None:
        lora_act_out[:M].copy_((x @ lora_down.float())[:M])
        if M_pad > M:
            lora_act_out[M:].zero_()

    if smooth_factor is not None:
        x = x / smooth_factor.float()

    num_groups = N // group_size
    x_grouped = x.view(M, num_groups, group_size)
    group_max = x_grouped.abs().amax(dim=-1)
    scales = group_max / 7.0
    scales = scales.clamp(min=1e-10)
    rscale = 7.0 / group_max.clamp(min=1e-10)
    x_q = (
        torch.round(x_grouped * rscale.unsqueeze(-1))
        .clamp(-8, 7)
        .to(torch.int8)
        .view(M, N)
    )

    packed = _pack_int4(x_q)
    qout[:M].copy_(packed)
    if M_pad > M:
        qout[M:].zero_()

    oscales[:, :M].copy_(scales.T.to(oscales.dtype))
    if M_pad > M:
        oscales[:, M:].zero_()


def _svdq_quantize_linear_int4_cpu(
    input: torch.Tensor,
    smooth: torch.Tensor | None = None,
    group_size: int = 64,
    pad_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = input.float()
    if smooth is not None:
        x = x / smooth.float()

    M, K = x.shape
    if K % group_size != 0:
        raise ValueError(f"K={K} must be divisible by group_size={group_size}")

    M_pad = M if pad_to is None else ceil_divide(M, pad_to) * pad_to
    num_groups = K // group_size

    x_work = torch.zeros(M_pad, K, dtype=torch.float32, device=x.device)
    x_work[:M].copy_(x)

    x_grouped = x_work.view(M_pad, num_groups, group_size)
    group_max = x_grouped.abs().amax(dim=-1)
    scales = (group_max / 7.0).clamp(min=1e-10)
    rscale = 7.0 / group_max.clamp(min=1e-10)

    x_q = (
        torch.round(x_grouped * rscale.unsqueeze(-1))
        .clamp(-8, 7)
        .to(torch.int8)
        .view(M_pad, K)
    )
    packed = _pack_int4(x_q)
    return packed, scales.T.contiguous()


def _svdq_quantize_and_downproj_linear_int4_cpu(
    input: torch.Tensor,
    smooth: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    group_size: int = 64,
    pad_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x_raw = input.float()

    lora_act_out = None
    if lora_down is not None:
        lora_act_out = x_raw @ lora_down.float()

    packed, scales = _svdq_quantize_linear_int4_cpu(
        input=input,
        smooth=smooth,
        group_size=group_size,
        pad_to=pad_to,
    )

    if lora_act_out is not None and pad_to is not None:
        M = input.shape[0]
        M_pad = packed.shape[0]
        if M_pad > M:
            pad = torch.zeros(
                M_pad - M,
                lora_act_out.shape[1],
                dtype=lora_act_out.dtype,
                device=lora_act_out.device,
            )
            lora_act_out = torch.cat([lora_act_out, pad], dim=0)

    return packed, scales, lora_act_out


def _svdq_dequant_gemm_a16_cpu(
    act_packed: torch.Tensor,
    act_scales: torch.Tensor,
    wgt_packed: torch.Tensor,
    wgt_scales: torch.Tensor,
    act_unsigned: bool = False,
    alpha: float = 1.0,
    wcscales: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return _svdq_groupwise_intdot_lowp_accum(
        act_packed=act_packed,
        act_scales=act_scales,
        wgt_packed=wgt_packed,
        wgt_scales=wgt_scales,
        act_unsigned=act_unsigned,
        alpha=alpha,
        wcscales=wcscales,
        bias=bias,
        out_dtype=out_dtype,
    )


def _svdq_fused_w4a4_a16_linear_cpu(
    input_fp: torch.Tensor,
    wgt_packed: torch.Tensor,
    wgt_scales: torch.Tensor,
    smooth_factor: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_scales: list[float] | None = None,
    bias: torch.Tensor | None = None,
    alpha: float = 1.0,
    wcscales: torch.Tensor | None = None,
    group_size: int = 64,
    pad_to: int | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict]:
    qact, ascales, lora_act = _svdq_quantize_and_downproj_linear_int4_cpu(
        input=input_fp,
        smooth=smooth_factor,
        lora_down=lora_down,
        group_size=group_size,
        pad_to=pad_to,
    )

    out_main = _svdq_dequant_gemm_a16_cpu(
        act_packed=qact,
        act_scales=ascales,
        wgt_packed=wgt_packed,
        wgt_scales=wgt_scales,
        act_unsigned=False,
        alpha=alpha,
        wcscales=wcscales,
        bias=None,
        out_dtype=torch.float32,
    )

    if lora_act is not None and lora_up is not None:
        lora_res = lora_act.float() @ lora_up.float().T
        if lora_scales is not None:
            channels = lora_up.shape[0]
            scale_t = torch.ones(channels, dtype=torch.float32, device=lora_res.device)
            for g, s in enumerate(lora_scales):
                start = g * 16
                end = min(start + 16, channels)
                scale_t[start:end] = s
            lora_res = lora_res * scale_t
        out_main = out_main + lora_res

    if bias is not None:
        out_main = out_main + bias.float().view(1, -1)

    return out_main.to(out_dtype), {
        "qact": qact,
        "ascales": ascales,
        "lora_act": lora_act,
    }


def _svdq_verify_dequant_gemm_a16_cpu(
    act_fp: torch.Tensor,
    wgt_fp: torch.Tensor,
    act_smooth: torch.Tensor | None = None,
    act_group_size: int = 64,
    wgt_group_size: int = 64,
    act_pad_to: int | None = None,
) -> dict:
    if act_fp.dim() != 2 or wgt_fp.dim() != 2:
        raise ValueError("act_fp and wgt_fp must be rank-2 tensors")
    if act_fp.shape[1] != wgt_fp.shape[1]:
        raise ValueError("act_fp.shape[1] must equal wgt_fp.shape[1]")

    act_q, act_scales = _svdq_quantize_linear_int4_cpu(
        act_fp,
        smooth=act_smooth,
        group_size=act_group_size,
        pad_to=act_pad_to,
    )
    wgt_q, wgt_scales = _svdq_quantize_linear_int4_cpu(
        wgt_fp,
        smooth=None,
        group_size=wgt_group_size,
        pad_to=None,
    )

    pred = _svdq_dequant_gemm_a16_cpu(
        act_q,
        act_scales,
        wgt_q,
        wgt_scales,
        act_unsigned=False,
        out_dtype=torch.float32,
    )

    act_base = act_fp.float()
    if act_smooth is not None:
        act_base = act_base / act_smooth.float()
    target = act_base @ wgt_fp.float().T

    M = act_fp.shape[0]
    pred_valid = pred[:M, : target.shape[1]]
    diff = pred_valid - target
    mse = float((diff * diff).mean().item())
    mae = float(diff.abs().mean().item())
    max_abs = float(diff.abs().max().item())

    return {
        "shape": {
            "M": int(act_fp.shape[0]),
            "K": int(act_fp.shape[1]),
            "N": int(wgt_fp.shape[0]),
        },
        "quant": {
            "act_group_size": int(act_group_size),
            "wgt_group_size": int(wgt_group_size),
            "act_pad_to": None if act_pad_to is None else int(act_pad_to),
            "act_packed_shape": list(act_q.shape),
            "wgt_packed_shape": list(wgt_q.shape),
        },
        "metrics": {
            "mse": mse,
            "mae": mae,
            "max_abs": max_abs,
        },
    }


def awq_unpack_weights(kernel: torch.Tensor, n: int, k: int) -> torch.Tensor:
    qw = kernel.reshape(n, k // 8).to(torch.int64)

    num_groups_32 = k // 32

    qw = qw.reshape(n, num_groups_32, 4)

    shifts = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int64, device=qw.device
    )
    nibbles = (qw.unsqueeze(-1) >> shifts) & 0xF

    u_idx = torch.arange(4, device=qw.device)
    n_idx = torch.arange(8, device=qw.device)
    ic_offsets = (n_idx % 4) * 8 + 2 * u_idx.unsqueeze(-1) + (n_idx // 4)

    weight = torch.zeros(
        n, num_groups_32, 32, dtype=torch.float32, device=kernel.device
    )
    flat_offsets = ic_offsets.reshape(1, 1, 32).expand(n, num_groups_32, 32)
    weight.scatter_(2, flat_offsets, nibbles.reshape(n, num_groups_32, 32).float())

    return weight.reshape(n, k)


def awq_dequantize_weights(
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
) -> torch.Tensor:
    weight = awq_unpack_weights(kernel, n, k)

    num_groups = k // group_size
    weight = weight.view(n, num_groups, group_size)
    sc = scaling_factors.float().T.unsqueeze(-1)
    zp = zeros.float().T.unsqueeze(-1)
    dequant = weight * sc + zp
    return dequant.view(n, k)


def awq_gemv_w4a16_cpu(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> torch.Tensor:
    weight = awq_dequantize_weights(kernel, scaling_factors, zeros, n, k, group_size)
    x = in_feats.float().reshape(m, k)
    output = x @ weight.T
    return output.to(in_feats.dtype)
