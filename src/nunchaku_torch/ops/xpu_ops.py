"""
XPU-native ops for SVDQuant W4A4 inference using ESIMD kernels from omni_xpu_kernel.

These functions replace cpu_ops on XPU devices, using fused dequantize_w4
(unpack INT4 + per-group scale in a single ESIMD kernel) instead of the
separate unpack → groupwise-intdot path.

GEMM strategy:
    cpu_ops:  unpack_int4(act) * unpack_int4(wgt) → grouped intdot with scales
    xpu_ops:  oneDNN INT4 fused dequant+GEMM (f16 activations, u4 weights)
              Falls back to ESIMD dequant + bf16 matmul if oneDNN unavailable.
"""

import torch

try:
    from omni_xpu_kernel import svdq as _svdq
    _xpu_available = True
    try:
        _svdq.onednn_int4_gemm_preconverted
        _onednn_available = True
    except AttributeError:
        _onednn_available = False
    try:
        _svdq.fused_convert_add
        _fused_convert_add_available = True
    except AttributeError:
        _fused_convert_add_available = False
    try:
        _svdq.onednn_int4_gemm_add_to_output
        _append_sum_available = True
    except AttributeError:
        _append_sum_available = False
except ImportError:
    _svdq = None
    _xpu_available = False
    _onednn_available = False
    _fused_convert_add_available = False
    _append_sum_available = False


def is_available() -> bool:
    """Check whether XPU ESIMD kernels are available."""
    return _xpu_available


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------

def svdq_quantize_w4a4_act_fuse_lora_xpu(
    input: torch.Tensor,
    output: torch.Tensor,
    oscales: torch.Tensor,
    lora_down: torch.Tensor | None,
    lora_act_out: torch.Tensor | None,
    smooth: torch.Tensor | None,
    fuse_glu: bool = False,
    fp4: bool = False,
) -> None:
    """XPU-native activation quantization using ESIMD quantize_act_int4 kernel."""
    if fp4:
        raise NotImplementedError(
            "XPU kernels do not support NVFP4 (fp4) quantization"
        )
    if fuse_glu:
        raise NotImplementedError(
            "XPU kernels do not support fused GLU (Gated Linear Unit)"
        )

    M, K = input.shape
    M_pad = output.shape[0]

    x = input.float()

    # LoRA down-projection (still in PyTorch — no ESIMD kernel for this)
    if lora_down is not None and lora_act_out is not None:
        lora_result = x @ lora_down.float()
        lora_act_out[:M].copy_(lora_result)
        if M_pad > M:
            lora_act_out[M:].zero_()

    # Apply smooth factor before quantization
    if smooth is not None:
        x = x / smooth.float()

    # ESIMD quantize_act_int4: input [M, K] → (packed [M, K/2], scales [G, M])
    packed, scales = _svdq.quantize_act_int4(x, group_size=64)

    output[:M].copy_(packed)
    if M_pad > M:
        output[M:].zero_()

    # scales from ESIMD kernel are [G, M], oscales expects [G, M_pad] with dtype match
    oscales[:, :M].copy_(scales.to(oscales.dtype))
    if M_pad > M:
        oscales[:, M:].zero_()


# ---------------------------------------------------------------------------
# GEMM  (W4A4)
# ---------------------------------------------------------------------------

def svdq_gemm_w4a4_xpu(
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
    """
    XPU-native GEMM using ESIMD dequantize_w4 + bf16 matmul.

    Strategy:
        1. Dequantize weights:  ESIMD dequantize_w4([N, K/2], [G, N]) → [N, K] bf16
        2. Dequantize activations: ESIMD dequantize_w4([M, K/2], [G, M]) → [M, K] bf16
        3. Matmul: act_deq @ wgt_deq.T → [M, N] result
        4. Apply alpha, wcscales, LoRA, bias, SiLU as needed
        5. Optionally re-quantize output for next layer (fused GEMM chain)
    """
    if fp4:
        raise NotImplementedError(
            "XPU kernels do not support NVFP4 (fp4) quantization"
        )
    if ascales is None or wscales is None:
        raise ValueError("ascales and wscales are required")

    # --- Step 1: Dequantize weights via ESIMD kernel (fused unpack + scale) ---
    wgt_deq = _svdq.dequantize_w4(wgt.view(torch.uint8), wscales, torch.bfloat16)
    # wgt_deq: [N, K] bf16

    # --- Step 2: Dequantize activations via ESIMD kernel ---
    # ascales is [G, M], act is [M, K/2] packed
    act_deq = _svdq.dequantize_w4(act.view(torch.uint8), ascales, torch.bfloat16)
    # act_deq: [M, K] bf16

    # --- Step 3: Matmul ---
    result = (act_deq @ wgt_deq.T).float()
    # result: [M, N] float32

    # --- Step 4: Post-processing ---
    if alpha is not None and alpha != 1.0:
        result = result * float(alpha)
    if wcscales is not None:
        result = result * wcscales.float().view(1, -1)

    # LoRA residual
    if lora_act_in is not None:
        from .cpu_ops import compute_lora_bias_residual_cpu
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

    # --- Step 5: Write outputs ---
    if out is not None:
        M_out = out.shape[0]
        N_out = out.shape[1]
        out.copy_(result[:M_out, :N_out].to(out.dtype))

    # Re-quantize output for fused GEMM chain (e.g., fused_gelu_mlp)
    if qout is not None and oscales is not None:
        _quantize_output_for_next_layer_xpu(
            result, qout, oscales, lora_down, lora_act_out, smooth_factor
        )


def svdq_gemm_w4a4_xpu_bf16act(
    act_bf16: torch.Tensor,
    wgt: torch.Tensor,
    wscales: torch.Tensor,
    out: torch.Tensor | None = None,
    lora_act_in: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    lora_scales: list[float] | None = None,
    fuse_silu: bool = False,
    alpha: float | None = 1.0,
    wcscales: torch.Tensor | None = None,
    lora_mode: str = "naive",
    lora_up_effective: torch.Tensor | None = None,
    bias_effective: torch.Tensor | None = None,
    wgt_u4: torch.Tensor | None = None,
    wscales_f16: torch.Tensor | None = None,
) -> None:
    """
    XPU optimized GEMM that skips activation quantization entirely.

    Instead of:  bf16 act → quantize to INT4 → dequantize back to bf16 → matmul
    We do:       bf16 act → convert to f16 → oneDNN fused INT4 dequant+GEMM

    When oneDNN is available and pre-converted weights (wgt_u4, wscales_f16) are
    provided, uses the fast oneDNN f16×u4 path which is ~1.54x faster than
    separate ESIMD dequant + bf16 matmul.

    Args:
        act_bf16: [M, K] bf16 activations (already divided by smooth_factor)
        wgt: [N, K/2] packed INT4 weights (used for fallback path)
        wscales: [G, N] weight scales bf16 (used for fallback path)
        out: [M, N] output tensor (required)
        wgt_u4: [N, K/2] pre-converted unsigned u4 weights (packed ^ 0x88)
        wscales_f16: [G, N] pre-converted f16 weight scales
        lora_act_in, lora_up, bias, etc: standard post-processing params
    """
    # --- Compute LoRA residual in bf16 (needed before GEMM for append_sum path) ---
    input_dtype = act_bf16.dtype
    residual = None
    if lora_act_in is not None:
        lora_act = lora_act_in  # already in input_dtype (bf16)
        if lora_scales is not None and len(lora_scales) > 0:
            rank = lora_act.shape[1]
            scale_t = torch.ones(rank, dtype=input_dtype, device=lora_act.device)
            for g, s in enumerate(lora_scales):
                start = g * 16
                end = min(start + 16, rank)
                if start >= rank:
                    break
                scale_t[start:end] = s
            lora_act = lora_act * scale_t.view(1, -1)

        if lora_mode == "naive":
            if lora_up is None:
                raise ValueError("lora_up is required for lora_mode='naive'")
            residual = lora_act @ lora_up.T  # bf16 matmul — safe from overflow
            if bias is not None:
                residual = residual + bias.view(1, -1)
        elif lora_mode == "effective_linear":
            if lora_up_effective is None:
                raise ValueError("lora_up_effective is required for lora_mode='effective_linear'")
            residual = lora_act @ lora_up_effective  # bf16 matmul
            if bias_effective is not None:
                residual = residual + bias_effective.view(1, -1)
        else:
            raise ValueError(f"Unsupported lora_mode: {lora_mode}")

    # --- Check if we can use the append_sum fast path ---
    # append_sum: pre-fill out with residual, then GEMM adds result directly.
    # Eliminates separate fused_convert_add kernel. Saves ~0.17s per image.
    # Requirements: oneDNN path, residual exists, no SiLU, no alpha/wcscales scaling.
    _use_append_sum = (
        _append_sum_available
        and _onednn_available
        and wgt_u4 is not None
        and wscales_f16 is not None
        and residual is not None
        and out is not None
        and not fuse_silu
        and (alpha is None or alpha == 1.0)
        and wcscales is None
    )

    if _use_append_sum:
        # --- append_sum fast path: out = residual + GEMM(act, wgt) in one primitive ---
        if input_dtype == torch.float16:
            act_f16 = act_bf16
        else:
            act_f16 = act_bf16.to(torch.float16)
        out.copy_(residual)
        _svdq.onednn_int4_gemm_add_to_output(act_f16, wgt_u4, wscales_f16, out)
        return

    # --- GEMM: oneDNN fused path (f16) or fallback (ESIMD dequant + matmul) ---
    if _onednn_available and wgt_u4 is not None and wscales_f16 is not None:
        if input_dtype == torch.float16:
            act_f16 = act_bf16  # already f16 — no conversion needed
        else:
            act_f16 = act_bf16.to(torch.float16)
        result = _svdq.onednn_int4_gemm_preconverted(act_f16, wgt_u4, wscales_f16)
        # Keep result in f16 — do post-processing in f16 to avoid conversion overhead.
        # Convert to output dtype only at the end (via out.copy_).
        compute_dtype = torch.float16
    else:
        wgt_deq = _svdq.dequantize_w4(wgt.view(torch.uint8), wscales, input_dtype)
        result = act_bf16 @ wgt_deq.T
        compute_dtype = input_dtype

    # --- Post-processing ---
    # GEMM result is in compute_dtype (f16 for oneDNN, input_dtype for fallback).
    # Alpha/wcscales are safe in f16 (small values), but LoRA residuals can exceed
    # f16 max (65504) so LoRA must be computed in bf16.
    #
    # Performance: Using in-place ops (copy_ + add_) is ~4.7x faster than creating
    # temporary tensors (.to(bf16) → new tensor + add → new tensor + copy_).
    # We write the GEMM result directly to `out` via copy_ (which handles f16→bf16),
    # then add LoRA residual and bias in-place.
    if alpha is not None and alpha != 1.0:
        result = result * alpha
    if wcscales is not None:
        result = result * wcscales.to(compute_dtype).view(1, -1)

    # --- Write output using in-place ops (4.7x faster than separate .to + add + copy) ---
    if out is not None:
        M_out = out.shape[0]
        N_out = out.shape[1]
        if fuse_silu:
            # SiLU needs float32 precision — can't do in-place on bf16 out
            result_f32 = result[:M_out, :N_out].float()
            result_f32 = result_f32 * torch.sigmoid(result_f32)
            out.copy_(result_f32)
            if residual is not None:
                out.add_(residual[:M_out, :N_out])
        else:
            # Fast path: fused_convert_add does f16→bf16 + add in single ESIMD kernel
            # ~2x faster than separate copy_ + add_ (saves ~0.68s per image)
            if residual is not None and _fused_convert_add_available:
                _svdq.fused_convert_add(out, result, residual)
            else:
                # copy_ handles f16→bf16 conversion implicitly
                out.copy_(result[:M_out, :N_out])
                if residual is not None:
                    out.add_(residual[:M_out, :N_out])
                elif bias is not None:
                    out.add_(bias.to(out.dtype).view(1, -1))
    elif fuse_silu:
        # No output tensor — modify result in-place (rare path)
        if compute_dtype != input_dtype:
            result = result.to(input_dtype)
        if residual is not None:
            result = result + residual
        result = result.float()
        result = result * torch.sigmoid(result)
        result = result.to(input_dtype)


def _quantize_output_for_next_layer_xpu(
    result: torch.Tensor,
    qout: torch.Tensor,
    oscales: torch.Tensor,
    lora_down: torch.Tensor | None,
    lora_act_out: torch.Tensor | None,
    smooth_factor: torch.Tensor | None,
) -> None:
    """Re-quantize GEMM output for the next quantized layer using ESIMD kernel."""
    M, N = result.shape
    M_pad = qout.shape[0]

    x = result.float()

    # LoRA down-projection for next layer
    if lora_down is not None and lora_act_out is not None:
        lora_act_out[:M].copy_((x @ lora_down.float())[:M])
        if M_pad > M:
            lora_act_out[M:].zero_()

    # Apply smooth factor
    if smooth_factor is not None:
        x = x / smooth_factor.float()

    # ESIMD quantize_act_int4: [M, N] → ([M, N/2], [G, M])
    packed, scales = _svdq.quantize_act_int4(x, group_size=64)

    qout[:M].copy_(packed)
    if M_pad > M:
        qout[M:].zero_()

    oscales[:, :M].copy_(scales.to(oscales.dtype))
    if M_pad > M:
        oscales[:, M:].zero_()


# ---------------------------------------------------------------------------
# AWQ GEMV (W4A16) — no ESIMD kernel, delegate to cpu_ops
# ---------------------------------------------------------------------------

def awq_gemv_w4a16_xpu(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> torch.Tensor:
    """AWQ GEMV on XPU — uses cpu_ops (PyTorch native ops work on XPU)."""
    from .cpu_ops import awq_gemv_w4a16_cpu
    return awq_gemv_w4a16_cpu(
        in_feats, kernel, scaling_factors, zeros, m, n, k, group_size
    )
