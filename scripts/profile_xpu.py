"""Profile XPU inference with fine-grained instrumentation of ALL operations.

Instruments:
  - SVDQW4A4Linear.forward (top-level + sub-ops: smooth+lora_down, dequant, matmul, post_proc)
  - NunchakuZImageFusedModule.forward (top-level)
  - FusedModule internals: _apply_rmsnorm, _apply_rotary_emb, chunk+reshape, cat
  - NunchakuZSingleStreamAttnProcessor.__call__ sub-ops: SDPA, flatten, etc.
  - NunchakuSDXLFeedForward.forward (MLP blocks)
"""
from __future__ import annotations
import time
import torch
import sys
import functools

# --- Timing infrastructure ---

timers: dict[str, list[float]] = {}
call_counts: dict[str, int] = {}


def timed(name: str):
    """Decorator that accumulates wall-clock time per function."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            torch.xpu.synchronize()
            dt = time.perf_counter() - t0
            timers.setdefault(name, []).append(dt)
            call_counts[name] = call_counts.get(name, 0) + 1
            return result
        return wrapper
    return decorator


def record(name: str, t0: float):
    """Record elapsed time since t0 (must call torch.xpu.synchronize() before & after)."""
    dt = time.perf_counter() - t0
    timers.setdefault(name, []).append(dt)
    call_counts[name] = call_counts.get(name, 0) + 1


# ============================================================================
# Patch 1: SVDQW4A4Linear.forward — with sub-op profiling
# ============================================================================
from nunchaku_torch.models.linear import SVDQW4A4Linear
from nunchaku_torch.ops import xpu_ops
from omni_xpu_kernel import svdq as _svdq

_orig_bf16act = xpu_ops.svdq_gemm_w4a4_xpu_bf16act


_onednn_available = hasattr(_svdq, 'onednn_int4_gemm_preconverted')


def _profiled_bf16act(
    act_bf16, wgt, wscales, out=None,
    lora_act_in=None, lora_up=None, bias=None,
    lora_scales=None, fuse_silu=False,
    alpha=1.0, wcscales=None,
    lora_mode="naive", lora_up_effective=None, bias_effective=None,
    wgt_u4=None, wscales_f16=None,
):
    input_dtype = act_bf16.dtype
    if _onednn_available and wgt_u4 is not None and wscales_f16 is not None:
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        act_f16 = act_bf16.to(torch.float16) if input_dtype != torch.float16 else act_bf16
        torch.xpu.synchronize()
        record("    act_bf16_to_f16", t0)

        torch.xpu.synchronize()
        t1 = time.perf_counter()
        result = _svdq.onednn_int4_gemm_preconverted(act_f16, wgt_u4, wscales_f16)
        torch.xpu.synchronize()
        record("    onednn_gemm", t1)

        compute_dtype = torch.float16
    else:
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        wgt_deq = _svdq.dequantize_w4(wgt.view(torch.uint8), wscales, input_dtype)
        torch.xpu.synchronize()
        record("    wgt_dequant", t0)

        torch.xpu.synchronize()
        t1 = time.perf_counter()
        result = act_bf16 @ wgt_deq.T
        torch.xpu.synchronize()
        record("    matmul", t1)

        compute_dtype = input_dtype

    torch.xpu.synchronize()
    t3 = time.perf_counter()
    if alpha is not None and alpha != 1.0:
        result = result * alpha
    if wcscales is not None:
        result = result * wcscales.to(compute_dtype).view(1, -1)

    # Compute LoRA residual in bf16
    residual = None
    if lora_act_in is not None:
        lora_act = lora_act_in
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
            residual = lora_act @ lora_up.T
            if bias is not None:
                residual = residual + bias.view(1, -1)
        elif lora_mode == "effective_linear":
            residual = lora_act @ lora_up_effective
            if bias_effective is not None:
                residual = residual + bias_effective.view(1, -1)

    # Write output using in-place ops
    if out is not None:
        M_out, N_out = out.shape
        if fuse_silu:
            result_f32 = result[:M_out, :N_out].float()
            result_f32 = result_f32 * torch.sigmoid(result_f32)
            out.copy_(result_f32)
            if residual is not None:
                out.add_(residual[:M_out, :N_out])
        else:
            out.copy_(result[:M_out, :N_out])
            if residual is not None:
                out.add_(residual[:M_out, :N_out])
            elif bias is not None:
                out.add_(bias.to(out.dtype).view(1, -1))

    torch.xpu.synchronize()
    record("    post_proc", t3)


# Replace the function everywhere it's been imported
xpu_ops.svdq_gemm_w4a4_xpu_bf16act = _profiled_bf16act
from nunchaku_torch.models import linear as linear_mod
linear_mod.svdq_gemm_w4a4_xpu_bf16act = _profiled_bf16act
from nunchaku_torch.models.transformers import transformer_zimage as tz_mod
tz_mod.svdq_gemm_w4a4_xpu_bf16act = _profiled_bf16act


# Detailed SVDQW4A4Linear.forward with smooth+lora_down profiling
def _detailed_linear_forward(self, x, output=None):
    batch_size, seq_len, channels = x.shape
    x = x.reshape(batch_size * seq_len, channels)
    if output is None:
        output = torch.empty(batch_size * seq_len, self.out_features, dtype=x.dtype, device=x.device)

    from nunchaku_torch.ops.xpu_ops import is_available as _xpu_kernels_available
    if x.device.type == "xpu" and _xpu_kernels_available():
        if not hasattr(self, '_wgt_u4'):
            if _onednn_available:
                u4, sf16 = _svdq.prepare_onednn_weights(self.qweight.view(torch.uint8), self.wscales)
                self._wgt_u4 = u4
                self._wscales_f16 = sf16
                self.qweight.data = torch.empty(0, dtype=torch.int8, device=x.device)
                self.wscales.data = torch.empty(0, dtype=self.wscales.dtype, device=x.device)
            else:
                self._wgt_u4 = None
                self._wscales_f16 = None

        torch.xpu.synchronize()
        t0 = time.perf_counter()
        lora_act_out = x @ self.proj_down
        act_smoothed = x / self.smooth_factor
        torch.xpu.synchronize()
        record("    smooth+lora_down", t0)

        _profiled_bf16act(
            act_bf16=act_smoothed,
            wgt=self.qweight, wscales=self.wscales, out=output,
            lora_act_in=lora_act_out, lora_up=self.proj_up,
            bias=self.bias, alpha=self.wtscale, wcscales=self.wcscales,
            wgt_u4=self._wgt_u4,
            wscales_f16=self._wscales_f16,
        )
    else:
        quantized_x, ascales, lora_act_out = self.quantize(x)
        output = self.forward_quant(quantized_x, ascales, lora_act_out, output)

    output = output.reshape(batch_size, seq_len, -1)
    return output


@timed("SVDQW4A4Linear.forward")
def _patched_linear_forward(self, x, output=None):
    return _detailed_linear_forward(self, x, output)

SVDQW4A4Linear.forward = _patched_linear_forward


# ============================================================================
# Patch 2: FusedModule.forward — instrument norm/rotary/chunk/cat
# ============================================================================
from nunchaku_torch.models.transformers.transformer_zimage import (
    NunchakuZImageFusedModule, _use_cuda_fused_kernels,
    _esimd_norm_rotary_available, _esimd_norm, _esimd_rotary,
)
from typing import Optional, cast


def _profiled_fused_forward(self, x: torch.Tensor, freqs_cis=None):
    batch_size, seq_len, channels = x.shape
    norm_q_weight = cast(torch.Tensor, self.norm_q_weight)
    norm_k_weight = cast(torch.Tensor, self.norm_k_weight)

    device_type = x.device.type
    if not _use_cuda_fused_kernels(device_type):
        x_2d = x.view(batch_size * seq_len, channels)

        qkv_proj_down = cast(torch.Tensor, self.qkv_proj_down)
        qkv_smooth_factor = cast(torch.Tensor, self.qkv_smooth_factor)
        qkv_qweight = cast(torch.Tensor, self.qkv_qweight)
        qkv_wscales = cast(torch.Tensor, self.qkv_wscales)
        qkv_proj_up = cast(torch.Tensor, self.qkv_proj_up)
        qkv_bias = cast(Optional[torch.Tensor], getattr(self, "qkv_bias", None))
        qkv_wcscales = cast(Optional[torch.Tensor], getattr(self, "qkv_wcscales", None))

        output = torch.empty(
            batch_size * seq_len, self.qkv_out_features,
            dtype=x.dtype, device=x.device,
        )

        if device_type == "xpu" and tz_mod._xpu_kernels_available():
            # Lazy weight pre-conversion for oneDNN (mirrors transformer_zimage.py)
            if not hasattr(self, '_qkv_wgt_u4'):
                if _onednn_available:
                    u4, sf16 = _svdq.prepare_onednn_weights(
                        qkv_qweight.view(torch.uint8), qkv_wscales
                    )
                    self._qkv_wgt_u4 = u4
                    self._qkv_wscales_f16 = sf16
                    self.qkv_qweight.data = torch.empty(0, dtype=torch.int8, device=x.device)
                    self.qkv_wscales.data = torch.empty(0, dtype=qkv_wscales.dtype, device=x.device)
                else:
                    self._qkv_wgt_u4 = None
                    self._qkv_wscales_f16 = None

            torch.xpu.synchronize()
            t0 = time.perf_counter()
            lora_act_out = x_2d @ qkv_proj_down
            act_smoothed = x_2d / qkv_smooth_factor
            torch.xpu.synchronize()
            record("    smooth+lora_down", t0)

            _profiled_bf16act(
                act_bf16=act_smoothed,
                wgt=qkv_qweight, wscales=qkv_wscales, out=output,
                lora_act_in=lora_act_out, lora_up=qkv_proj_up,
                bias=qkv_bias,
                alpha=1.0 if self.qkv_precision == "nvfp4" else None,
                wcscales=qkv_wcscales if self.qkv_precision == "nvfp4" else None,
                wgt_u4=self._qkv_wgt_u4,
                wscales_f16=self._qkv_wscales_f16,
            )
        else:
            from nunchaku_torch.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
            from nunchaku_torch.ops.gemm import svdq_gemm_w4a4_cuda
            quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
                x_2d, lora_down=qkv_proj_down, smooth=qkv_smooth_factor,
                fp4=self.qkv_precision == "nvfp4", pad_size=256,
            )
            svdq_gemm_w4a4_cuda(
                act=quantized_x, wgt=qkv_qweight, out=output,
                ascales=ascales, wscales=qkv_wscales,
                lora_act_in=lora_act_out, lora_up=qkv_proj_up,
                bias=qkv_bias, fp4=self.qkv_precision == "nvfp4",
                alpha=1.0 if self.qkv_precision == "nvfp4" else None,
                wcscales=qkv_wcscales if self.qkv_precision == "nvfp4" else None,
            )

        output = output.view(batch_size, seq_len, -1)

        # --- Profile chunk + reshape ---
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        query, key, value = output.chunk(3, dim=-1)
        head_dim = int(norm_q_weight.numel())
        heads = query.shape[-1] // head_dim
        query = query.view(batch_size, seq_len, heads, head_dim)
        key = key.view(batch_size, seq_len, heads, head_dim)
        value = value.view(batch_size, seq_len, heads, head_dim)
        torch.xpu.synchronize()
        record("    fused_chunk+reshape", t0)

        # --- Profile RMSNorm ---
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        if device_type == "xpu" and _esimd_norm_rotary_available:
            flat_q = query.reshape(-1, head_dim).contiguous()
            flat_k = key.reshape(-1, head_dim).contiguous()
            query = _esimd_norm.rms_norm(norm_q_weight, flat_q, 1e-6).reshape(
                batch_size, seq_len, heads, head_dim
            )
            key = _esimd_norm.rms_norm(norm_k_weight, flat_k, 1e-6).reshape(
                batch_size, seq_len, heads, head_dim
            )
        else:
            query = NunchakuZImageFusedModule._apply_rmsnorm(query, norm_q_weight)
            key = NunchakuZImageFusedModule._apply_rmsnorm(key, norm_k_weight)
        torch.xpu.synchronize()
        record("    fused_rmsnorm", t0)

        # --- Profile rotary embedding ---
        if freqs_cis is not None:
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            if device_type == "xpu" and _esimd_norm_rotary_available:
                cos_cache = freqs_cis[0].real.contiguous()
                sin_cache = freqs_cis[0].imag.contiguous()
                flat_q = query.reshape(-1, head_dim).contiguous()
                flat_k = key.reshape(-1, head_dim).contiguous()
                query = _esimd_rotary.rotary_emb(
                    flat_q, cos_cache, sin_cache, seq_len, heads
                ).reshape(batch_size, seq_len, heads, head_dim)
                key = _esimd_rotary.rotary_emb(
                    flat_k, cos_cache, sin_cache, seq_len, heads
                ).reshape(batch_size, seq_len, heads, head_dim)
            else:
                query = NunchakuZImageFusedModule._apply_rotary_emb(query, freqs_cis)
                key = NunchakuZImageFusedModule._apply_rotary_emb(key, freqs_cis)
            torch.xpu.synchronize()
            record("    fused_rotary_emb", t0)

        # --- Profile final cat ---
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        output = torch.cat(
            [query.flatten(2, 3), key.flatten(2, 3), value.flatten(2, 3)], dim=-1
        )
        torch.xpu.synchronize()
        record("    fused_cat+flatten", t0)

        return output

    # CUDA path — not instrumented
    return NunchakuZImageFusedModule.forward.__wrapped__(self, x, freqs_cis)


@timed("FusedModule.forward")
def _patched_fused_forward(self, x, freqs_cis=None):
    return _profiled_fused_forward(self, x, freqs_cis)

NunchakuZImageFusedModule.forward = _patched_fused_forward


# ============================================================================
# Patch 3: Attention processor — instrument SDPA and surrounding ops
# ============================================================================
from nunchaku_torch.models.attention_processors.zimage import NunchakuZSingleStreamAttnProcessor
from diffusers.models.attention_dispatch import dispatch_attention_fn

_orig_attn_call = NunchakuZSingleStreamAttnProcessor.__call__


def _profiled_attn_call(self, attn, hidden_states, encoder_hidden_states=None,
                        attention_mask=None, freqs_cis=None):
    # fused_module is already profiled as FusedModule.forward
    qkv = attn.fused_module(hidden_states, freqs_cis)

    # --- chunk + unflatten ---
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    query, key, value = qkv.chunk(3, dim=-1)
    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))
    dtype = query.dtype
    query, key = query.to(dtype), key.to(dtype)

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask[:, None, None, :]
    torch.xpu.synchronize()
    record("    attn_chunk+unflatten", t0)

    # --- SDPA ---
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    hidden_states = dispatch_attention_fn(
        query, key, value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
        parallel_config=self._parallel_config,
    )
    torch.xpu.synchronize()
    record("    attn_SDPA", t0)

    # --- flatten + to_out ---
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.to(dtype)
    torch.xpu.synchronize()
    record("    attn_flatten", t0)

    # to_out is a SVDQW4A4Linear — already profiled
    output = attn.to_out[0](hidden_states)
    if len(attn.to_out) > 1:
        output = attn.to_out[1](output)

    return output

NunchakuZSingleStreamAttnProcessor.__call__ = _profiled_attn_call


# ============================================================================
# Patch 4: FeedForward — instrument the MLP blocks
# ============================================================================
from nunchaku_torch.models.unets.unet_sdxl import NunchakuSDXLFeedForward

_orig_ff_forward = NunchakuSDXLFeedForward.forward

@timed("FeedForward.forward")
def _patched_ff_forward(self, hidden_states):
    return _orig_ff_forward(self, hidden_states)

NunchakuSDXLFeedForward.forward = _patched_ff_forward


# ============================================================================
# Run the pipeline
# ============================================================================
print("Loading pipeline...", flush=True)
from nunchaku_torch.zimage import GenerationConfig, load_pipeline

quant_path = "/llm/workspace/models/svdq-int4_r256-z-image-turbo.safetensors"
base_model = "/llm/models/Z-Image-Turbo"

gen_config = GenerationConfig(
    quant_path=quant_path,
    base_model=base_model,
    prompt="a cat sitting on a windowsill",
    device="xpu",
    height=1024, width=1024,
    steps=9,
    guidance=0.0,
    seed=42,
)
pipe, device_obj, dtype = load_pipeline(gen_config)

print("Running warmup...", flush=True)
with torch.no_grad():
    _ = pipe(
        prompt="a cat sitting on a windowsill",
        height=1024, width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
    ).images[0]
torch.xpu.synchronize()

# Clear timers from warmup
timers.clear()
call_counts.clear()

print("Running profiled inference...", flush=True)
torch.xpu.synchronize()
t_start = time.perf_counter()
with torch.no_grad():
    img = pipe(
        prompt="a cat sitting on a windowsill",
        height=1024, width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
    ).images[0]
torch.xpu.synchronize()
t_total = time.perf_counter() - t_start


# ============================================================================
# Report
# ============================================================================
print(f"\n{'='*80}")
print(f"  DETAILED PROFILED INFERENCE — Total: {t_total:.3f}s")
print(f"{'='*80}")
print(f"{'Component':<35} {'Total(s)':>10} {'Calls':>8} {'Avg(ms)':>10} {'% of total':>10}")
print(f"{'-'*80}")

# Categorize timers by depth (leading spaces)
def depth(name):
    return len(name) - len(name.lstrip())

# Sort: top-level first, then sub-components
top_level = {k: v for k, v in timers.items() if depth(k) == 0}
sub_level = {k: v for k, v in timers.items() if depth(k) > 0}

for name, times in sorted(top_level.items(), key=lambda x: -sum(x[1])):
    total = sum(times)
    count = len(times)
    avg_ms = (total / count) * 1000
    pct = total / t_total * 100
    print(f"{name:<35} {total:>10.3f} {count:>8} {avg_ms:>10.2f} {pct:>9.1f}%")

print(f"{'-'*80}")
for name, times in sorted(sub_level.items(), key=lambda x: -sum(x[1])):
    total = sum(times)
    count = len(times)
    avg_ms = (total / count) * 1000
    pct = total / t_total * 100
    print(f"{name:<35} {total:>10.3f} {count:>8} {avg_ms:>10.2f} {pct:>9.1f}%")

sum_tracked_top = sum(sum(v) for k, v in top_level.items())
sum_tracked_sub = sum(sum(v) for k, v in sub_level.items())
print(f"{'-'*80}")
print(f"{'Tracked (top-level sum)':<35} {sum_tracked_top:>10.3f} {'':>8} {'':>10} {sum_tracked_top/t_total*100:>9.1f}%")
print(f"{'Tracked (sub-ops sum)':<35} {sum_tracked_sub:>10.3f} {'':>8} {'':>10} {sum_tracked_sub/t_total*100:>9.1f}%")
print(f"{'Untracked':<35} {t_total - sum_tracked_top:>10.3f} {'':>8} {'':>10} {(t_total-sum_tracked_top)/t_total*100:>9.1f}%")
print(f"{'TOTAL':<35} {t_total:>10.3f}")

img.save("benchmark_output/profile_test.png")
print(f"\nImage saved to benchmark_output/profile_test.png")

# Memory
mem_mb = torch.xpu.max_memory_allocated(0) / 1e6
print(f"Peak XPU memory: {mem_mb:.0f} MB")
