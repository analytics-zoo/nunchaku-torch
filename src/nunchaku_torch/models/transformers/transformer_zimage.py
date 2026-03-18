import json
import os
from pathlib import Path
from typing import List, Optional, cast

import torch
import torch.nn as nn

from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_z_image import (
    FeedForward as ZImageFeedForward,
)
from diffusers.models.transformers.transformer_z_image import (
    ZImageTransformer2DModel,
    ZImageTransformerBlock,
)
from huggingface_hub import utils

from ..attention import NunchakuBaseAttention
from ..attention_processors.zimage import NunchakuZSingleStreamAttnProcessor
from ..embeddings import pack_rotemb
from ..linear import SVDQW4A4Linear
from ..unets.unet_sdxl import NunchakuSDXLFeedForward
from ..utils import fuse_linears
from ...ops.gemm import svdq_gemm_w4a4_cuda
from ...ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
from ...ops.xpu_ops import is_available as _xpu_kernels_available, svdq_gemm_w4a4_xpu_bf16act

# XPU ESIMD kernels for fused norm / rotary (optional, from omni_xpu_kernel)
try:
    from omni_xpu_kernel import norm as _esimd_norm, rotary as _esimd_rotary
    _esimd_norm_rotary_available = True
except ImportError:
    _esimd_norm = None
    _esimd_rotary = None
    _esimd_norm_rotary_available = False
from ...utils import get_precision, pad_tensor
from .utils import (
    NunchakuModelLoaderMixin,
    convert_fp16,
    decode_int4_state_dict_for_cpu,
    patch_scale_key,
)

try:
    from ..._C import ops as _C_ops
except ImportError:
    _C_ops = None


def _use_cuda_fused_kernels(device_type: str) -> bool:
    """Return True only when CUDA fused kernels are actually available."""
    return device_type == "cuda" and _C_ops is not None


class NunchakuZImageRopeHook:
    def __init__(self):
        self.packed_cache = {}

    def __call__(self, module: nn.Module, input_args: tuple, input_kwargs: dict):
        freqs_cis = input_kwargs.get("freqs_cis", None)
        if not isinstance(freqs_cis, torch.Tensor):
            return None
        cache_key = freqs_cis.data_ptr()
        packed_freqs_cis = self.packed_cache.get(cache_key, None)
        if packed_freqs_cis is None:
            packed_freqs_cis = torch.view_as_real(freqs_cis).unsqueeze(3)
            packed_freqs_cis = torch.flip(packed_freqs_cis, dims=[-1])
            packed_freqs_cis = pack_rotemb(pad_tensor(packed_freqs_cis, 256, 1))
            self.packed_cache[cache_key] = packed_freqs_cis
        new_input_kwargs = input_kwargs.copy()
        new_input_kwargs["freqs_cis"] = packed_freqs_cis
        return input_args, new_input_kwargs


class NunchakuZImageFusedModule(nn.Module):
    def __init__(self, qkv: SVDQW4A4Linear, norm_q: RMSNorm, norm_k: RMSNorm):
        super().__init__()
        for name, param in qkv.named_parameters(prefix="qkv_"):
            setattr(self, name.replace(".", ""), param)
        self.qkv_precision = qkv.precision
        self.qkv_out_features = qkv.out_features
        self.qkv_in_features = qkv.in_features
        self.qkv_group_size = qkv.group_size
        self.qkv_torch_dtype = qkv.torch_dtype
        for name, param in norm_q.named_parameters(prefix="norm_q_"):
            setattr(self, name.replace(".", ""), param)
        for name, param in norm_k.named_parameters(prefix="norm_k_"):
            setattr(self, name.replace(".", ""), param)

    @staticmethod
    def _apply_rmsnorm(
        x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(variance + eps)
        return (out.to(weight.dtype) * weight).to(x.dtype)

    @staticmethod
    def _apply_rotary_emb(
        x_in: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        # Disable autocast regardless of device type (cpu, cuda, xpu, ...)
        device_type = x_in.device.type if x_in.device.type != "meta" else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            x_out = torch.view_as_real(x * freqs_cis.unsqueeze(2)).flatten(3)
            return x_out.type_as(x_in)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None):
        batch_size, seq_len, channels = x.shape
        norm_q_weight = cast(torch.Tensor, self.norm_q_weight)
        norm_k_weight = cast(torch.Tensor, self.norm_k_weight)

        device_type = x.device.type
        if not _use_cuda_fused_kernels(device_type):
            # --- Non-CUDA path: quantized GEMM + manual norm/rotary ---
            x_2d = x.view(batch_size * seq_len, channels)

            qkv_proj_down = cast(torch.Tensor, self.qkv_proj_down)
            qkv_smooth_factor = cast(torch.Tensor, self.qkv_smooth_factor)
            qkv_qweight = cast(torch.Tensor, self.qkv_qweight)
            qkv_wscales = cast(torch.Tensor, self.qkv_wscales)
            qkv_proj_up = cast(torch.Tensor, self.qkv_proj_up)
            qkv_bias = cast(Optional[torch.Tensor], getattr(self, "qkv_bias", None))
            qkv_wcscales = cast(
                Optional[torch.Tensor], getattr(self, "qkv_wcscales", None)
            )

            output = torch.empty(
                batch_size * seq_len,
                self.qkv_out_features,
                dtype=x.dtype,
                device=x.device,
            )

            # XPU optimization: skip activation INT4 quantize+dequant round-trip
            if device_type == "xpu" and _xpu_kernels_available():
                if not hasattr(self, '_qkv_wgt_u4'):
                    try:
                        from omni_xpu_kernel.svdq import prepare_onednn_weights
                        u4, sf16 = prepare_onednn_weights(qkv_qweight.view(torch.uint8), qkv_wscales)
                        self._qkv_wgt_u4 = u4
                        self._qkv_wscales_f16 = sf16
                        self.qkv_qweight.data = torch.empty(0, dtype=torch.int8, device=x.device)
                        self.qkv_wscales.data = torch.empty(0, dtype=self.qkv_wscales.dtype, device=x.device)
                    except (ImportError, AttributeError):
                        self._qkv_wgt_u4 = None
                        self._qkv_wscales_f16 = None

                lora_act_out = x_2d @ qkv_proj_down
                act_smoothed = x_2d / qkv_smooth_factor
                svdq_gemm_w4a4_xpu_bf16act(
                    act_bf16=act_smoothed,
                    wgt=qkv_qweight,
                    wscales=qkv_wscales,
                    out=output,
                    lora_act_in=lora_act_out,
                    lora_up=qkv_proj_up,
                    bias=qkv_bias,
                    alpha=1.0 if self.qkv_precision == "nvfp4" else None,
                    wcscales=qkv_wcscales if self.qkv_precision == "nvfp4" else None,
                    wgt_u4=self._qkv_wgt_u4,
                    wscales_f16=self._qkv_wscales_f16,
                )
            else:
                quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
                    x_2d,
                    lora_down=qkv_proj_down,
                    smooth=qkv_smooth_factor,
                    fp4=self.qkv_precision == "nvfp4",
                    pad_size=256,
                )
                svdq_gemm_w4a4_cuda(
                    act=quantized_x,
                    wgt=qkv_qweight,
                    out=output,
                    ascales=ascales,
                    wscales=qkv_wscales,
                    lora_act_in=lora_act_out,
                    lora_up=qkv_proj_up,
                    bias=qkv_bias,
                    fp4=self.qkv_precision == "nvfp4",
                    alpha=1.0 if self.qkv_precision == "nvfp4" else None,
                    wcscales=qkv_wcscales if self.qkv_precision == "nvfp4" else None,
                )

            output = output.view(batch_size, seq_len, -1)

            # Split QKV, apply norms and rotary
            query, key, value = output.chunk(3, dim=-1)
            head_dim = int(norm_q_weight.numel())
            if query.shape[-1] % head_dim != 0:
                raise ValueError(
                    f"query dim {query.shape[-1]} is not divisible by head_dim {head_dim}"
                )
            heads = query.shape[-1] // head_dim

            query = query.view(batch_size, seq_len, heads, head_dim)
            key = key.view(batch_size, seq_len, heads, head_dim)
            value = value.view(batch_size, seq_len, heads, head_dim)

            if device_type == "xpu" and _esimd_norm_rotary_available:
                # ESIMD RMSNorm: 3.43x faster than PyTorch
                flat_q = query.reshape(-1, head_dim).contiguous()
                flat_k = key.reshape(-1, head_dim).contiguous()
                query = _esimd_norm.rms_norm(norm_q_weight, flat_q, 1e-6).reshape(
                    batch_size, seq_len, heads, head_dim
                )
                key = _esimd_norm.rms_norm(norm_k_weight, flat_k, 1e-6).reshape(
                    batch_size, seq_len, heads, head_dim
                )

                if freqs_cis is not None:
                    # ESIMD Rotary: 5.61x faster than PyTorch
                    # freqs_cis is [B, S, HD/2] complex64; extract cos/sin as [S, HD/2] f32
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
                query = self._apply_rmsnorm(query, norm_q_weight)
                key = self._apply_rmsnorm(key, norm_k_weight)
                if freqs_cis is not None:
                    query = self._apply_rotary_emb(query, freqs_cis)
                    key = self._apply_rotary_emb(key, freqs_cis)

            output = torch.cat(
                [query.flatten(2, 3), key.flatten(2, 3), value.flatten(2, 3)], dim=-1
            )
            return output

        # --- Original CUDA path (unchanged) ---
        x = x.view(batch_size * seq_len, channels)
        qkv_proj_down = cast(torch.Tensor, self.qkv_proj_down)
        qkv_smooth_factor = cast(torch.Tensor, self.qkv_smooth_factor)
        qkv_qweight = cast(torch.Tensor, self.qkv_qweight)
        qkv_wscales = cast(torch.Tensor, self.qkv_wscales)
        qkv_proj_up = cast(torch.Tensor, self.qkv_proj_up)
        qkv_bias = cast(Optional[torch.Tensor], getattr(self, "qkv_bias", None))
        qkv_wcscales = cast(Optional[torch.Tensor], getattr(self, "qkv_wcscales", None))
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=qkv_proj_down,
            smooth=qkv_smooth_factor,
            fp4=self.qkv_precision == "nvfp4",
            pad_size=256,
        )
        output = torch.empty(
            batch_size * seq_len, self.qkv_out_features, dtype=x.dtype, device=x.device
        )
        # CUDA with _C_ops: fused norm + rotary inside the GEMM kernel
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=qkv_qweight,
            out=output,
            ascales=ascales,
            wscales=qkv_wscales,
            lora_act_in=lora_act_out,
            lora_up=qkv_proj_up,
            bias=qkv_bias,
            fp4=self.qkv_precision == "nvfp4",
            alpha=1.0 if self.qkv_precision == "nvfp4" else None,
            wcscales=qkv_wcscales if self.qkv_precision == "nvfp4" else None,
            norm_q=norm_q_weight,
            norm_k=norm_k_weight,
            rotary_emb=freqs_cis,
        )
        output = output.view(batch_size, seq_len, -1)
        return output


class NunchakuZImageAttention(NunchakuBaseAttention):
    def __init__(self, orig_attn: Attention, processor: str = "flashattn2", **kwargs):
        super(NunchakuZImageAttention, self).__init__(processor)
        self.inner_dim = orig_attn.inner_dim
        self.query_dim = orig_attn.query_dim
        self.use_bias = orig_attn.use_bias
        self.dropout = orig_attn.dropout
        self.out_dim = orig_attn.out_dim
        self.context_pre_only = orig_attn.context_pre_only
        self.pre_only = orig_attn.pre_only
        self.heads = orig_attn.heads
        self.rescale_output_factor = orig_attn.rescale_output_factor
        self.is_cross_attention = orig_attn.is_cross_attention

        self.norm_q = orig_attn.norm_q
        self.norm_k = orig_attn.norm_k
        with torch.device("meta"):
            to_qkv = fuse_linears([orig_attn.to_q, orig_attn.to_k, orig_attn.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        self.to_out = orig_attn.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def set_processor(self, processor: str):
        if processor == "flashattn2":
            self.processor = NunchakuZSingleStreamAttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


def _convert_z_image_ff(z_ff: ZImageFeedForward) -> FeedForward:
    assert isinstance(z_ff, ZImageFeedForward)
    assert z_ff.w1.in_features == z_ff.w3.in_features
    assert z_ff.w1.out_features == z_ff.w3.out_features
    assert z_ff.w1.out_features == z_ff.w2.in_features
    converted_ff = FeedForward(
        dim=z_ff.w1.in_features,
        dim_out=z_ff.w2.out_features,
        dropout=0.0,
        activation_fn="swiglu",
        inner_dim=z_ff.w2.in_features,
        bias=False,
    ).to(dtype=z_ff.w1.weight.dtype, device=z_ff.w1.weight.device)
    return converted_ff


def replace_fused_module(module, incompatible_keys):
    assert isinstance(module, NunchakuZImageAttention)
    module.fused_module = NunchakuZImageFusedModule(
        module.to_qkv, module.norm_q, module.norm_k
    )
    del module.to_qkv
    del module.norm_q
    del module.norm_k


class NunchakuZImageFeedForward(NunchakuSDXLFeedForward):
    def __init__(self, ff: ZImageFeedForward, **kwargs):
        converted_ff = _convert_z_image_ff(ff)
        NunchakuSDXLFeedForward.__init__(self, converted_ff, **kwargs)


class NunchakuZImageTransformer2DModel(
    ZImageTransformer2DModel, NunchakuModelLoaderMixin
):
    def _patch_model(self, skip_refiners: bool = False, **kwargs):
        def _patch_transformer_block(block_list: List[ZImageTransformerBlock]):
            for _, block in enumerate(block_list):
                block.attention = NunchakuZImageAttention(block.attention, **kwargs)
                block.attention.register_load_state_dict_post_hook(replace_fused_module)
                block.feed_forward = NunchakuZImageFeedForward(
                    block.feed_forward, **kwargs
                )

        def _convert_feed_forward(block_list: List[ZImageTransformerBlock]):
            for _, block in enumerate(block_list):
                block.feed_forward = _convert_z_image_ff(block.feed_forward)

        self.skip_refiners = skip_refiners
        _patch_transformer_block(self.layers)
        if skip_refiners:
            _convert_feed_forward(self.noise_refiner)
            _convert_feed_forward(self.context_refiner)
        else:
            _patch_transformer_block(self.noise_refiner)
            _patch_transformer_block(self.context_refiner)
        return self

    def register_rope_hook(self, rope_hook: NunchakuZImageRopeHook):
        self.rope_hook_handles = []
        for _, ly in enumerate(self.layers):
            self.rope_hook_handles.append(
                ly.attention.register_forward_pre_hook(rope_hook, with_kwargs=True)
            )
        if not self.skip_refiners:
            for _, nr in enumerate(self.noise_refiner):
                self.rope_hook_handles.append(
                    nr.attention.register_forward_pre_hook(rope_hook, with_kwargs=True)
                )
            for _, cr in enumerate(self.context_refiner):
                self.rope_hook_handles.append(
                    cr.attention.register_forward_pre_hook(rope_hook, with_kwargs=True)
                )

    def unregister_rope_hook(self):
        for h in self.rope_hook_handles:
            h.remove()
        self.rope_hook_handles.clear()

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        return_dict: bool = True,
    ):
        model_device = next(self.parameters()).device.type
        if not _use_cuda_fused_kernels(model_device):
            # CPU / XPU / any device without _C_ops: standard forward
            # (no rope hook — freqs_cis stays in complex form for manual rotary)
            return super().forward(
                x, t, cap_feats, patch_size, f_patch_size, return_dict
            )

        # CUDA with _C_ops: pack rotary embeddings for fused kernels
        rope_hook = NunchakuZImageRopeHook()
        self.register_rope_hook(rope_hook)
        try:
            return super().forward(
                x, t, cap_feats, patch_size, f_patch_size, return_dict
            )
        finally:
            self.unregister_rope_hook()
            del rope_hook

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs
    ):
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)

        if offload:
            raise NotImplementedError(
                "Offload is not supported for ZImageTransformer2DModel"
            )

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert (
            pretrained_model_name_or_path.is_file()
            or pretrained_model_name_or_path.name.endswith((".safetensors", ".sft"))
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(
            pretrained_model_name_or_path, **kwargs
        )
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))

        rank = quantization_config.get("rank", 32)
        skip_refiners = quantization_config.get("skip_refiners", False)
        transformer = transformer.to(torch_dtype)

        precision = get_precision(
            kwargs.get("precision", "auto"), device, pretrained_model_name_or_path
        )
        if precision == "fp4":
            precision = "nvfp4"

        print(
            f"quantization_config: {quantization_config}, rank={rank}, skip_refiners={skip_refiners}"
        )

        transformer._patch_model(
            skip_refiners=skip_refiners, precision=precision, rank=rank, **kwargs
        )
        transformer = transformer.to_empty(device=device)

        device_type = (
            torch.device(device).type
            if not isinstance(device, torch.device)
            else device.type
        )
        if not _use_cuda_fused_kernels(device_type) and precision == "int4":
            # CPU / XPU / any device without _C_ops:
            # decode weights from CUDA tile layout to row-major for cpu_ops
            decode_int4_state_dict_for_cpu(model_state_dict)

        patch_scale_key(transformer, model_state_dict)
        if torch_dtype == torch.float16:
            convert_fp16(transformer, model_state_dict)

        transformer.load_state_dict(model_state_dict)

        return transformer
