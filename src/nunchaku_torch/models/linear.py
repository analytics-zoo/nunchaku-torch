import torch
from torch import nn

from ..ops.gemm import svdq_gemm_w4a4_cuda
from ..ops.gemv import awq_gemv_w4a16_cuda
from ..ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
from ..ops.xpu_ops import is_available as _xpu_kernels_available, svdq_gemm_w4a4_xpu_bf16act

_onednn_prepare = None
def _get_onednn_prepare():
    global _onednn_prepare
    if _onednn_prepare is None:
        try:
            from omni_xpu_kernel.svdq import prepare_onednn_weights
            _onednn_prepare = prepare_onednn_weights
        except (ImportError, AttributeError):
            _onednn_prepare = False
    return _onednn_prepare


class SVDQW4A4Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        act_unsigned: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        super(SVDQW4A4Linear, self).__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.precision = precision
        self.torch_dtype = torch_dtype

        if precision == "nvfp4":
            self.group_size = 16
        elif precision == "int4":
            self.group_size = 64
        else:
            raise ValueError(f"Invalid precision: {precision}")

        self.qweight = nn.Parameter(
            torch.empty(
                out_features, in_features // 2, dtype=torch.int8, device=device
            ),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                torch.empty(out_features, dtype=torch_dtype, device=device),
                requires_grad=True,
            )
            if bias
            else None
        )

        self.wscales = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.smooth_factor = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )
        self.smooth_factor_orig = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )

        self.proj_down = nn.Parameter(
            torch.empty(in_features, rank, dtype=torch_dtype, device=device)
        )
        self.proj_up = nn.Parameter(
            torch.empty(out_features, rank, dtype=torch_dtype, device=device)
        )

        if precision == "nvfp4":
            self.wcscales = nn.Parameter(
                torch.ones(out_features, dtype=torch_dtype, device=device),
                requires_grad=False,
            )
            self.wtscale = 1.0
        else:
            self.wtscale = None
            self.wcscales = None

        self.act_unsigned = act_unsigned

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs):
        in_features = kwargs.pop("in_features", linear.in_features)
        torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)
        device = kwargs.pop("device", linear.weight.device)
        return cls(
            in_features=in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            torch_dtype=torch_dtype,
            device=device,
            **kwargs,
        )

    def forward(
        self, x: torch.Tensor, output: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        x = x.reshape(batch_size * seq_len, channels)
        if output is None:
            output = torch.empty(
                batch_size * seq_len, self.out_features, dtype=x.dtype, device=x.device
            )

        # XPU optimization: skip activation INT4 quantize+dequant round-trip.
        # Instead, compute LoRA down-proj and smooth_factor division on bf16
        # activations directly, then pass to weight-only-dequant GEMM.
        # This is MORE accurate (no quantization noise) and faster (~6s saved).
        if x.device.type == "xpu" and _xpu_kernels_available():
            if not hasattr(self, '_wgt_u4'):
                prepare_fn = _get_onednn_prepare()
                if prepare_fn:
                    u4, sf16 = prepare_fn(self.qweight.view(torch.uint8), self.wscales)
                    self._wgt_u4 = u4
                    self._wscales_f16 = sf16
                    self.qweight.data = torch.empty(0, dtype=torch.int8, device=x.device)
                    self.wscales.data = torch.empty(0, dtype=self.wscales.dtype, device=x.device)
                else:
                    self._wgt_u4 = None
                    self._wscales_f16 = None

            lora_act_out = x @ self.proj_down
            act_smoothed = x / self.smooth_factor
            svdq_gemm_w4a4_xpu_bf16act(
                act_bf16=act_smoothed,
                wgt=self.qweight,
                wscales=self.wscales,
                out=output,
                lora_act_in=lora_act_out,
                lora_up=self.proj_up,
                bias=self.bias,
                alpha=self.wtscale,
                wcscales=self.wcscales,
                wgt_u4=self._wgt_u4,
                wscales_f16=self._wscales_f16,
            )
        else:
            quantized_x, ascales, lora_act_out = self.quantize(x)
            output = self.forward_quant(quantized_x, ascales, lora_act_out, output)

        output = output.reshape(batch_size, seq_len, -1)
        return output

    def quantize(
        self, x: torch.Tensor, pad_size: int = 256
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=self.proj_down,
            smooth=self.smooth_factor,
            fp4=self.precision == "nvfp4",
            pad_size=pad_size,
        )
        return quantized_x, ascales, lora_act_out

    def forward_quant(
        self,
        quantized_x: torch.Tensor,
        ascales: torch.Tensor,
        lora_act: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty(
                quantized_x.shape[0],
                self.out_features,
                dtype=self.proj_up.dtype,
                device=quantized_x.device,
            )

        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=self.qweight,
            out=output,
            ascales=ascales,
            wscales=self.wscales,
            lora_act_in=lora_act,
            lora_up=self.proj_up,
            bias=self.bias,
            fp4=self.precision == "nvfp4",
            alpha=self.wtscale,
            wcscales=self.wcscales,
            act_unsigned=self.act_unsigned,
        )
        return output

    def __repr__(self):
        return (
            f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned})"
        )


class AWQW4A16Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        super(AWQW4A16Linear, self).__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.qweight = nn.Parameter(
            torch.empty(
                out_features // 4, in_features // 2, dtype=torch.int32, device=device
            ),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                torch.empty(out_features, dtype=torch_dtype, device=device),
                requires_grad=True,
            )
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        self.wzeros = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype,
                device=device,
            ),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = awq_gemv_w4a16_cuda(
            in_feats=x,
            kernel=self.qweight,
            scaling_factors=self.wscales,
            zeros=self.wzeros,
            m=x.shape[0],
            n=self.out_features,
            k=self.in_features,
            group_size=self.group_size,
        )
        if self.bias is not None:
            view_shape = [1] * (output.ndim - 1) + [-1]
            output.add_(self.bias.view(view_shape))
        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        **kwargs,
    ):
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            torch_dtype=torch_dtype,
            device=device,
        )

    def __repr__(self):
        return f"AWQW4A16Linear(in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size})"
