import json
import logging
import os
from pathlib import Path

import torch
from diffusers import __version__
from huggingface_hub import constants, hf_hub_download
from torch import nn

from ...lora.flux.packer import NunchakuWeightPacker
from ...utils import load_state_dict_in_safetensors
from ..linear import SVDQW4A4Linear

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_P = torch.tensor(
    [
        0,
        1,
        2,
        3,
        16,
        17,
        18,
        19,
        32,
        33,
        34,
        35,
        48,
        49,
        50,
        51,
        8,
        9,
        10,
        11,
        24,
        25,
        26,
        27,
        40,
        41,
        42,
        43,
        56,
        57,
        58,
        59,
    ],
    dtype=torch.long,
)


def _build_wgt_src_index() -> torch.Tensor:
    n_local = torch.arange(16, dtype=torch.long).view(16, 1)
    b = torch.arange(32, dtype=torch.long).view(1, 32)
    base = (n_local % 8) * 64 + (n_local // 8) * 4 + _P[b]
    inpack = base % 16
    orig_inpack = inpack
    inpack = torch.where(
        (orig_inpack >= 4) & (orig_inpack < 8), orig_inpack + 4, orig_inpack
    )
    inpack = torch.where(
        (orig_inpack >= 8) & (orig_inpack < 12), orig_inpack - 4, inpack
    )
    return (base // 16) * 16 + inpack


def _build_wscale_src_index() -> torch.Tensor:
    idx = torch.empty(128, dtype=torch.long)
    for lane in range(32):
        src = lane * 4
        a = (lane // 4) * 16 + (lane % 4) * 2
        idx[a + 0] = src + 0
        idx[a + 1] = src + 1
        idx[a + 8] = src + 2
        idx[a + 9] = src + 3
    return idx


_WGT_SRC_INDEX = _build_wgt_src_index()
_WSCALE_SRC_INDEX = _build_wscale_src_index()
_LOWRANK_PACKER_FP16 = NunchakuWeightPacker(bits=16)
_QWEIGHT_BYTE_INDEX_CACHE: dict[
    tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]
] = {}


def _decode_packed_f16_m16n16_tiles_to_rowmajor(
    tensor_2d: torch.Tensor,
) -> torch.Tensor:
    rows, cols = tensor_2d.shape
    if rows % 16 != 0 or cols % 16 != 0:
        raise ValueError(f"tensor shape must be multiples of 16, got ({rows}, {cols})")

    src = tensor_2d.contiguous().view(rows // 16, cols // 16, 32, 8)
    out = torch.empty_like(tensor_2d)

    for rb in range(rows // 16):
        for cb in range(cols // 16):
            pack = src[rb, cb]
            tile = torch.empty(16, 16, dtype=tensor_2d.dtype, device=tensor_2d.device)
            for lane in range(32):
                row_base = lane // 4
                col_base = (lane % 4) * 2
                vals = pack[lane]
                tile[row_base, col_base : col_base + 2] = vals[0:2]
                tile[row_base, col_base + 8 : col_base + 10] = vals[4:6]
                tile[row_base + 8, col_base : col_base + 2] = vals[2:4]
                tile[row_base + 8, col_base + 8 : col_base + 10] = vals[6:8]

            out[rb * 16 : (rb + 1) * 16, cb * 16 : (cb + 1) * 16] = tile

    return out


def _decode_kernel_lora_tile_layout(tensor_2d: torch.Tensor) -> torch.Tensor:
    rows, cols = tensor_2d.shape
    if rows % 16 != 0 or cols % 16 != 0:
        raise ValueError(f"tensor shape must be multiples of 16, got ({rows}, {cols})")

    src = tensor_2d.contiguous().view(rows // 16, cols // 16, 32, 8)
    out = torch.empty_like(tensor_2d)

    for rb in range(rows // 16):
        for cb in range(cols // 16):
            pack = src[rb, cb]
            tile = torch.empty(16, 16, dtype=tensor_2d.dtype, device=tensor_2d.device)
            for lane in range(32):
                row_base = (lane % 4) * 2
                col_base = lane // 4
                vals = pack[lane]
                tile[row_base + 0, col_base + 0] = vals[0]
                tile[row_base + 1, col_base + 0] = vals[1]
                tile[row_base + 8, col_base + 0] = vals[2]
                tile[row_base + 9, col_base + 0] = vals[3]
                tile[row_base + 0, col_base + 8] = vals[4]
                tile[row_base + 1, col_base + 8] = vals[5]
                tile[row_base + 8, col_base + 8] = vals[6]
                tile[row_base + 9, col_base + 8] = vals[7]

            out[rb * 16 : (rb + 1) * 16, cb * 16 : (cb + 1) * 16] = tile

    return out


def decode_lora_updown_cuda_layout_kernel_consumed(
    tensor_2d: torch.Tensor,
) -> torch.Tensor:
    if tensor_2d.ndim != 2:
        raise ValueError(f"tensor_2d must be 2D, got ndim={tensor_2d.ndim}")
    return _decode_kernel_lora_tile_layout(tensor_2d)


def decode_lora_updown_cuda_layout_to_cpu(lora_wgt: torch.Tensor) -> torch.Tensor:
    if lora_wgt.ndim != 2:
        raise ValueError(f"lora_wgt must be 2D, got ndim={lora_wgt.ndim}")
    if lora_wgt.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"lora_wgt must be float16 or bfloat16, got {lora_wgt.dtype}")
    return _LOWRANK_PACKER_FP16.unpack_lowrank_weight(lora_wgt, down=False)


def decode_lora_down_cuda_layout_to_cpu(lora_wgt: torch.Tensor) -> torch.Tensor:
    if lora_wgt.ndim != 2:
        raise ValueError(f"lora_wgt must be 2D, got ndim={lora_wgt.ndim}")
    if lora_wgt.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"lora_wgt must be float16 or bfloat16, got {lora_wgt.dtype}")
    return _LOWRANK_PACKER_FP16.unpack_lowrank_weight(
        lora_wgt, down=True
    ).T.contiguous()


def decode_lora_act_cuda_layout_to_cpu(lora_act: torch.Tensor) -> torch.Tensor:
    if lora_act.ndim != 2:
        raise ValueError(f"lora_act must be 2D, got ndim={lora_act.ndim}")
    m, r = lora_act.shape
    if m % 256 != 0 or r % 16 != 0:
        raise ValueError(f"expected M%256==0 and R%16==0, got M={m}, R={r}")

    src = lora_act.contiguous().view(m // 256, r // 16, 8, 2, 8, 32)
    out = torch.empty_like(lora_act)

    for bm in range(m // 256):
        for rt in range(r // 16):
            for warp in range(8):
                for mt in range(2):
                    pack = src[bm, rt, warp, mt]
                    tile = torch.empty(
                        16, 16, dtype=lora_act.dtype, device=lora_act.device
                    )
                    for lane in range(32):
                        row_base = lane // 4
                        col_base = (lane % 4) * 2
                        vals = pack[:, lane]
                        tile[row_base, col_base : col_base + 2] = vals[0:2]
                        tile[row_base, col_base + 8 : col_base + 10] = vals[4:6]
                        tile[row_base + 8, col_base : col_base + 2] = vals[2:4]
                        tile[row_base + 8, col_base + 8 : col_base + 10] = vals[6:8]

                    row0 = bm * 256 + warp * 32 + mt * 16
                    col0 = rt * 16
                    out[row0 : row0 + 16, col0 : col0 + 16] = tile

    return out


def decode_bias_int4_cuda_layout_to_cpu(bias: torch.Tensor) -> torch.Tensor:
    if bias.ndim != 1:
        raise ValueError(f"bias must be 1D, got ndim={bias.ndim}")
    n_out = int(bias.numel())
    if n_out % 128 != 0:
        raise ValueError(f"bias length must be divisible by 128, got {n_out}")
    blk = bias.contiguous().view(n_out // 128, 128)
    idx = _WSCALE_SRC_INDEX.to(blk.device)
    return blk[:, idx].contiguous().view(n_out)


def decode_wgt_int4_cuda_layout_to_cpu(
    qweight: torch.Tensor, k_total: int
) -> torch.Tensor:
    n_out, width = qweight.shape
    if k_total % 64 != 0:
        raise ValueError(f"k_total={k_total} must be divisible by 64")
    if width * 2 != k_total:
        raise ValueError(f"qweight width mismatch: width={width}, k_total={k_total}")
    if n_out % 128 != 0:
        raise ValueError(f"n_out={n_out} must be divisible by 128")

    src = qweight.contiguous().view(torch.uint8)
    dst = torch.empty_like(src)

    cache_key = (k_total, width, str(src.device))
    if cache_key not in _QWEIGHT_BYTE_INDEX_CACHE:
        groups = k_total // 64
        k_even = torch.arange(0, k_total, 2, dtype=torch.long, device=src.device).view(
            1, -1
        )
        k_odd = k_even + 1
        n_block = torch.arange(128, dtype=torch.long, device=src.device).view(128, 1)
        tile_id = n_block // 16
        n_local = n_block % 16

        def build_lin(k_idx: torch.Tensor) -> torch.Tensor:
            g = k_idx // 64
            b = (k_idx % 64) // 2
            q = (n_local % 8) * 64 + (n_local // 8) * 4 + _P[b]
            inpack = q % 16
            orig_inpack = inpack
            inpack = torch.where(
                (orig_inpack >= 4) & (orig_inpack < 8), orig_inpack + 4, orig_inpack
            )
            inpack = torch.where(
                (orig_inpack >= 8) & (orig_inpack < 12), orig_inpack - 4, inpack
            )
            q = (q // 16) * 16 + inpack
            total = ((g * 8) + tile_id) * 512 + q
            rr = total // width
            cc = total % width
            return rr * width + cc

        _QWEIGHT_BYTE_INDEX_CACHE[cache_key] = (build_lin(k_even), build_lin(k_odd))

    lin_even_base, lin_odd_base = _QWEIGHT_BYTE_INDEX_CACHE[cache_key]
    src_flat = src.view(-1)
    groups = k_total // 64
    block_stride = groups * 8 * 512
    for block_idx in range(n_out // 128):
        block_offset = block_idx * block_stride
        low = src_flat[lin_even_base + block_offset] & 0x0F
        high = (src_flat[lin_odd_base + block_offset] >> 4) & 0x0F
        dst[block_idx * 128 : (block_idx + 1) * 128] = (low | (high << 4)).to(
            torch.uint8
        )

    return dst.view(torch.int8)


def decode_wscale_int4_cuda_layout_to_cpu(wscales: torch.Tensor) -> torch.Tensor:
    groups, n_out = wscales.shape
    if n_out % 128 != 0:
        raise ValueError(f"wscales n_out={n_out} must be divisible by 128")
    out = torch.empty_like(wscales)
    idx = _WSCALE_SRC_INDEX.to(wscales.device)
    src = wscales.contiguous().view(-1)
    num_blocks = n_out // 128

    for g in range(groups):
        for block in range(num_blocks):
            dst_base = block * 128
            src_base = (block * groups + g) * 128
            out[g, dst_base : dst_base + 128] = src[src_base + idx]

    return out


def decode_int4_state_dict_for_cpu(state_dict: dict[str, torch.Tensor]) -> int:
    decoded = 0
    decoded_lora = 0
    decoded_bias = 0
    for qk in list(state_dict.keys()):
        if not qk.endswith(".qweight"):
            continue
        base = qk[: -len(".qweight")]
        wk = f"{base}.wscales"
        sk = f"{base}.smooth_factor"
        uk = f"{base}.proj_up"
        dk = f"{base}.proj_down"
        bk = f"{base}.bias"
        if wk not in state_dict or sk not in state_dict:
            continue

        qweight = state_dict[qk]
        wscales = state_dict[wk]
        smooth = state_dict[sk]

        if qweight.ndim != 2 or wscales.ndim != 2 or smooth.ndim != 1:
            continue

        k_total = int(smooth.numel())
        if k_total == 0 or k_total % 64 != 0:
            continue
        if int(qweight.shape[1]) * 2 != k_total:
            continue
        if int(wscales.shape[0]) != (k_total // 64):
            continue

        state_dict[qk] = decode_wgt_int4_cuda_layout_to_cpu(qweight.cpu(), k_total)
        state_dict[wk] = decode_wscale_int4_cuda_layout_to_cpu(wscales.cpu())
        state_dict[sk] = decode_bias_int4_cuda_layout_to_cpu(smooth.cpu())
        decoded += 1

        if uk in state_dict:
            proj_up = state_dict[uk]
            if proj_up.ndim == 2 and proj_up.shape[0] == qweight.shape[0]:
                state_dict[uk] = decode_lora_updown_cuda_layout_to_cpu(proj_up.cpu())
                decoded_lora += 1

        if dk in state_dict:
            proj_down = state_dict[dk]
            if proj_down.ndim == 2 and proj_down.shape[0] == k_total:
                state_dict[dk] = decode_lora_down_cuda_layout_to_cpu(proj_down.cpu())
                decoded_lora += 1

        if bk in state_dict:
            bias = state_dict[bk]
            if bias.ndim == 1 and bias.numel() == qweight.shape[0]:
                state_dict[bk] = decode_bias_int4_cuda_layout_to_cpu(bias.cpu())
                decoded_bias += 1

    if decoded > 0:
        logger.info(
            f"Decoded INT4 CPU layout for {decoded} linear cores, {decoded_lora} lora up/down tensors, {decoded_bias} bias tensors"
        )
    return decoded


class NunchakuModelLoaderMixin:
    @classmethod
    def _build_model(
        cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs
    ) -> tuple[nn.Module, dict[str, torch.Tensor], dict[str, str]]:
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        state_dict, metadata = load_state_dict_in_safetensors(
            pretrained_model_name_or_path, return_metadata=True
        )

        config = json.loads(metadata["config"])

        with torch.device("meta"):
            transformer = cls.from_config(config).to(
                kwargs.get("torch_dtype", torch.bfloat16)
            )

        return transformer, state_dict, metadata

    @classmethod
    def _build_model_legacy(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[nn.Module, str, str]:
        logger.warning(
            "Loading models from a folder will be deprecated in December 2025. "
            "Please download the latest safetensors model, or use one of the following tools to "
            "merge your model into a single file: the CLI utility `python -m nunchaku.merge_safetensors` "
            "or the ComfyUI workflow `merge_safetensors.json`."
        )
        subfolder = kwargs.get("subfolder", None)
        if os.path.exists(pretrained_model_name_or_path):
            dirname = (
                pretrained_model_name_or_path
                if subfolder is None
                else os.path.join(pretrained_model_name_or_path, subfolder)
            )
            unquantized_part_path = os.path.join(
                dirname, "unquantized_layers.safetensors"
            )
            transformer_block_path = os.path.join(
                dirname, "transformer_blocks.safetensors"
            )
        else:
            download_kwargs = {
                "subfolder": subfolder,
                "repo_type": "model",
                "revision": kwargs.get("revision", None),
                "cache_dir": kwargs.get("cache_dir", None),
                "local_dir": kwargs.get("local_dir", None),
                "user_agent": kwargs.get("user_agent", None),
                "force_download": kwargs.get("force_download", False),
                "proxies": kwargs.get("proxies", None),
                "etag_timeout": kwargs.get(
                    "etag_timeout", constants.DEFAULT_ETAG_TIMEOUT
                ),
                "token": kwargs.get("token", None),
                "local_files_only": kwargs.get("local_files_only", None),
                "headers": kwargs.get("headers", None),
                "endpoint": kwargs.get("endpoint", None),
                "resume_download": kwargs.get("resume_download", None),
                "force_filename": kwargs.get("force_filename", None),
                "local_dir_use_symlinks": kwargs.get("local_dir_use_symlinks", "auto"),
            }
            unquantized_part_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename="unquantized_layers.safetensors",
                **download_kwargs,
            )
            transformer_block_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename="transformer_blocks.safetensors",
                **download_kwargs,
            )

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        config, _, _ = cls.load_config(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            user_agent={
                "diffusers": __version__,
                "file_type": "model",
                "framework": "pytorch",
            },
            **kwargs,
        )

        with torch.device("meta"):
            transformer = cls.from_config(config).to(
                kwargs.get("torch_dtype", torch.bfloat16)
            )
        return transformer, unquantized_part_path, transformer_block_path


def patch_scale_key(
    transformer_from_config: nn.Module, state_dict_from_checkpoint: dict
):
    state_dict = transformer_from_config.state_dict()
    for k in state_dict.keys():
        if k not in state_dict_from_checkpoint:
            assert ".wcscales" in k
            state_dict_from_checkpoint[k] = torch.ones_like(state_dict[k])

    for n, m in transformer_from_config.named_modules():
        if isinstance(m, SVDQW4A4Linear):
            if m.wtscale is not None:
                m.wtscale = state_dict_from_checkpoint.pop(f"{n}.wtscale", 1.0)


def convert_fp16(transformer_from_config: nn.Module, state_dict_from_checkpoint: dict):
    state_dict = transformer_from_config.state_dict()
    for k in state_dict.keys():
        if state_dict[k].dtype != state_dict_from_checkpoint[k].dtype:
            assert (
                state_dict[k].dtype == torch.float16
                and state_dict_from_checkpoint[k].dtype == torch.bfloat16
            ), (
                f"Unexpected dtype difference for key: {k}, model dtype: {state_dict[k].dtype}, checkpoint dtype: {state_dict_from_checkpoint[k].dtype}"
            )
            state_dict_from_checkpoint[k] = torch.nan_to_num(
                state_dict_from_checkpoint[k].to(torch.float16),
                nan=0.0,
                posinf=65504,
                neginf=-65504,
            )
