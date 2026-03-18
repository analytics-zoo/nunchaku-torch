from __future__ import annotations

import torch


SUPPORTED_DEVICES = ("auto", "cpu", "xpu", "cuda")


def has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def resolve_device(preferred: str = "auto") -> torch.device:
    if preferred not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device '{preferred}'. Expected one of {SUPPORTED_DEVICES}."
        )

    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if has_xpu():
            return torch.device("xpu")
        return torch.device("cpu")

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if preferred == "xpu":
        if not has_xpu():
            raise RuntimeError("Intel XPU was requested but is not available.")
        return torch.device("xpu")

    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "xpu":
        return torch.bfloat16
    return torch.float32
