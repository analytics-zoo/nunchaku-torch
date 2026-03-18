# nunchaku-torch

Standalone PyTorch runtime for Nunchaku SVDQuant W4A4 inference on CPU, XPU, and CUDA.

## Overview

- Pure Python package — no custom C/CUDA extensions required
- Supports CPU, Intel XPU (via `omni_xpu_kernel` ESIMD/oneDNN kernels), and CUDA backends
- SVDQuant W4A4 quantized model inference with LoRA support
- Currently validated models: ZImage (Z-Image-Turbo), QwenImage

### Performance (Intel XPU, 1024x1024, 9 steps)

| Configuration | Time |
|---|---|
| Original unquantized model | ~10s |
| nunchaku-torch (SVDQuant W4A4) | ~12.35s |

## Install

```bash
# Install PyTorch for your backend first, then:
pip install -e .

# For XPU acceleration, also install omni_xpu_kernel:
cd /path/to/omni_xpu_kernel && pip install -e . --no-build-isolation
```

## CLI

```bash
nunchaku-torch \
  --quant-path /path/to/model.safetensors \
  --base-model /path/to/Z-Image-Turbo \
  --prompt "a realistic photo of a red bicycle" \
  --device auto \
  --height 1024 --width 1024 \
  --steps 9 \
  --output out.png
```

## Python API

```python
from nunchaku_torch import GenerationConfig, generate_image

config = GenerationConfig(
    quant_path="/path/to/model.safetensors",
    base_model="/path/to/Z-Image-Turbo",
    prompt="a tiny pixel-art cat",
    device="xpu",
    height=1024, width=1024,
    steps=9,
)

image = generate_image(config)
image.save("out.png")
```

## Package Structure

```
nunchaku_torch/
  __init__.py          # Lazy-loading public API
  device.py            # Device resolution (cpu/xpu/cuda)
  zimage.py            # ZImage pipeline (GenerationConfig, load_pipeline, generate_image)
  zimage_cli.py        # CLI entry point
  ops/                 # Backend-dispatched ops (CPU fallback, XPU ESIMD/oneDNN, CUDA)
  models/
    linear.py          # SVDQW4A4Linear, AWQW4A16Linear
    attention.py       # Quantized attention and feedforward wrappers
    embeddings.py      # Rotary embeddings
    transformers/      # Model-specific transformer implementations
    attention_processors/  # Model-specific attention processors
    unets/             # UNet implementations (SDXL)
  lora/flux/           # LoRA weight packing utilities
```

## Benchmarking

```bash
# E2E benchmark on XPU
python scripts/benchmark.py --backend runtime --device xpu \
  --quant-path /path/to/model.safetensors \
  --base-model /path/to/Z-Image-Turbo \
  --height 1024 --width 1024 --steps 9

# Validation smoke test
python scripts/validate.py --device xpu \
  --quant-path /path/to/model.safetensors \
  --base-model /path/to/Z-Image-Turbo
```

## Notes

- Self-contained: does not depend on the `nunchaku` PyPI package
- XPU acceleration requires `omni_xpu_kernel` with ESIMD RMSNorm, Rotary, and oneDNN INT4 GEMM kernels
- For multi-GPU setups, use `ZE_AFFINITY_MASK` to select XPU devices
