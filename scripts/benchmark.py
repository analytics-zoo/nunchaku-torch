#!/usr/bin/env python3
"""End-to-end benchmark for Nunchaku ZImage generation.

Compares performance across devices (cpu / xpu / cuda) and backends
(nunchaku_torch vs. native nunchaku library).

Usage examples:

  # Benchmark nunchaku_torch on XPU, 3 runs, 256×256, 12 steps
  python scripts/benchmark.py \
      --quant-path /path/to/svdq-int4_r256-z-image-turbo.safetensors \
      --base-model /path/to/Z-Image-Turbo \
      --device xpu --runs 3

  # Benchmark on all available devices
  python scripts/benchmark.py \
      --quant-path /path/to/weights.safetensors \
      --base-model /path/to/Z-Image-Turbo \
      --device all

  # Compare native nunchaku (CUDA) vs nunchaku_torch (CUDA)
  python scripts/benchmark.py \
      --quant-path /path/to/weights.safetensors \
      --base-model /path/to/Z-Image-Turbo \
      --device cuda --backend both

  # Quick smoke test (1 run, 64×64, 4 steps)
  python scripts/benchmark.py \
      --quant-path /path/to/weights.safetensors \
      --base-model /path/to/Z-Image-Turbo \
      --device xpu --preset quick
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRESETS = {
    "quick": {"height": 64, "width": 64, "steps": 4, "runs": 1, "warmup": 0},
    "small": {"height": 256, "width": 256, "steps": 9, "runs": 3, "warmup": 1},
    "default": {"height": 1024, "width": 1024, "steps": 9, "runs": 5, "warmup": 2},
    "full": {"height": 1024, "width": 1024, "steps": 9, "runs": 5, "warmup": 2},
}


@dataclass(slots=True)
class BenchmarkConfig:
    quant_path: str
    base_model: str
    prompt: str = "a tiny pixel-art cat sitting on a windowsill"
    device: str = "auto"
    backend: str = "runtime"  # "runtime" | "native" | "both"
    height: int = 1024
    width: int = 1024
    steps: int = 9
    guidance: float = 0.0
    seed: int = 12345
    runs: int = 5
    warmup: int = 2
    save_images: bool = False
    output_dir: str = "benchmark_output"
    output_json: str = ""


@dataclass
class TimingResult:
    backend: str
    device: str
    height: int
    width: int
    steps: int
    warmup_runs: int
    timed_runs: int
    # All times in seconds
    warmup_times: list[float] = field(default_factory=list)
    run_times: list[float] = field(default_factory=list)
    mean: float = 0.0
    median: float = 0.0
    stddev: float = 0.0
    min: float = 0.0
    max: float = 0.0
    throughput_img_per_min: float = 0.0
    peak_memory_mb: float = 0.0


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append("xpu")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _resolve_devices(requested: str) -> list[str]:
    available = _available_devices()
    if requested == "all":
        return available
    if requested == "auto":
        # Pick best available
        for pref in ("cuda", "xpu", "cpu"):
            if pref in available:
                return [pref]
    if requested not in available:
        print(f"ERROR: Device '{requested}' not available. Available: {available}")
        sys.exit(1)
    return [requested]


def _sync_device(device_type: str) -> None:
    """Block until all pending ops on device complete."""
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "xpu":
        torch.xpu.synchronize()


def _peak_memory_mb(device_type: str) -> float:
    """Return peak memory allocated in MB, or 0 for CPU."""
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if device_type == "xpu":
        return torch.xpu.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _reset_memory_stats(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device_type == "xpu":
        torch.xpu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Pipeline loaders
# ---------------------------------------------------------------------------


def _load_runtime_pipeline(
    config: BenchmarkConfig, device_str: str
) -> Any:
    """Load pipeline using nunchaku_torch (our standalone implementation)."""
    from nunchaku_torch.zimage import GenerationConfig, load_pipeline

    gen_config = GenerationConfig(
        quant_path=config.quant_path,
        base_model=config.base_model,
        prompt=config.prompt,
        device=device_str,
        height=config.height,
        width=config.width,
        steps=config.steps,
        guidance=config.guidance,
        seed=config.seed,
    )
    pipe, device, dtype = load_pipeline(gen_config)
    return pipe, device, dtype


def _load_native_pipeline(
    config: BenchmarkConfig, device_str: str
) -> Any:
    """Load pipeline using the native nunchaku library (CUDA only)."""
    try:
        from nunchaku import NunchakuZImageTransformer2DModel
    except ImportError:
        print(
            "ERROR: Native nunchaku library not installed. "
            "Install with: pip install nunchaku"
        )
        sys.exit(1)

    from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

    dtype = torch.bfloat16
    # Try to detect Turing GPUs for float16
    try:
        from nunchaku.utils import is_turing

        if is_turing():
            dtype = torch.float16
    except ImportError:
        pass

    transformer = NunchakuZImageTransformer2DModel.from_pretrained(
        config.quant_path, torch_dtype=dtype
    )
    pipe = ZImagePipeline.from_pretrained(
        config.base_model,
        transformer=transformer,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        local_files_only=True,
    )
    device = torch.device(device_str)
    pipe = pipe.to(device_str)
    return pipe, device, dtype


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------


def _build_gen_kwargs(
    pipe: Any,
    config: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Build generation kwargs (encode prompt + prepare latents once)."""
    do_cfg = config.guidance > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=config.prompt,
        device=device,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=None,
        max_sequence_length=512,
    )

    # Offload text_encoder to CPU — it is not needed during denoising
    from nunchaku_torch.zimage import offload_text_encoder
    offload_text_encoder(pipe)

    num_channels_latents = pipe.transformer.config.in_channels
    generator = torch.Generator(device=str(device)).manual_seed(config.seed)
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=config.height,
        width=config.width,
        dtype=dtype,
        device=str(device),
        generator=generator,
        latents=None,
    )
    return {
        "prompt": None,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "height": config.height,
        "width": config.width,
        "num_inference_steps": config.steps,
        "guidance_scale": config.guidance,
        "generator": None,
        "latents": latents,
    }


def _run_once(pipe: Any, gen_kwargs: dict[str, Any]) -> Any:
    """Run a single generation and return the PIL image."""
    result = pipe(**gen_kwargs)
    images = getattr(result, "images", None)
    if images is None:
        raise RuntimeError("Pipeline did not return images.")
    return images[0]


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------


def _validate_image(image: Image.Image, path: Path) -> None:
    """Quick sanity check on the generated image."""
    arr = np.array(image)
    mean_val = arr.mean()
    std_val = arr.std()
    unique = len(np.unique(arr))
    h, w = arr.shape[:2]
    size_kb = path.stat().st_size / 1024

    ok = True
    issues: list[str] = []

    # Degenerate: all black / all white / constant
    if std_val < 1.0:
        issues.append(f"nearly constant (std={std_val:.1f})")
        ok = False
    # Very few unique values suggests broken output
    if unique < 8:
        issues.append(f"only {unique} unique values")
        ok = False
    # NaN / Inf leakage
    if np.isnan(arr).any() or np.isinf(arr.astype(float)).any():
        issues.append("contains NaN/Inf")
        ok = False

    status = "✓ PASS" if ok else "✗ FAIL"
    print(
        f"  Validation {status}: {path.name}  "
        f"({w}×{h}, {size_kb:.0f}KB, mean={mean_val:.0f}, "
        f"std={std_val:.0f}, unique={unique})"
    )
    if issues:
        for issue in issues:
            print(f"    WARNING: {issue}")


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def run_benchmark(
    config: BenchmarkConfig, device_str: str, backend: str
) -> TimingResult:
    """Run benchmark for one (device, backend) combination."""
    print(f"\n{'='*60}")
    print(f"  Backend: {backend}  |  Device: {device_str}")
    print(f"  Resolution: {config.width}×{config.height}  |  Steps: {config.steps}")
    print(f"  Warmup: {config.warmup}  |  Timed runs: {config.runs}")
    print(f"{'='*60}")

    # --- Load pipeline ---
    print("Loading pipeline...", end=" ", flush=True)
    t0 = time.perf_counter()
    if backend == "native":
        pipe, device, dtype = _load_native_pipeline(config, device_str)
    else:
        pipe, device, dtype = _load_runtime_pipeline(config, device_str)
    load_time = time.perf_counter() - t0
    print(f"done ({load_time:.1f}s)")

    # --- Pre-encode prompt & prepare latents ---
    print("Encoding prompt & preparing latents...", end=" ", flush=True)
    gen_kwargs = _build_gen_kwargs(pipe, config, device, dtype)
    _sync_device(device_str)
    print("done")

    result = TimingResult(
        backend=backend,
        device=device_str,
        height=config.height,
        width=config.width,
        steps=config.steps,
        warmup_runs=config.warmup,
        timed_runs=config.runs,
    )

    # --- Warmup ---
    for i in range(config.warmup):
        print(f"  Warmup {i + 1}/{config.warmup}...", end=" ", flush=True)
        t0 = time.perf_counter()
        _run_once(pipe, gen_kwargs)
        _sync_device(device_str)
        elapsed = time.perf_counter() - t0
        result.warmup_times.append(elapsed)
        print(f"{elapsed:.3f}s")

    # --- Timed runs ---
    _reset_memory_stats(device_str)
    image = None

    for i in range(config.runs):
        print(f"  Run {i + 1}/{config.runs}...", end=" ", flush=True)
        t0 = time.perf_counter()
        image = _run_once(pipe, gen_kwargs)
        _sync_device(device_str)
        elapsed = time.perf_counter() - t0
        result.run_times.append(elapsed)
        print(f"{elapsed:.3f}s")

        if config.save_images:
            out_dir = Path(config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{backend}_{device_str}_{config.width}x{config.height}_run{i}.png"
            image.save(out_dir / fname)

    # --- Save validation image (always, from last run) ---
    if image is not None:
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        val_name = f"validate_{backend}_{device_str}.png"
        val_path = out_dir / val_name
        image.save(val_path)
        _validate_image(image, val_path)

    # --- Stats ---
    result.peak_memory_mb = _peak_memory_mb(device_str)

    if result.run_times:
        result.mean = statistics.mean(result.run_times)
        result.median = statistics.median(result.run_times)
        result.min = min(result.run_times)
        result.max = max(result.run_times)
        result.stddev = (
            statistics.stdev(result.run_times) if len(result.run_times) > 1 else 0.0
        )
        result.throughput_img_per_min = 60.0 / result.mean if result.mean > 0 else 0.0

    # --- Cleanup ---
    del pipe, gen_kwargs
    gc.collect()
    if device_str == "cuda":
        torch.cuda.empty_cache()
    elif device_str == "xpu":
        torch.xpu.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[TimingResult]) -> None:
    print(f"\n{'='*72}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*72}")

    # Header
    header = (
        f"{'Backend':<12} {'Device':<8} {'Resolution':<12} "
        f"{'Steps':<6} {'Mean':>8} {'Median':>8} {'Std':>8} "
        f"{'Min':>8} {'Max':>8} {'img/min':>8} {'Mem(MB)':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        res_str = f"{r.width}×{r.height}"
        mem_str = f"{r.peak_memory_mb:.0f}" if r.peak_memory_mb > 0 else "n/a"
        print(
            f"{r.backend:<12} {r.device:<8} {res_str:<12} "
            f"{r.steps:<6} {r.mean:>7.3f}s {r.median:>7.3f}s {r.stddev:>7.3f}s "
            f"{r.min:>7.3f}s {r.max:>7.3f}s {r.throughput_img_per_min:>7.1f} {mem_str:>10}"
        )

    print(f"{'='*72}")

    # Cross-device speedup comparison
    if len(results) > 1:
        baseline = results[0]
        print(f"\nSpeedup vs {baseline.backend}/{baseline.device}:")
        for r in results[1:]:
            if baseline.mean > 0:
                speedup = baseline.mean / r.mean
                print(f"  {r.backend}/{r.device}: {speedup:.2f}x")


def save_json_report(results: list[TimingResult], path: str) -> None:
    data = []
    for r in results:
        d = {
            "backend": r.backend,
            "device": r.device,
            "resolution": f"{r.width}x{r.height}",
            "steps": r.steps,
            "warmup_runs": r.warmup_runs,
            "timed_runs": r.timed_runs,
            "warmup_times_s": r.warmup_times,
            "run_times_s": r.run_times,
            "mean_s": round(r.mean, 4),
            "median_s": round(r.median, 4),
            "stddev_s": round(r.stddev, 4),
            "min_s": round(r.min, 4),
            "max_s": round(r.max, 4),
            "throughput_img_per_min": round(r.throughput_img_per_min, 2),
            "peak_memory_mb": round(r.peak_memory_mb, 1),
        }
        data.append(d)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nJSON report saved to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Nunchaku ZImage end-to-end generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quant-path",
        required=True,
        help="Path to quantized safetensors weights.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Path to base Z-Image model directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "xpu", "cuda", "all"),
        help="Device to benchmark on. 'all' runs on every available device.",
    )
    parser.add_argument(
        "--backend",
        default="runtime",
        choices=("runtime", "native", "both"),
        help=(
            "'runtime' = nunchaku_torch (this repo), "
            "'native' = original nunchaku library (CUDA only), "
            "'both' = run both for comparison."
        ),
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=tuple(PRESETS.keys()),
        help="Preset config (overrides height/width/steps/runs/warmup).",
    )
    parser.add_argument("--prompt", default="a tiny pixel-art cat sitting on a windowsill")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs.")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs.")
    parser.add_argument("--save-images", action="store_true", help="Save generated images.")
    parser.add_argument("--output-dir", default="benchmark_output")
    parser.add_argument("--output-json", default="", help="Path to save JSON report.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Apply preset if specified
    if args.preset:
        preset = PRESETS[args.preset]
        for k, v in preset.items():
            setattr(args, k, v)

    config = BenchmarkConfig(
        quant_path=args.quant_path,
        base_model=args.base_model,
        prompt=args.prompt,
        device=args.device,
        backend=args.backend,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        runs=args.runs,
        warmup=args.warmup,
        save_images=args.save_images,
        output_dir=args.output_dir,
        output_json=args.output_json,
    )

    devices = _resolve_devices(config.device)
    backends_to_run: list[str] = []
    if config.backend in ("runtime", "both"):
        backends_to_run.append("runtime")
    if config.backend in ("native", "both"):
        backends_to_run.append("native")

    print(f"Devices:  {devices}")
    print(f"Backends: {backends_to_run}")
    print(f"Config:   {config.width}×{config.height}, {config.steps} steps, "
          f"{config.warmup} warmup + {config.runs} timed runs")

    results: list[TimingResult] = []

    for device_str in devices:
        for backend in backends_to_run:
            if backend == "native" and device_str != "cuda":
                print(
                    f"\n  SKIP: native nunchaku only supports CUDA "
                    f"(requested {device_str})"
                )
                continue
            result = run_benchmark(config, device_str, backend)
            results.append(result)

    if results:
        print_report(results)
        if config.output_json:
            save_json_report(results, config.output_json)
    else:
        print("\nNo benchmarks were run.")


if __name__ == "__main__":
    main()
