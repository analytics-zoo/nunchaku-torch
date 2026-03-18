from __future__ import annotations

import argparse

from .zimage import GenerationConfig, save_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Nunchaku runtime ZImage on CPU, XPU, or CUDA."
    )
    parser.add_argument("--quant-path", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "xpu", "cuda")
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = GenerationConfig(
        quant_path=args.quant_path,
        base_model=args.base_model,
        prompt=args.prompt,
        device=args.device,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )
    out = save_image(config, args.output)
    print(out)


if __name__ == "__main__":
    main()
