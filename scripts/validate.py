from __future__ import annotations

import argparse

from nunchaku_torch.zimage import GenerationConfig, save_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-validate standalone ZImage generation."
    )
    parser.add_argument("--quant-path", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--device", default="cpu", choices=("cpu", "xpu", "cuda", "auto")
    )
    parser.add_argument("--prompt", default="a tiny pixel-art cat")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", default="validate.png")
    args = parser.parse_args()

    config = GenerationConfig(
        quant_path=args.quant_path,
        base_model=args.base_model,
        prompt=args.prompt,
        device=args.device,
        height=args.height,
        width=args.width,
        steps=args.steps,
    )
    out = save_image(config, args.output)
    print(out)


if __name__ == "__main__":
    main()
