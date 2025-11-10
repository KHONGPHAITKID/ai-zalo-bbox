from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def _default_reference_dir(sample: str | None) -> str:
    if sample is None:
        raise ValueError("Either --refs or --sample must be provided")
    root = Path("data/train/samples") / sample / "object_images"
    if not root.exists():
        raise FileNotFoundError(f"Reference directory not found: {root}")
    return str(root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UAV target object finder pipeline")
    parser.add_argument("--video", required=True, help="Path to UAV video")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--refs", help="Directory or glob of reference images")
    parser.add_argument(
        "--sample",
        help="Sample name under data/train/samples to load references from",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_suffix("").parent / "cfg" / "default.yaml"),
        help="Path to YAML config file",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    refs = args.refs or _default_reference_dir(args.sample)
    run_pipeline(
        video_path=args.video,
        references_root=refs,
        output_path=args.output,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
