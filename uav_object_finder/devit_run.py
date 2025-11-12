from __future__ import annotations

import argparse
from pathlib import Path

from .devit_pipeline import run_devit_pipeline


def _default_reference_dir(sample: str | None) -> str:
    if sample is None:
        raise ValueError("Either --refs or --sample must be provided")
    root = Path("data/train/samples") / sample / "object_images"
    if not root.exists():
        raise FileNotFoundError(f"Reference directory not found: {root}")
    return str(root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DE-ViT UAV object localizer")
    parser.add_argument("--video", required=True, help="Path to UAV video")
    parser.add_argument("--refs", help="Directory or glob of reference images")
    parser.add_argument("--sample", help="Sample name under data/train/samples to resolve references")
    parser.add_argument("--out-dir", default="video_inference_results/devit", help="Directory for CSV/video outputs")
    parser.add_argument("--device", default="cuda:0", help="Torch device for DE-ViT + DINOv2")
    parser.add_argument("--conf", default=0.25, type=float, help="Confidence threshold for detections")
    parser.add_argument("--iou-nms", default=0.5, type=float, help="IoU threshold for per-frame NMS")
    parser.add_argument("--batch", default=8, type=int, help="Frames per DE-ViT forward pass")
    parser.add_argument("--iou-track", default=0.3, type=float, help="Tracker IoU/feature matching threshold")
    parser.add_argument("--feat-w", default=0.0, type=float, help="Weight for feature cosine during matching")
    parser.add_argument("--max-age", default=30, type=int, help="Frames to keep unmatched tracks alive")
    parser.add_argument("--overlay-fps", default=0.0, type=float, help="Override FPS for the overlay video (0 = inherit)")
    parser.add_argument("--overlay", help="Explicit overlay MP4 path (defaults to <out-dir>/overlay.mp4)")
    parser.add_argument("--csv", help="Explicit CSV path (defaults to <out-dir>/detections.csv)")
    parser.add_argument("--label-name", default="target", help="Class label name for DE-ViT prototypes")
    parser.add_argument("--topk", default=1, type=int, help="Top-K proposals to keep in DE-ViT RPN")
    parser.add_argument("--mask-on", action="store_true", help="Enable mask branch during inference")
    parser.add_argument("--proto-model", default="dinov2_vitl14", help="Torch hub identifier for support prototype encoder (e.g. dinov2_vits14)")
    parser.add_argument("--config-file", default="configs/open-vocabulary/lvis/vitl.yaml", help="DE-ViT config file")
    parser.add_argument("--rpn-config-file", default="configs/RPN/mask_rcnn_R_50_FPN_1x.yaml", help="Offline RPN config")
    parser.add_argument("--model-path", default="weights/trained/open-vocabulary/lvis/vitl_0069999.pth", help="Checkpoint path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    refs = args.refs or _default_reference_dir(args.sample)
    overlay_fps = None if args.overlay_fps <= 0 else args.overlay_fps
    out_dir = Path(args.out_dir)
    if args.sample:
        out_dir = out_dir / args.sample
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.overlay or str(out_dir / "overlay.mp4")
    csv_path = args.csv or str(out_dir / "detections.csv")
    overlay, csv_file = run_devit_pipeline(
        video_path=args.video,
        supports=refs,
        out_dir=str(out_dir),
        device=args.device,
        conf_thr=args.conf,
        iou_nms=args.iou_nms,
        batch_size=args.batch,
        iou_track=args.iou_track,
        feat_w=args.feat_w,
        max_age=args.max_age,
        overlay_fps=overlay_fps,
        overlay_path=overlay_path,
        csv_path=csv_path,
        config_file=args.config_file,
        rpn_config_file=args.rpn_config_file,
        model_path=args.model_path,
        label_name=args.label_name,
        topk=args.topk,
        mask_on=args.mask_on,
        proto_model=args.proto_model,
    )
    print(f"Saved overlay video to {overlay}")
    print(f"Saved detections CSV to {csv_file}")


if __name__ == "__main__":
    main()
