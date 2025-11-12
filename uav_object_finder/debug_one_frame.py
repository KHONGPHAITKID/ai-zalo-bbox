from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required for single-frame debugging") from exc

from .config import DEFAULT_CONFIG_PATH, PipelineConfig, load_config
from .embed import build_embedder
from .gallery import build_reference_gallery
from .match import conf_fuse, SimilarityScorer
from .proposals import build_proposal_generator
from .types import Box
from .util import crop_from_boxes


def _default_reference_dir(sample: str | None) -> str:
    if sample is None:
        raise ValueError("Either --refs or --sample must be provided")
    root = Path("data/train/samples") / sample / "object_images"
    if not root.exists():
        raise FileNotFoundError(f"Reference directory not found: {root}")
    return str(root)


def _pick_frame(video_path: Path, frame_idx: Optional[int], seed: Optional[int]) -> tuple[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_idx is None:
        rng = random.Random(seed)
        frame_idx = rng.randint(0, max(total - 1, 0))
    frame_idx = max(0, frame_idx)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame_idx, frame


def _summarize_candidates(candidates: List[Box]) -> List[Dict[str, float]]:
    summary: List[Dict[str, float]] = []
    for box in candidates:
        summary.append(
            {
                "bbox": [float(x) for x in box.xyxy.tolist()],
                "score_det": float(box.score_det),
                "score_app": float(box.score_app),
            }
        )
    return summary


def run_single_frame(
    video_path: str,
    references_root: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    frame_index: Optional[int] = None,
    seed: Optional[int] = None,
    save_image: Optional[str] = None,
) -> Dict[str, object]:
    cfg = load_config(config_path)
    proposal_generator = build_proposal_generator(cfg.proposals)
    embedder = build_embedder(cfg.embedder, cfg.runtime)
    reference_paths = sorted(Path(references_root).glob("*.jpg")) or sorted(Path(references_root).glob("*.png"))
    if not reference_paths:
        raise FileNotFoundError(f"No reference images found under {references_root}")
    gallery = build_reference_gallery(reference_paths, cfg.gallery, embedder, cfg.runtime)
    similarity = SimilarityScorer(cfg.similarity)

    idx, frame = _pick_frame(Path(video_path), frame_index, seed)
    proposals = proposal_generator.generate(frame)
    if cfg.proposals.topk_embed and len(proposals) > cfg.proposals.topk_embed:
        proposals = sorted(proposals, key=lambda b: b.score_det, reverse=True)[: cfg.proposals.topk_embed]
    crops, boxes = crop_from_boxes(frame, proposals)
    embeddings = embedder.encode(crops, cfg.runtime.batch_embed) if crops else np.zeros((0, 1), dtype=np.float32)

    sim_result = similarity.score(embeddings, gallery.embeddings)
    for i, box in enumerate(boxes):
        if i < len(sim_result.sim01):
            box.score_app = float(sim_result.sim01[i])
            box.crop_idx = i

    adaptive_gate = max(cfg.assigner.sim_gate, similarity.threshold())
    candidates = [
        b
        for b in boxes
        if b.score_app >= adaptive_gate or b.score_det >= cfg.proposals.conf_thres + 0.05
    ]
    for cand in candidates:
        cand.score_app = conf_fuse(cand.score_app, cand.score_det)

    if candidates:
        ordered = sorted(candidates, key=lambda b: b.score_app, reverse=True)
        best = ordered[0]
        max_draw = 10
        output = {
            "frame": idx,
            "present": True,
            "conf": float(best.score_app),
            "bbox": [float(x) for x in best.xyxy.tolist()],
            "candidates": _summarize_candidates(ordered[:max_draw]),
        }
        if save_image:
            annotated = frame.copy()
            palette = [
                (0, 255, 0),
                (0, 200, 255),
                (255, 200, 0),
                (200, 0, 200),
                (255, 128, 0),
            ]
            for idx_rank, cand in enumerate(ordered[:max_draw]):
                x1, y1, x2, y2 = np.round(cand.xyxy).astype(int)
                color = palette[idx_rank % len(palette)]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{idx_rank+1}:{cand.score_app:.2f}"
                text_origin = (x1, max(12, y1 - 6))
                cv2.putText(
                    annotated,
                    label,
                    text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )
            cv2.imwrite(save_image, annotated)
    else:
        output = {
            "frame": idx,
            "present": False,
            "conf": 0.0,
            "bbox": None,
            "candidates": [],
        }
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug the pipeline on a single random frame.")
    parser.add_argument("--video", required=True, help="Path to UAV video")
    parser.add_argument("--refs", help="Directory or glob of reference images")
    parser.add_argument("--sample", help="Sample name under data/train/samples to resolve references")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path")
    parser.add_argument("--frame", type=int, help="Specific frame index to debug (otherwise random)")
    parser.add_argument("--seed", type=int, help="Seed for random frame selection")
    parser.add_argument("--output", help="Optional JSON file to store the result")
    parser.add_argument("--save-image", help="Optional path to save the annotated frame")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    refs = args.refs or _default_reference_dir(args.sample)
    result = run_single_frame(
        video_path=args.video,
        references_root=refs,
        config_path=args.config,
        frame_index=args.frame,
        seed=args.seed,
        save_image=args.save_image,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
