from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter
import torch
from rfdetr import RFDETRBase

try:
    import cv2
except ImportError as exc:  # pragma: no cover - hard requirement for video I/O
    raise ImportError("OpenCV (cv2) is required to run this script.") from exc

from similarity import (
    RMACFeatureExtractor,
    ReferenceFeature,
    crop_bbox,
    list_reference_image_paths,
    load_reference_features,
    rank_similarities,
)


DATA_ROOTS = {
    "train": Path("data/train/samples"),
    "public": Path("data/public_test/samples"),
}
DEFAULT_VIDEO_NAME = "drone_video.mp4"

MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.0"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.13"))
MAX_DETECTIONS_PER_FRAME = int(os.getenv("MAX_DETECTIONS_PER_FRAME", "15"))
SIMILARITY_TOP_K = 1

DENOISE_RADIUS = float(os.getenv("DENOISE_RADIUS", "0"))
CLAHE_ENABLED = os.getenv("CLAHE_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
CLAHE_CLIP_LIMIT = float(os.getenv("CLAHE_CLIP_LIMIT", "2.5"))
_clahe_grid = os.getenv("CLAHE_TILE_GRID_SIZE", "8,8").split(",")
if len(_clahe_grid) != 2:
    CLAHE_TILE_GRID_SIZE = (8, 8)
else:
    CLAHE_TILE_GRID_SIZE = (
        max(1, int(_clahe_grid[0].strip() or 8)),
        max(1, int(_clahe_grid[1].strip() or 8)),
    )
del _clahe_grid

CLAHE_OPERATOR = (
    cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    if CLAHE_ENABLED and CLAHE_CLIP_LIMIT > 0
    else None
)

VIDEO_CODEC = os.getenv("VIDEO_CODEC", "mp4v")
DEFAULT_OUTPUT_DIR = Path(os.getenv("ANNOTATED_VIDEO_DIR", "video_annotation_outputs"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the submission annotation pipeline on a single video and "
            "export an annotated preview."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATA_ROOTS.keys()),
        help="Dataset root to use (defaults to interactive selection)",
    )
    parser.add_argument(
        "--sample",
        help="Sample folder name (defaults to interactive selection)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Full path to a video file (overrides dataset/sample selection)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory for annotated videos (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-json",
        action="store_true",
        help="Do not write the JSON annotation payload",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra progress details",
    )
    return parser.parse_args()


def maybe_denoise_image(image: Image.Image) -> Image.Image:
    if DENOISE_RADIUS <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=DENOISE_RADIUS))


def maybe_apply_clahe(image: Image.Image) -> Image.Image:
    if CLAHE_OPERATOR is None:
        return image
    working = image if image.mode == "RGB" else image.convert("RGB")
    np_image = np.array(working)
    if np_image.ndim != 3 or np_image.shape[2] != 3:
        return working
    bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_enhanced = CLAHE_OPERATOR.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    rgb_enhanced = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_enhanced)


def filter_detections(detections):
    if len(detections) == 0:
        return detections
    conf = detections.confidence
    mask = conf >= CONFIDENCE_THRESHOLD
    if not np.any(mask):
        mask[:] = True
    filtered = detections[mask]
    if len(filtered) <= MAX_DETECTIONS_PER_FRAME:
        return filtered
    top_indices = np.argsort(filtered.confidence)[-MAX_DETECTIONS_PER_FRAME:]
    top_indices.sort()
    return filtered[top_indices]


def compute_similarity_scores(
    feature_extractor: RMACFeatureExtractor,
    image: Image.Image,
    detections,
    references: Sequence[ReferenceFeature],
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    if not references or len(detections) == 0:
        return scores
    for idx, bbox in enumerate(detections.xyxy):
        crop = crop_bbox(image, bbox)
        if crop is None:
            continue
        feature = feature_extractor.extract(crop)
        matches = rank_similarities(feature, references, SIMILARITY_TOP_K)
        if matches:
            scores[idx] = matches[0].similarity
    return scores


def select_detection_idx(detections, similarity_scores: Dict[int, float]) -> Optional[int]:
    if len(detections) == 0:
        return None
    if similarity_scores:
        return max(similarity_scores.items(), key=lambda item: item[1])[0]
    return int(np.argmax(detections.confidence))


def prepare_references(
    feature_extractor: RMACFeatureExtractor, samples_root: Path, sample_name: str
) -> List[ReferenceFeature]:
    reference_paths = list_reference_image_paths(samples_root, sample_name)
    if not reference_paths:
        return []
    feature_extractor.reset()
    feature_extractor.fine_tune(reference_paths)
    return load_reference_features(samples_root, sample_name, feature_extractor)


def frame_entry(frame_idx: int, bbox) -> dict:
    x1, y1, x2, y2 = [int(round(float(val))) for val in bbox]
    return {"frame": frame_idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def discover_videos(samples_root: Path) -> List[Path]:
    if not samples_root.exists():
        return []
    return sorted(
        path for path in samples_root.glob(f"*/{DEFAULT_VIDEO_NAME}") if path.is_file()
    )


def prompt_choice(options: Iterable[str], label: str) -> str:
    indexed = list(sorted(set(options)))
    if not indexed:
        raise ValueError(f"No options available for {label}")
    print(f"\nAvailable {label}:")
    for idx, name in enumerate(indexed, 1):
        print(f"  [{idx}] {name}")
    while True:
        choice = input(f"Select {label} (1-{len(indexed)}): ").strip()
        if not choice:
            continue
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(indexed):
                return indexed[idx - 1]
        matches = [name for name in indexed if name.lower() == choice.lower()]
        if matches:
            return matches[0]
        print("Invalid choice, please try again.")


def pick_dataset(dataset_arg: Optional[str]) -> Tuple[str, Path]:
    available = {name: root for name, root in DATA_ROOTS.items() if root.exists()}
    if not available:
        raise FileNotFoundError(
            "No dataset roots found. Expected at least one of "
            f"{', '.join(str(root) for root in DATA_ROOTS.values())}"
        )
    if dataset_arg:
        if dataset_arg not in available:
            raise FileNotFoundError(
                f"Dataset '{dataset_arg}' not available. "
                f"Existing roots: {', '.join(available.keys())}"
            )
        return dataset_arg, available[dataset_arg]
    if len(available) == 1:
        (name, root) = next(iter(available.items()))
        print(f"Auto-selecting dataset '{name}' at {root}")
        return name, root
    name = prompt_choice(available.keys(), "datasets")
    return name, available[name]


def pick_video(args: argparse.Namespace) -> Tuple[Path, Optional[Path], str]:
    if args.video:
        if not args.video.exists():
            raise FileNotFoundError(f"Video not found: {args.video}")
        return args.video.resolve(), None, args.video.parent.name

    dataset_name, dataset_root = pick_dataset(args.dataset)
    videos = discover_videos(dataset_root)
    if not videos:
        raise FileNotFoundError(
            f"No '{DEFAULT_VIDEO_NAME}' files found under {dataset_root}"
        )

    if args.sample:
        matches = [v for v in videos if v.parent.name == args.sample]
        if not matches:
            available = ", ".join(v.parent.name for v in videos)
            raise FileNotFoundError(
                f"Sample '{args.sample}' not found in {dataset_root}. "
                f"Available samples: {available}"
            )
        video = matches[0]
    else:
        sample_name = prompt_choice([v.parent.name for v in videos], "samples")
        video = next(v for v in videos if v.parent.name == sample_name)

    return video, dataset_root, video.parent.name


def draw_bbox(frame: np.ndarray, bbox, label: str) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(float(val))) for val in bbox]
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_bg = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(
        frame,
        (x1, max(0, y1 - text_height - 6)),
        (x1 + text_width + 6, y1),
        label_bg,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 3, y1 - 4),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def annotate_video(
    video_path: Path,
    samples_root: Optional[Path],
    model: RFDETRBase,
    feature_extractor: RMACFeatureExtractor,
    output_dir: Path,
    verbose: bool = False,
) -> Tuple[Path, List[dict]]:
    video_id = video_path.parent.name
    print(f"\nProcessing video: {video_id}")

    samples_root = samples_root or video_path.parent.parent
    reference_features = prepare_references(feature_extractor, samples_root, video_id)
    if reference_features:
        print(f"  Loaded {len(reference_features)} reference embeddings")
    else:
        print("  No reference images found; falling back to confidence scores")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_bboxes: List[dict] = []
    frame_idx = -1
    last_reported_pct = -5.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_image = maybe_apply_clahe(maybe_denoise_image(pil_image))

            detections = model.predict(pil_image, threshold=MODEL_THRESHOLD)
            detections = filter_detections(detections)
            if len(detections) == 0:
                writer.write(frame)
                continue

            similarity_scores = compute_similarity_scores(
                feature_extractor, pil_image, detections, reference_features
            )
            best_idx = select_detection_idx(detections, similarity_scores)
            if best_idx is None:
                writer.write(frame)
                continue

            bbox = detections.xyxy[best_idx]
            confidence = float(detections.confidence[best_idx])
            similarity = similarity_scores.get(best_idx)
            label_parts = [f"{confidence:.2f} conf"]
            if similarity is not None:
                label_parts.append(f"{similarity:.2f} sim")
            label = " | ".join(label_parts)

            frame_bboxes.append(frame_entry(frame_idx, bbox))

            annotated_frame = draw_bbox(frame.copy(), bbox, label)
            cv2.putText(
                annotated_frame,
                f"Frame {frame_idx}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated_frame)

            if total_frames:
                pct = frame_idx / total_frames * 100
                if pct - last_reported_pct >= 5:
                    last_reported_pct = pct
                    print(f"  Progress: {pct:.1f}% ({frame_idx}/{total_frames} frames)")
            elif verbose and frame_idx % 100 == 0:
                print(f"  Processed frame {frame_idx}")
    finally:
        cap.release()
        writer.release()

    print(f"  Annotated {len(frame_bboxes)} frames -> {output_path}")
    return output_path, frame_bboxes


def main() -> None:
    args = parse_args()

    video_path, samples_root, video_id = pick_video(args)
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = RFDETRBase(device=device)
    try:
        model.optimize_for_inference()
        print("Model optimized for inference.")
    except RuntimeError:
        print("Model optimization skipped (not critical).")

    feature_extractor = RMACFeatureExtractor(device=device, backbone_name="mobilenetv3_small_075")
    video_output_path, frame_bboxes = annotate_video(
        video_path, samples_root, model, feature_extractor, output_dir, args.verbose
    )

    if args.skip_json:
        return

    annotations = {
        "video_id": video_id,
        "annotations": [{"bboxes": frame_bboxes}],
    }
    json_dir = output_dir
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"{video_id}_annotations.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(annotations, fp, indent=2)

    print(f"Saved annotation payload to {json_path}")
    print(f"Annotated video saved to {video_output_path}")


if __name__ == "__main__":
    main()
