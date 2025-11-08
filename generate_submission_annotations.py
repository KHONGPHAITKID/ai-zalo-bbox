from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter
import torch
from rfdetr import RFDETRBase

try:
    import cv2
except ImportError as exc:  # pragma: no cover - hard requirement for video I/O
    raise ImportError("OpenCV (cv2) is required to run this script.") from exc

from similarity import (
    DINOFeatureExtractor,
    DINOFeatureExtractor,
    ReferenceFeature,
    crop_bbox,
    list_reference_image_paths,
    load_reference_features,
    rank_similarities,
)


PUBLIC_SAMPLES_ROOT = Path(os.getenv("PUBLIC_SAMPLES_ROOT", "data/public_test/samples"))
OUTPUT_PATH = Path(os.getenv("ANNOTATIONS_OUTPUT", "annotations.json"))

MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.0"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.13"))
MAX_DETECTIONS_PER_FRAME = int(os.getenv("MAX_DETECTIONS_PER_FRAME", "15"))
SIMILARITY_TOP_K = 1

DENOISE_RADIUS = float(os.getenv("DENOISE_RADIUS", "0"))
CLAHE_ENABLED = os.getenv("CLAHE_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
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
    feature_extractor: DINOFeatureExtractor,
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
    feature_extractor: DINOFeatureExtractor, samples_root: Path, sample_name: str
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


def process_video(
    video_path: Path,
    model: RFDETRBase,
    feature_extractor: DINOFeatureExtractor,
) -> dict:
    video_id = video_path.parent.name
    print(f"\nProcessing video: {video_id}")

    reference_features = prepare_references(feature_extractor, PUBLIC_SAMPLES_ROOT, video_id)
    if reference_features:
        print(f"  Loaded {len(reference_features)} reference embeddings")
    else:
        print("  No reference images found; falling back to confidence scores")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_bboxes: List[dict] = []
    frame_idx = -1

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
            continue

        similarity_scores = compute_similarity_scores(
            feature_extractor, pil_image, detections, reference_features
        )
        best_idx = select_detection_idx(detections, similarity_scores)
        if best_idx is None:
            continue

        bbox = detections.xyxy[best_idx]
        frame_bboxes.append(frame_entry(frame_idx, bbox))

        if frame_idx % 200 == 0:
            pct = (frame_idx / total_frames * 100) if total_frames else 0
            print(f"  Processed frame {frame_idx} ({pct:.1f}% done)")

    cap.release()
    print(f"  Collected {len(frame_bboxes)} bounding boxes")

    return {
        "video_id": video_id,
        "annotations": [
            {
                "bboxes": frame_bboxes,
            }
        ],
    }


def main() -> None:
    if not PUBLIC_SAMPLES_ROOT.exists():
        raise FileNotFoundError(
            f"Public test samples not found at {PUBLIC_SAMPLES_ROOT.resolve()}"
        )

    video_paths = sorted(PUBLIC_SAMPLES_ROOT.glob("*/drone_video.mp4"))
    if not video_paths:
        raise FileNotFoundError(
            f"No videos found under {PUBLIC_SAMPLES_ROOT.resolve()}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = RFDETRBase(device=device)
    try:
        model.optimize_for_inference()
        print("Model optimized for inference.")
    except RuntimeError:
        print("Model optimization skipped (not critical).")

    feature_extractor = DINOFeatureExtractor(device=device)

    submission: List[dict] = []
    for video_path in video_paths:
        submission.append(process_video(video_path, model, feature_extractor))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(submission, fp, indent=2)

    print(f"\nSaved annotations to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
