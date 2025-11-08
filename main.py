import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import supervision as sv
from PIL import Image, ImageFilter
import torch
from rfdetr import RFDETRBase

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency in some envs
    cv2 = None

from similarity import (
    DINOFeatureExtractor,
    crop_bbox,
    list_reference_image_paths,
    load_reference_features,
    rank_similarities,
    select_top_detection_indices,
)
from similarity.matching import SimilarityResult

SIMILARITY_TOP_K = None  # use all reference images for similarity scoring
ANNOTATION_TOP_K = 1
OUTPUT_MATCH_TOP_K = 2
MAX_DETECTIONS = 15
DENOISE_RADIUS = float(os.getenv("DENOISE_RADIUS", "0"))
CLAHE_ENABLED = os.getenv("CLAHE_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
CLAHE_CLIP_LIMIT = float(os.getenv("CLAHE_CLIP_LIMIT", "2.5"))
SKIP_LARGEST_DETECTIONS = int(os.getenv("SKIP_LARGEST_DETECTIONS", "0"))
_tile_env = os.getenv("CLAHE_TILE_GRID_SIZE", "8,8").split(",")
if len(_tile_env) != 2:
    CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)
else:
    CLAHE_TILE_GRID_SIZE = (
        max(1, int(_tile_env[0].strip() or 8)),
        max(1, int(_tile_env[1].strip() or 8)),
    )
del _tile_env

if CLAHE_ENABLED and cv2 is None:
    print("CLAHE requested but OpenCV (cv2) is unavailable; disabling CLAHE")
    CLAHE_ENABLED = False

CLAHE_OPERATOR = (
    cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    if CLAHE_ENABLED and CLAHE_CLIP_LIMIT > 0 and cv2 is not None
    else None
)


def maybe_denoise_image(image: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return image
    # Light Gaussian blur works well as a simple, fast denoiser
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def maybe_apply_clahe(image: Image.Image) -> Image.Image:
    if CLAHE_OPERATOR is None:
        return image

    working_image = image if image.mode == "RGB" else image.convert("RGB")
    np_image = np.array(working_image)
    if np_image.ndim != 3 or np_image.shape[2] != 3:
        return working_image

    bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_enhanced = CLAHE_OPERATOR.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    rgb_enhanced = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_enhanced)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

if DENOISE_RADIUS > 0:
    print(f"Denoising enabled (Gaussian blur radius={DENOISE_RADIUS})")
else:
    print("Denoising disabled")

if CLAHE_OPERATOR is not None:
    print(
        "CLAHE enabled "
        f"(clip_limit={CLAHE_CLIP_LIMIT}, tile_grid={CLAHE_TILE_GRID_SIZE})"
    )
else:
    print("CLAHE disabled")

if SKIP_LARGEST_DETECTIONS > 0:
    print(f"Ignoring top {SKIP_LARGEST_DETECTIONS} largest detections before similarity scoring")

# Initialize model
print(f"Loading RF-DETR model on {device.upper()}...")
model = RFDETRBase(device=device)

# Try to optimize for inference (may not work with all PyTorch versions)
try:
    model.optimize_for_inference()
    print("Model loaded and optimized!")
except RuntimeError as e:
    print("Model loaded (optimization skipped - not critical)")
    print(f"  Note: JIT tracing not compatible with this model output")

# Find all raw extracted images and reference objects
samples_root = Path("data/train/samples")
raw_images = list(samples_root.glob("MobilePhone_0/extract_images/raw/.jpg"))
raw_images = list(samples_root.glob("Backpack_0/extract_images/raw/frame_5103_cropped.jpg"))

if not raw_images:
    print("No extracted images found! Please run extract_data.py first.")
    exit(1)

print(f"Found {len(raw_images)} extracted images")

feature_extractor = DINOFeatureExtractor(device=device)

# Select a single random image for testing
num_test_images = 1
test_images = random.sample(raw_images, num_test_images)
test_images = [raw_images[0]]

print(f"Testing on {num_test_images} random images...")

# Create output folder for inference results
output_folder = Path("inference_results")
output_folder.mkdir(exist_ok=True)

# Run inference on each test image
for i, image_path in enumerate(test_images, 1):
    sample_name = image_path.parent.parent.parent.name
    reference_paths = list_reference_image_paths(samples_root, sample_name)
    has_references = bool(reference_paths)
    print(f"\n[{i}/{num_test_images}] Processing: {image_path.name}")
    print(f"  Source: {sample_name}")
    if not has_references:
        print("  No reference images available for this sample; similarity skipped.")
        reference_features: List[SimilarityResult] = []
    else:
        feature_extractor.reset()
        # feature_extractor.fine_tune(reference_paths)
        reference_features = load_reference_features(
            samples_root, sample_name, feature_extractor
        )

    # Load image
    image = Image.open(image_path).convert("RGB")
    preprocessed_image = maybe_denoise_image(image, DENOISE_RADIUS)
    if DENOISE_RADIUS > 0:
        denoised_folder = output_folder / "denoised"
        denoised_folder.mkdir(exist_ok=True)
        denoised_path = denoised_folder / f"{image_path.stem}_denoised.jpg"
        preprocessed_image.save(denoised_path)
        print(f"  Saved denoised image to: {denoised_path}")

    preprocessed_image = maybe_apply_clahe(preprocessed_image)
    if CLAHE_OPERATOR is not None:
        clahe_folder = output_folder / "clahe"
        clahe_folder.mkdir(exist_ok=True)
        clahe_path = clahe_folder / f"{image_path.stem}_clahe.jpg"
        preprocessed_image.save(clahe_path)
        print(f"  Saved CLAHE image to: {clahe_path}")

    # Run inference
    detections = model.predict(preprocessed_image, threshold=0.00)
    if len(detections) > MAX_DETECTIONS:
        top_indices = np.argsort(detections.confidence)[-MAX_DETECTIONS:]
        top_indices.sort()
        detections = detections[top_indices]

    SKIP_LARGEST_DETECTIONS = len(detections) // 3
    if SKIP_LARGEST_DETECTIONS > 0 and len(detections) > SKIP_LARGEST_DETECTIONS:
        bbox_areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (
            detections.xyxy[:, 3] - detections.xyxy[:, 1]
        )
        largest_indices = np.argsort(bbox_areas)[-SKIP_LARGEST_DETECTIONS:]
        keep_mask = np.ones(len(detections), dtype=bool)
        keep_mask[largest_indices] = False
        detections = detections[np.where(keep_mask)[0]]
        print(
            f"  Removed {len(largest_indices)} largest detections "
            "before similarity comparison"
        )

    print(f"  Detected {len(detections)} objects:")

    detection_matches: List[List[SimilarityResult]] = []
    bbox_crops: List[Optional[Image.Image]] = []
    for bbox in detections.xyxy:
        cropped_image = crop_bbox(image, bbox)
        bbox_crops.append(cropped_image)
        if not has_references or cropped_image is None:
            detection_matches.append([])
            continue
        bbox_feature = feature_extractor.extract(cropped_image)
        detection_matches.append(
            rank_similarities(bbox_feature, reference_features, SIMILARITY_TOP_K)
        )

    # Print detections with confidence and similarity rankings
    for idx in range(len(detections)):
        confidence = float(detections.confidence[idx])
        print(f"    - Detection #{idx + 1} | conf: {confidence:.2f}")
        matches = detection_matches[idx] if idx < len(detection_matches) else []
        if matches:
            for rank, match in enumerate(matches[:OUTPUT_MATCH_TOP_K], 1):
                print(
                    f"       [{rank}] {match.sample_name} ({match.image_name}): {match.similarity:.3f}"
                )
        elif has_references:
            print("       No valid crop for similarity comparison")
        else:
            print("       Reference images unavailable")

    # Annotate image
    annotation_indices = select_top_detection_indices(
        detections.confidence,
        detection_matches,
        has_references,
        ANNOTATION_TOP_K,
    )
    if annotation_indices:
        annotation_detections = detections[annotation_indices]
    else:
        annotation_detections = detections

    box_annotator = sv.BoxAnnotator()
    crops_folder = output_folder / "crops" / image_path.stem
    crops_folder.mkdir(parents=True, exist_ok=True)

    crop_paths: List[Optional[str]] = []
    reference_copy_paths: List[List[str]] = [[] for _ in range(len(detections))]
    for idx, crop in enumerate(bbox_crops):
        if crop is None:
            crop_paths.append(None)
            continue
        crop_path = crops_folder / f"detection_{idx + 1}.jpg"
        crop.save(crop_path)
        crop_paths.append(str(crop_path))
        print(f"  Saved crop for detection #{idx + 1} to: {crop_path}")
        matches = detection_matches[idx][:OUTPUT_MATCH_TOP_K] if idx < len(detection_matches) else []
        match_paths: List[str] = []
        for rank, match in enumerate(matches, 1):
            dest_path = (
                crops_folder
                / f"detection_{idx + 1}_match_{rank}_{match.image_name}"
            )
            shutil.copy(match.image_path, dest_path)
            match_paths.append(str(dest_path))
            print(
                f"    Copied ref image for detection #{idx + 1} match #{rank} to: {dest_path}"
            )
        reference_copy_paths[idx] = match_paths

    # Save full annotation (original behavior)
    full_annotated = box_annotator.annotate(image.copy(), detections)
    full_output_path = output_folder / f"inference_{i}_{image_path.stem}.jpg"
    full_annotated.save(full_output_path)
    print(f"  Saved full annotations to: {full_output_path}")

    # Save similarity-filtered annotation
    filtered_annotated = box_annotator.annotate(image.copy(), annotation_detections)
    filtered_output_path = (
        output_folder / f"inference_{i}_{image_path.stem}_top{ANNOTATION_TOP_K}.jpg"
    )
    filtered_annotated.save(filtered_output_path)
    print(f"  Saved top-{ANNOTATION_TOP_K} annotations to: {filtered_output_path}")

    # Save summary JSON
    summary_records = []
    for idx in range(len(detections)):
        matches = detection_matches[idx] if idx < len(detection_matches) else []
        match_copies = reference_copy_paths[idx] if idx < len(reference_copy_paths) else []
        match_entries = []
        for match_idx, match in enumerate(matches):
            copied_path = match_copies[match_idx] if match_idx < len(match_copies) else None
            match_entries.append(
                {
                    "sample_name": match.sample_name,
                    "image_name": match.image_name,
                    "similarity": match.similarity,
                    "object_image_path": str(match.image_path),
                    "copied_path": copied_path,
                }
            )

        summary_records.append(
            {
                "detection_index": idx + 1,
                "confidence": float(detections.confidence[idx]),
                "bbox": detections.xyxy[idx].tolist(),
                "crop_path": crop_paths[idx] if idx < len(crop_paths) else None,
                "matches": match_entries,
            }
        )

    summary_payload = {
        "sample_name": sample_name,
        "image_path": str(image_path),
        "full_annotation_path": str(full_output_path),
        "top_k_annotation_path": str(filtered_output_path),
        "records": summary_records,
    }

    summary_path = output_folder / f"inference_{i}_{image_path.stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"  Saved similarity summary to: {summary_path}")

print(f"\n✓ Inference complete! Results saved to '{output_folder}/' folder")
print(f"  Processed {num_test_images} images")
