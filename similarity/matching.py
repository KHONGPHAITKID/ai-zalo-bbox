from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .reference_loader import ReferenceFeature
from .utils import score_with_checks


@dataclass
class SimilarityResult:
    sample_name: str
    image_name: str
    image_path: Path
    distance: float
    group_id: Optional[str] = None


def euclidean_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.size == 0 or vec_b.size == 0:
        return float("inf")
    return float(np.linalg.norm(vec_a - vec_b))


def rank_similarities(
    target_feature: np.ndarray,
    references: Sequence[ReferenceFeature],
    top_k: Optional[int] = None,
    *,
    crop_image: Optional[Image.Image] = None,
    feature_extractor: Optional[Any] = None,
) -> List[SimilarityResult]:
    if target_feature.size == 0 or not references:
        return []
    if top_k is not None and top_k <= 0:
        return []

    aggregated: Dict[str, dict] = {}
    for reference in references:
        distance = euclidean_distance(target_feature, reference.feature)
        group_id = getattr(reference, "group_id", None) or reference.image_name
        display_name = reference.image_name
        if group_id and display_name.startswith(f"{group_id}_aug"):
            display_name = f"{group_id}.jpg"
        group_state = aggregated.setdefault(
            group_id,
            {
                "values": [],
                "features": [],
                "sample_name": reference.sample_name,
                "image_name": display_name,
                "image_path": reference.image_path,
            },
        )
        group_state["values"].append(distance)
        group_state["features"].append(reference.feature)

    prototypes: Dict[str, np.ndarray] = {}
    for group_id, state in aggregated.items():
        feats = state["features"]
        if not feats:
            continue
        proto = np.mean(np.stack(feats, axis=0), axis=0)
        norm = np.linalg.norm(proto) + 1e-8
        prototypes[group_id] = proto / norm

    scored = [
        SimilarityResult(
            sample_name=state["sample_name"],
            image_name=state["image_name"],
            image_path=state["image_path"],
            distance=float(np.mean(state["values"])) if state["values"] else float("inf"),
            group_id=group_id,
        )
        for group_id, state in aggregated.items()
    ]

    # gating_enabled = (
    #     crop_image is not None and feature_extractor is not None and bool(prototypes)
    # )
    # if gating_enabled:
    #     reference_images: Dict[str, Image.Image] = {}
    #     for group_id, state in aggregated.items():
    #         try:
    #             with Image.open(state["image_path"]) as ref_img:
    #                 reference_images[group_id] = ref_img.convert("RGB")
    #         except Exception:
    #             reference_images.clear()
    #             break
    #     if reference_images and len(reference_images) == len(prototypes):
    #         allowed_keys = score_with_checks(
    #             crop_image,
    #             feature_extractor,
    #             prototypes,
    #             reference_images,
    #             precomputed_feature=target_feature,
    #         )
    #         if not allowed_keys:
    #             return []
    #         allowed_set = set(allowed_keys)
    #         scored = [result for result in scored if result.group_id in allowed_set]
    #         if not scored:
    #             return []

    scored.sort(key=lambda item: item.distance)
    if top_k is None or top_k >= len(scored):
        return scored
    return scored[:top_k]


def select_top_detection_indices(
    confidences: Sequence[float],
    detection_matches: Sequence[Sequence[SimilarityResult]],
    references_available: bool,
    top_k: int,
) -> List[int]:
    num_detections = len(confidences)
    if num_detections == 0 or top_k <= 0:
        return []
    if num_detections <= top_k:
        return list(range(num_detections))

    if references_available:
        scores: List[float] = []
        for matches in detection_matches:
            if matches:
                mean_distance = float(
                    sum(match.distance for match in matches) / len(matches)
                )
                scores.append(mean_distance)
            else:
                scores.append(float("inf"))
    else:
        scores = [float(conf) for conf in confidences]

    if references_available:
        top_indices = np.argsort(scores)[:top_k]
    else:
        top_indices = np.argsort(scores)[-top_k:]
    return sorted(top_indices.tolist())
