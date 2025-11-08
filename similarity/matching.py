from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .reference_loader import ReferenceFeature


@dataclass
class SimilarityResult:
    sample_name: str
    image_name: str
    image_path: Path
    similarity: float


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def rank_similarities(
    target_feature: np.ndarray,
    references: Sequence[ReferenceFeature],
    top_k: Optional[int] = None,
) -> List[SimilarityResult]:
    if target_feature.size == 0 or not references:
        return []
    if top_k is not None and top_k <= 0:
        return []

    aggregated: Dict[str, dict] = {}
    for reference in references:
        similarity = cosine_similarity(target_feature, reference.feature)
        group_id = getattr(reference, "group_id", None) or reference.image_name
        display_name = reference.image_name
        if group_id and display_name.startswith(f"{group_id}_aug"):
            display_name = f"{group_id}.jpg"
        group_state = aggregated.setdefault(
            group_id,
            {
                "values": [],
                "sample_name": reference.sample_name,
                "image_name": display_name,
                "image_path": reference.image_path,
            },
        )
        group_state["values"].append(similarity)

    scored = [
        SimilarityResult(
            sample_name=state["sample_name"],
            image_name=state["image_name"],
            image_path=state["image_path"],
            similarity=float(np.mean(state["values"])) if state["values"] else 0.0,
        )
        for state in aggregated.values()
    ]

    scored.sort(key=lambda item: item.similarity, reverse=True)
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
                mean_similarity = float(
                    sum(match.similarity for match in matches) / len(matches)
                )
                scores.append(mean_similarity)
            else:
                scores.append(-1.0)
    else:
        scores = [float(conf) for conf in confidences]

    top_indices = np.argsort(scores)[-top_k:]
    return sorted(top_indices.tolist())
