from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..types import Box


def clamp_box(box: NDArray[np.float32], shape: Tuple[int, int, int]) -> NDArray[np.float32]:
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    x2 = np.clip(x2, x1 + 1, w)
    y2 = np.clip(y2, y1 + 1, h)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_matrix(a: Sequence[NDArray[np.float32]], b: Sequence[NDArray[np.float32]]) -> NDArray[np.float32]:
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a_arr = np.stack(a)
    b_arr = np.stack(b)

    area_a = (a_arr[:, 2] - a_arr[:, 0]) * (a_arr[:, 3] - a_arr[:, 1])
    area_b = (b_arr[:, 2] - b_arr[:, 0]) * (b_arr[:, 3] - b_arr[:, 1])

    ious = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, box_a in enumerate(a_arr):
        x1 = np.maximum(box_a[0], b_arr[:, 0])
        y1 = np.maximum(box_a[1], b_arr[:, 1])
        x2 = np.minimum(box_a[2], b_arr[:, 2])
        y2 = np.minimum(box_a[3], b_arr[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        union = area_a[i] + area_b - inter
        ious[i] = np.where(union <= 0, 0.0, inter / union)
    return ious


def crop_from_boxes(frame: NDArray[np.uint8], boxes: Iterable[Box]) -> Tuple[List[NDArray[np.uint8]], List[Box]]:
    crops: List[NDArray[np.uint8]] = []
    new_boxes: List[Box] = []
    for box in boxes:
        x1, y1, x2, y2 = clamp_box(box.xyxy, frame.shape)
        crop = frame[int(y1) : int(y2), int(x1) : int(x2)]
        if crop.size == 0:
            continue
        crops.append(crop)
        new_boxes.append(
            Box(
                xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                score_det=box.score_det,
                score_app=box.score_app,
                class_id=box.class_id,
                crop_idx=box.crop_idx,
            )
        )
    return crops, new_boxes


def resize_boxes(boxes: Sequence[NDArray[np.float32]], scale_x: float, scale_y: float) -> List[NDArray[np.float32]]:
    resized: List[NDArray[np.float32]] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        resized.append(
            np.array([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=np.float32)
        )
    return resized
