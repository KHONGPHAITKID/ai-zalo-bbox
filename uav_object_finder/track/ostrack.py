from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from ..config import TrackerConfig
from ..types import Box


@dataclass
class TrackerState:
    template: np.ndarray
    box: Box


class OStrackFallback:
    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.state: Optional[TrackerState] = None

    def reset(self) -> None:
        self.state = None

    def update_template(self, frame: np.ndarray, box: Box) -> None:
        x1, y1, x2, y2 = box.xyxy.astype(int)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        self.state = TrackerState(template=crop.copy(), box=box)

    def track(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.state is None:
            raise RuntimeError("Tracker not initialized")
        if cv2 is None:
            raise RuntimeError("OpenCV is required for fallback tracker")
        h, w = frame.shape[:2]
        last_box = self.state.box.xyxy
        cx = (last_box[0] + last_box[2]) / 2
        cy = (last_box[1] + last_box[3]) / 2
        width = last_box[2] - last_box[0]
        height = last_box[3] - last_box[1]
        search_scale = self.cfg.search_scale
        x1 = int(max(0, cx - width * search_scale / 2))
        y1 = int(max(0, cy - height * search_scale / 2))
        x2 = int(min(w, cx + width * search_scale / 2))
        y2 = int(min(h, cy + height * search_scale / 2))
        window = frame[y1:y2, x1:x2]
        if window.shape[0] < self.state.template.shape[0] or window.shape[1] < self.state.template.shape[1]:
            return last_box, 0.0
        res = cv2.matchTemplate(window, self.state.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (x1 + max_loc[0], y1 + max_loc[1])
        bottom_right = (
            top_left[0] + self.state.template.shape[1],
            top_left[1] + self.state.template.shape[0],
        )
        tracked = np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]], dtype=np.float32)
        return tracked, float(max_val)
