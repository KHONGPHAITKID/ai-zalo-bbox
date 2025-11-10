from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from ..config import PostConfig
from ..types import Box


class TemporalSmoother:
    def __init__(self, cfg: PostConfig) -> None:
        self.cfg = cfg
        self.window: Deque[np.ndarray] = deque(maxlen=cfg.smooth_window)

    def update(self, box: Optional[Box]) -> Tuple[Optional[np.ndarray], float]:
        if box is not None:
            self.window.append(box.xyxy)
        if not self.window:
            return None, 0.0
        stacked = np.stack(list(self.window))
        median = np.median(stacked, axis=0)
        conf = 1.0
        return median.astype(np.float32), conf
