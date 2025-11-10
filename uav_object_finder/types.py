from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class Box:
    xyxy: NDArray[np.float32]
    score_det: float
    score_app: float = 0.0
    class_id: Optional[int] = None
    crop_idx: Optional[int] = None

    def as_xyxy(self) -> NDArray[np.float32]:
        return self.xyxy.astype(np.float32)


@dataclass
class KalmanState:
    state: NDArray[np.float32]
    covariance: NDArray[np.float32]


@dataclass
class Track:
    id: int
    state: Literal["Tentative", "Confirmed", "Lost"] = "Tentative"
    kf: Optional[KalmanState] = None
    last_box: Optional[Box] = None
    last_conf: float = 0.0
    age: int = 0
    time_since_update: int = 0
    history: list[Box] = field(default_factory=list)

    def is_active(self) -> bool:
        return self.state in {"Tentative", "Confirmed"}
