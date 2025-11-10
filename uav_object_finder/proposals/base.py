from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from ..config import ProposalConfig
from ..types import Box
from .detr import DetrProposalGenerator
from .yolo import YoloProposalGenerator


class ProposalGenerator(ABC):
    def __init__(self, config: ProposalConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, frame: np.ndarray) -> List[Box]:
        raise NotImplementedError


class ContourProposalGenerator(ProposalGenerator):
    """Simple contour-based class-agnostic proposals."""

    def generate(self, frame: np.ndarray) -> List[Box]:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for contour proposal generation")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(blur, 30, 120)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[Box] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h <= 16:
                continue
            boxes.append(
                Box(
                    xyxy=np.array([x, y, x + w, y + h], dtype=np.float32),
                    score_det=float(min(0.99, 0.5 + 0.5 * (w * h) / (frame.shape[0] * frame.shape[1]))),
                    class_id=None,
                    score_app=0.0,
                )
            )
        boxes.sort(key=lambda b: b.score_det, reverse=True)
        return boxes[: self.config.max_det]


def build_proposal_generator(config: ProposalConfig) -> ProposalGenerator:
    engine = config.engine.lower()
    if engine == "yolo":
        try:
            return YoloProposalGenerator(config)
        except RuntimeError:
            pass
    if engine == "detr":
        try:
            return DetrProposalGenerator(config)
        except RuntimeError:
            pass
    return ContourProposalGenerator(config)
