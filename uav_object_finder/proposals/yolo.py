from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ..config import ProposalConfig
from ..types import Box


class YoloProposalGenerator:
    def __init__(self, config: ProposalConfig) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Ultralytics YOLO is not installed") from exc

        weight_path = (
            Path("yolov8n.pt")
            if not Path(config.engine).exists()
            else Path(config.engine)
        )
        self.model = YOLO(str(weight_path))
        self.config = config

    def generate(self, frame: np.ndarray) -> List[Box]:
        results = self.model.predict(
            source=frame,
            imgsz=self.config.input_size,
            conf=self.config.conf_thres,
            iou=self.config.iou_thres,
            max_det=self.config.max_det,
            verbose=False,
        )
        boxes: List[Box] = []
        for result in results:
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None
            for i, row in enumerate(xyxy[: self.config.max_det]):
                boxes.append(
                    Box(
                        xyxy=row.astype(np.float32),
                        score_det=float(scores[i]),
                        class_id=int(cls[i]) if cls is not None else None,
                        score_app=0.0,
                    )
                )
        return boxes[: self.config.max_det]
