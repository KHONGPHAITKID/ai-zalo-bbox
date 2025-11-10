from __future__ import annotations

from typing import List

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from torchvision import transforms
    from torchvision.models.detection import Detr_ResNet50_Weights, detr_resnet50
except Exception:  # pragma: no cover
    transforms = None  # type: ignore
    detr_resnet50 = None  # type: ignore
    Detr_ResNet50_Weights = None  # type: ignore

from ..config import ProposalConfig
from ..types import Box


class DetrProposalGenerator:
    def __init__(self, config: ProposalConfig) -> None:
        if torch is None or transforms is None or detr_resnet50 is None:
            raise RuntimeError("TorchVision detection models are unavailable")
        self.config = config
        try:
            weights = Detr_ResNet50_Weights.DEFAULT if Detr_ResNet50_Weights else None
            self.model = detr_resnet50(weights=weights)
        except Exception:  # pragma: no cover - optional weights
            self.model = detr_resnet50(weights=None)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.config.input_size, antialias=True),
            ]
        )

    @torch.no_grad()  # type: ignore[attr-defined]
    def generate(self, frame: np.ndarray) -> List[Box]:
        tensor = self.transform(frame)
        outputs = self.model([tensor])[0]
        scores = outputs["scores"].cpu().numpy()
        keep = scores >= self.config.conf_thres
        boxes: List[Box] = []
        for xyxy, score in zip(outputs["boxes"].cpu().numpy()[keep], scores[keep]):
            boxes.append(
                Box(
                    xyxy=xyxy.astype(np.float32),
                    score_det=float(score),
                    class_id=None,
                    score_app=0.0,
                )
            )
        boxes.sort(key=lambda b: b.score_det, reverse=True)
        return boxes[: self.config.max_det]
