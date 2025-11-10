from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


@dataclass
class FrameIterator:
    source: Path
    fps: Optional[float]
    total_frames: Optional[int]

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for video iteration")
        cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.source}")

        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield idx, frame
                idx += 1
        finally:
            cap.release()


def build_frame_iterator(video_path: str | Path, fps_override: Optional[int]) -> FrameIterator:
    path = Path(video_path)
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video IO in the new pipeline")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    cap.release()
    return FrameIterator(source=path, fps=fps_override or fps, total_frames=total)
