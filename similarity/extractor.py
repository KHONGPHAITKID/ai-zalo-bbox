from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image


class FeatureExtractor(Protocol):
    """Minimal interface implemented by all feature extractors."""

    device: str

    def reset(self) -> None: ...

    def fine_tune(self, image_paths: Sequence[Path]) -> None: ...

    def extract(self, image: Image.Image) -> np.ndarray: ...
