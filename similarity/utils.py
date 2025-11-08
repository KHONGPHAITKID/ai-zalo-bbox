from __future__ import annotations

from typing import Optional

from PIL import Image
import numpy as np

REFERENCE_MAX_DIM = 192
CROP_CONTEXT_PAD = 0.0  # 15% padding around each bbox


def crop_bbox(image: Image.Image, bbox: np.ndarray) -> Optional[Image.Image]:
    x1, y1, x2, y2 = bbox
    width, height = image.size
    pad_x = int((x2 - x1) * CROP_CONTEXT_PAD)
    pad_y = int((y2 - y1) * CROP_CONTEXT_PAD)
    x1 = max(0, min(width, int(x1) - pad_x))
    y1 = max(0, min(height, int(y1) - pad_y))
    x2 = max(0, min(width, int(x2) + pad_x))
    y2 = max(0, min(height, int(y2) + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def downscale_image(image: Image.Image, max_dim: int = REFERENCE_MAX_DIM) -> Image.Image:
    width, height = image.size
    largest = max(width, height)
    if largest <= max_dim:
        return image
    scale = max_dim / largest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.BICUBIC)
