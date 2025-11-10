from __future__ import annotations

from typing import Optional

from PIL import Image
import numpy as np

REFERENCE_MAX_DIM = 64
CROP_CONTEXT_PAD = 0.15  # 15% padding around each bbox


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

def hsv_hist_sim(a: Image.Image, b: Image.Image) -> float:
    import cv2, numpy as np
    def hist(im):
        h = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([h],[0,1,2],None,[12,8,4],[0,180,0,256,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    ha, hb = hist(a), hist(b)
    return float(1.0 - cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))

COLOR_MIN_SIM = 0.7
MARGIN_DELTA = 0.25
Z_THRESH = 2.0
ALPHA_INLIERS = 0.08

def score_with_checks(
    crop_img,
    feature_extractor,
    prototypes: dict[str, np.ndarray],
    ref_images: dict[str, Image.Image],
    *,
    precomputed_feature: np.ndarray | None = None,
):
    # 1) compute z-norm cosine to all prototypes
    q = precomputed_feature
    if q is None:
        q = feature_extractor.extract(crop_img)
    sims = []
    for k, proto in prototypes.items():
        z = float(np.dot(q, proto))
        # apply z-norm using extractor’s cohort
        if feature_extractor.cohort:
            co = np.dot(np.stack(feature_extractor.cohort,0), q)
            mu, sd = float(co.mean()), float(co.std()+1e-6)
            z = (z - mu)/sd
        sims.append((k, z))
    sims.sort(key=lambda x: x[1], reverse=True)
    if not sims: return []

    # 2) margin gating
    if len(sims) > 1 and (sims[0][1] - sims[1][1]) < MARGIN_DELTA:
        return []  # too ambiguous

    # 3) absolute threshold
    if sims[0][1] < Z_THRESH:
        return []

    # 4) color gate with the top candidate
    top_key, top_z = sims[0]
    if hsv_hist_sim(crop_img, ref_images[top_key]) < COLOR_MIN_SIM:
        return []

    return [top_key]
