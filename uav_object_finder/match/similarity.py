from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from ..config import SimilarityConfig
from ..types import Box


@dataclass
class SimilarityResult:
    scores: np.ndarray
    sim01: np.ndarray


class SimilarityScorer:
    def __init__(self, cfg: SimilarityConfig) -> None:
        self.cfg = cfg
        self.background_scores: deque[float] = deque(maxlen=cfg.adapt_bg_window)

    def score(self, embeddings: np.ndarray, gallery: np.ndarray) -> SimilarityResult:
        if embeddings.size == 0 or gallery.size == 0:
            sim = np.zeros((embeddings.shape[0],), dtype=np.float32)
        else:
            cos = embeddings @ gallery.T
            sim = cos.max(axis=1)
        sim01 = 0.5 * (sim + 1.0)
        return SimilarityResult(scores=sim, sim01=sim01)

    def update_background(self, neg_scores: Iterable[float]) -> None:
        for score in neg_scores:
            self.background_scores.append(float(score))

    def threshold(self) -> float:
        if not self.background_scores:
            return self.cfg.init_thresh
        mean_bg = sum(self.background_scores) / len(self.background_scores)
        adapted = self.cfg.init_thresh + 0.5 * (mean_bg - 0.5)
        return float(np.clip(adapted, 0.55, 0.72))


def conf_fuse(app: float, det: float) -> float:
    return 0.6 * app + 0.4 * det
