from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageEnhance

from .config import GalleryConfig, RuntimeConfig
from .embed import BaseEmbedder


@dataclass
class Gallery:
    embeddings: np.ndarray
    ids: List[str]

    def with_memory(self, memory: "MemoryBank") -> "Gallery":
        if not memory.embeddings:
            return Gallery(self.embeddings, list(self.ids))
        mem = np.stack(memory.embeddings) if memory.embeddings else np.zeros((0, self.embeddings.shape[1]))
        joined = np.concatenate([self.embeddings, mem], axis=0)
        ids = list(self.ids) + [f"mem_{i}" for i in range(len(memory.embeddings))]
        return Gallery(joined, ids)


class MemoryBank:
    def __init__(self, max_items: int, sim_cap: float) -> None:
        self.max_items = max_items
        self.sim_cap = sim_cap
        self.embeddings: List[np.ndarray] = []

    def maybe_add(self, embedding: np.ndarray) -> None:
        if embedding.ndim == 2:
            embedding = embedding[0]
        norm = np.linalg.norm(embedding) + 1e-6
        embedding = embedding / norm
        if not self.embeddings:
            self.embeddings.append(embedding)
            return
        sims = [float(np.dot(embedding, existing)) for existing in self.embeddings]
        if max(sims) >= self.sim_cap:
            return
        if len(self.embeddings) >= self.max_items:
            self.embeddings.pop(0)
        self.embeddings.append(embedding)


AUG_ROTATIONS = [0, 90, 180, 270]


def _augment(image: Image.Image, cfg: GalleryConfig) -> List[Image.Image]:
    augmented = [image]
    for _ in range(cfg.aug_per_ref):
        scale = random.uniform(0.8, 1.2)
        new_size = (max(4, int(image.width * scale)), max(4, int(image.height * scale)))
        jittered = image.resize(new_size, Image.BICUBIC)
        hue = ImageEnhance.Color(jittered).enhance(random.uniform(0.85, 1.15))
        bright = ImageEnhance.Brightness(hue).enhance(random.uniform(0.9, 1.1))
        contrast = ImageEnhance.Contrast(bright).enhance(random.uniform(0.9, 1.1))
        if random.random() > 0.5:
            contrast = contrast.transpose(Image.FLIP_LEFT_RIGHT)
        angle = random.choice(AUG_ROTATIONS)
        augmented.append(contrast.rotate(angle))
    return augmented[: cfg.max_refs]


def build_reference_gallery(
    reference_paths: Sequence[Path],
    cfg: GalleryConfig,
    embedder: BaseEmbedder,
    runtime: RuntimeConfig,
) -> Gallery:
    images: List[np.ndarray] = []
    ids: List[str] = []
    for path in reference_paths[: cfg.max_refs]:
        img = Image.open(path).convert("RGB")
        for aug in _augment(img, cfg):
            images.append(np.array(aug))
            ids.append(path.stem)
    embeddings = embedder.encode(images, batch=runtime.batch_embed) if images else np.zeros((0, 1), dtype=np.float32)
    return Gallery(embeddings=embeddings, ids=ids)
