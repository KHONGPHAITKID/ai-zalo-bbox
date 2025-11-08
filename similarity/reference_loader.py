from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Protocol

from PIL import Image, ImageEnhance, ImageOps
import numpy as np

from .utils import downscale_image


REFERENCE_AUG_TARGET = int(os.getenv("REFERENCE_AUG_TARGET", "3"))


class FeatureExtractor(Protocol):
    def extract(self, image: Image.Image) -> np.ndarray: ...


@dataclass
class ReferenceFeature:
    sample_name: str
    image_name: str
    image_path: Path
    feature: np.ndarray
    group_id: str


ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _iter_reference_images(
    samples_root: Path, target_sample: str | None = None
) -> Iterable[tuple[str, Path]]:
    for sample_dir in samples_root.iterdir():
        if not sample_dir.is_dir():
            continue
        if target_sample and sample_dir.name != target_sample:
            continue
        object_dir = sample_dir / "object_images"
        if not object_dir.exists():
            continue
        for image_path in sorted(object_dir.iterdir()):
            if image_path.suffix.lower() not in ALLOWED_IMAGE_EXTS:
                continue
            yield sample_dir.name, image_path


def list_reference_image_paths(samples_root: Path, sample_name: str) -> List[Path]:
    return [
        image_path
        for _, image_path in _iter_reference_images(
            samples_root, target_sample=sample_name
        )
    ]


def load_reference_features(
    samples_root: Path, sample_name: str, extractor: FeatureExtractor
) -> List[ReferenceFeature]:
    base_entries: List[tuple[str, Path, Image.Image]] = []
    for sample_name_entry, image_path in _iter_reference_images(
        samples_root, target_sample=sample_name
    ):
        with Image.open(image_path) as ref_image:
            base_entries.append(
                (sample_name_entry, image_path, ref_image.convert("RGB"))
            )

    references: List[ReferenceFeature] = []
    if not base_entries:
        return references

    preserve_resolution = getattr(extractor, "preserve_resolution", False)
    target_total = max(len(base_entries), REFERENCE_AUG_TARGET)
    variant_counters: dict[str, int] = {entry[1].stem: 0 for entry in base_entries}

    def prepare(image: Image.Image) -> Image.Image:
        return image if preserve_resolution else downscale_image(image)

    for sample_name_entry, image_path, prepared in base_entries:
        feature = extractor.extract(prepare(prepared))
        references.append(
            ReferenceFeature(
                sample_name=sample_name_entry,
                image_name=image_path.name,
                image_path=image_path,
                feature=feature,
                group_id=image_path.stem,
            )
        )

    variant_index = 0
    while len(references) < target_total:
        sample_name_entry, image_path, base_image = base_entries[
            variant_index % len(base_entries)
        ]
        variant = _generate_variant(
            base_image, variant_counters[image_path.stem], in_place=False
        )
        variant_counters[image_path.stem] += 1
        feature = extractor.extract(prepare(variant))
        variant_name = f"{image_path.stem}_aug{variant_counters[image_path.stem]}"
        references.append(
            ReferenceFeature(
                sample_name=sample_name_entry,
                image_name=variant_name,
                image_path=image_path,
                feature=feature,
                group_id=image_path.stem,
            )
        )
        variant_index += 1

    return references


def _generate_variant(image: Image.Image, variant_id: int, in_place: bool = True) -> Image.Image:
    base = image if in_place else image.copy()
    operations = [
        lambda img: ImageOps.mirror(img),
        lambda img: img.rotate(8, resample=Image.BICUBIC),
        lambda img: img.rotate(-8, resample=Image.BICUBIC),
        lambda img: ImageEnhance.Brightness(img).enhance(1.15),
        lambda img: ImageEnhance.Brightness(img).enhance(0.85),
        lambda img: ImageEnhance.Contrast(img).enhance(1.2),
        lambda img: ImageEnhance.Contrast(img).enhance(0.9),
        lambda img: ImageEnhance.Color(img).enhance(1.2),
        lambda img: ImageEnhance.Color(img).enhance(0.85),
        lambda img: ImageOps.autocontrast(img),
    ]
    op = operations[variant_id % len(operations)]
    augmented = op(base)
    if augmented.size != image.size:
        augmented = augmented.resize(image.size, Image.BICUBIC)
    return augmented
