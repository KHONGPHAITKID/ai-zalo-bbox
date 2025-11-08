from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor, AutoModel

from .utils import downscale_image


@dataclass
class DINOFeatureExtractor:
    """Extract normalized embeddings from a pretrained DINO ViT model.

    Uses the Hugging Face Hub weights (facebook/dino-vits16) so no manual
    checkpoint handling is required.
    """

    device: str = "cpu"
    dtype: Optional[torch.dtype] = None
    model_name: str = "facebook/dino-vits16"

    finetune_steps: int = 10
    finetune_lr: float = 1e-5

    def __post_init__(self) -> None:
        torch_dtype = self.dtype or torch.float32
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()
        self._base_state_dict = copy.deepcopy(self.model.state_dict())
        self._augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            ]
        )

    def reset(self) -> None:
        self.model.load_state_dict(self._base_state_dict)
        self.model.to(self.device)
        self.model.eval()

    def fine_tune(self, image_paths: Sequence[Path]) -> None:
        if not image_paths or self.finetune_steps <= 0 or self.finetune_lr <= 0:
            return

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.finetune_lr)

        for _ in range(self.finetune_steps):
            image_path = Path(random.choice(image_paths))
            with Image.open(image_path) as img:
                base_image = downscale_image(img.convert("RGB"))

            view_a = self._augmentation(base_image)
            view_b = self._augmentation(base_image)

            inputs_a = self.processor(images=view_a, return_tensors="pt")
            inputs_b = self.processor(images=view_b, return_tensors="pt")
            inputs_a = {k: v.to(self.device) for k, v in inputs_a.items()}
            inputs_b = {k: v.to(self.device) for k, v in inputs_b.items()}
            if self.dtype is not None:
                inputs_a = {k: v.to(self.dtype) for k, v in inputs_a.items()}
                inputs_b = {k: v.to(self.dtype) for k, v in inputs_b.items()}

            outputs_a = self.model(**inputs_a).last_hidden_state[:, 0]
            outputs_b = self.model(**inputs_b).last_hidden_state[:, 0]
            outputs_a = F.normalize(outputs_a, dim=-1)
            outputs_b = F.normalize(outputs_b, dim=-1)

            loss = 1 - F.cosine_similarity(outputs_a, outputs_b).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()

    def extract(self, image: Image.Image) -> np.ndarray:
        """Return a normalized embedding for the given image."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.dtype is not None:
            inputs = {k: v.to(self.dtype) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0]
        feature = cls_embedding.squeeze(0).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature /= norm
        return feature
