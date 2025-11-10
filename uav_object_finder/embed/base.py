from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from PIL import Image

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from torchvision import transforms
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except Exception:  # pragma: no cover
    transforms = None  # type: ignore
    vit_b_16 = None  # type: ignore
    ViT_B_16_Weights = None  # type: ignore

from ..config import EmbedderConfig, RuntimeConfig


class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, crops: List[np.ndarray], batch: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class TorchVisionEmbedder(BaseEmbedder):
    def __init__(self, input_size: int, device: str) -> None:
        if torch is None or transforms is None or vit_b_16 is None:
            raise RuntimeError("TorchVision is unavailable")
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        try:
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if ViT_B_16_Weights else None
            backbone = vit_b_16(weights=weights)  # type: ignore
            backbone.heads = torch.nn.Identity()
        except Exception:  # pragma: no cover
            backbone = vit_b_16(weights=None)  # type: ignore
            backbone.heads = torch.nn.Identity()
        self.model = backbone.to(self.device)
        self.input_size = getattr(backbone, "image_size", input_size)
        self.model.eval()
        self.transform = transforms.Compose(  # type: ignore[arg-type]
            [
                transforms.ToPILImage(),
                transforms.Resize((self.input_size, self.input_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def reset(self) -> None:
        pass

    def encode(self, crops: List[np.ndarray], batch: int) -> np.ndarray:
        if not crops:
            return np.zeros((0, 1), dtype=np.float32)
        features: List[np.ndarray] = []
        for i in range(0, len(crops), batch):
            chunk = crops[i : i + batch]
            with torch.no_grad():  # type: ignore[attr-defined]
                tensor = torch.stack([self.transform(img) for img in chunk]).to(self.device)
                feats = self.model(tensor)
                if feats.dim() == 4:
                    feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
                else:
                    feats = feats.view(feats.size(0), -1)
                feats = torch.nn.functional.normalize(feats, dim=1)
                features.append(feats.cpu().numpy())
        return np.concatenate(features, axis=0)


class SimpleEmbedder(BaseEmbedder):
    """Fallback embedder using color moments + gradients (NumPy only)."""

    def __init__(self, input_size: int) -> None:
        self.input_size = input_size

    def reset(self) -> None:
        pass

    def encode(self, crops: List[np.ndarray], batch: int) -> np.ndarray:  # noqa: D401
        if not crops:
            return np.zeros((0, 1), dtype=np.float32)
        feats: List[np.ndarray] = []
        for crop in crops:
            img = Image.fromarray(crop)
            resized = np.asarray(img.resize((self.input_size, self.input_size))).astype(np.float32) / 255.0
            mean = resized.mean(axis=(0, 1))
            std = resized.std(axis=(0, 1))
            flat = resized.reshape(-1, resized.shape[-1])
            cov = np.cov(flat.T)
            cov_upper = cov[np.triu_indices_from(cov)]
            gradient_y, gradient_x = np.gradient(resized.mean(axis=2))
            grad_mag = np.sqrt(gradient_x**2 + gradient_y**2)
            grad_hist = np.histogram(grad_mag, bins=16, range=(0, grad_mag.max() + 1e-6))[0]
            feat = np.concatenate([mean, std, cov_upper, grad_hist])
            feat = feat / (np.linalg.norm(feat) + 1e-6)
            feats.append(feat.astype(np.float32))
        return np.stack(feats)


def build_embedder(config: EmbedderConfig, runtime: RuntimeConfig) -> BaseEmbedder:
    model_name = config.model.lower()
    if (
        model_name in {"dinov2_b14", "clip_vit_b16", "vit_b16"}
        and torch is not None
        and transforms is not None
        and vit_b_16 is not None
    ):
        try:
            return TorchVisionEmbedder(config.input_size, runtime.device)
        except RuntimeError:
            pass
    return SimpleEmbedder(config.input_size)
