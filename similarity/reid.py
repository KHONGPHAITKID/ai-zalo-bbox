# minimal deps: torch, torchvision, numpy, opencv-python, pillow
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Dict, List
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# ---------- building blocks ----------

class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)

class BNNeck(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim, affine=True)
        nn.init.constant_(self.bn.bias, 0)
        nn.init.constant_(self.bn.weight, 1.0)
    def forward(self, x):
        return self.bn(x)

def l2n(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

# ---------- backbone (swapable) ----------
def make_backbone():
    # EfficientNet-Lite0-ish via torchvision (good trade-off)
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # strip classifier
    feats = nn.Sequential(
        m.features,
        nn.Conv2d(1280, 512, kernel_size=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
    )
    return feats, 512

# ---------- main extractor ----------
class ReIDLikeExtractor:
    device: str
    def __init__(self, dim: int = 256, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        feats, c = make_backbone()
        self.backbone = feats.to(self.device).eval()
        self.pool = GeM().to(self.device).eval()
        self.bnneck = BNNeck(c).to(self.device).eval()
        self.fc = nn.Linear(c, dim, bias=False).to(self.device).eval()

        self.tf = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        # whitening (PCA) params
        self.W = None  # (dim, dim)
        self.mu = None # (dim,)
        # prototypes & cohort
        self.prototypes: Dict[str, np.ndarray] = {}
        self.cohort: List[np.ndarray] = []

        for p in self.parameters():
            p.requires_grad_(False)

    def parameters(self):
        for m in [self.backbone, self.pool, self.bnneck, self.fc]:
            yield from m.parameters()

    def reset(self) -> None:
        self.W, self.mu = None, None
        self.prototypes.clear()
        self.cohort.clear()

    @torch.inference_mode()
    def _forward(self, img: Image.Image) -> np.ndarray:
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device, memory_format=torch.channels_last)
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        f = self.bnneck(f)
        f = self.fc(f)
        f = l2n(f).squeeze(0).detach().cpu().numpy()
        return f

    def _post(self, f: np.ndarray) -> np.ndarray:
        # optional whitening if available
        if self.W is not None and self.mu is not None:
            f = (f - self.mu) @ self.W
        # final L2
        n = np.linalg.norm(f) + 1e-8
        return f / n

    def fine_tune(self, image_paths: Sequence[Path]) -> None:
        """
        Lightweight domain adaptation:
        - Build cohort from random backgrounds (first 128 imgs).
        - Learn PCA whitening on those background embeddings.
        """
        embs = []
        for p in image_paths[:128]:
            try:
                e = self._post(self._forward(Image.open(p)))
                embs.append(e)
            except Exception:
                continue
        if not embs:
            return
        X = np.stack(embs, 0)
        self.cohort = embs[:128]

        # PCA whitening to dim
        mu = X.mean(0, keepdims=True)
        Xc = X - mu
        eps = 1e-6
        n = Xc.shape[0]
        cov = (Xc.T @ Xc) / max(n - 1, 1)  # covariance in feature space
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))
        W = eigvecs @ inv_sqrt @ eigvecs.T  # dim x dim whitening matrix
        self.W = W.astype(np.float32)
        self.mu = mu.squeeze(0).astype(np.float32)

    def add_reference(self, key: str, images: Sequence[Image.Image]) -> None:
        """
        Create a prototype from multiple augmented views of the reference.
        """
        vecs = []
        for im in images:
            vecs.append(self._post(self._forward(im)))
        proto = np.mean(np.stack(vecs, 0), axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        self.prototypes[key] = proto

    def score(self, crop: Image.Image, key: str) -> float:
        """
        Cosine similarity with cohort normalization (z-norm).
        """
        q = self._post(self._forward(crop))
        s = float(np.dot(q, self.prototypes[key]))
        if self.cohort:
            co = np.dot(np.stack(self.cohort, 0), q)
            mu, std = float(co.mean()), float(co.std() + 1e-6)
            s = (s - mu) / std  # z-norm
        return s

    def extract(self, image: Image.Image) -> np.ndarray:
        return self._post(self._forward(image))
