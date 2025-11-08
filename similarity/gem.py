import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# --- GeM pooling over patch tokens ---
class GeM(torch.nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(p))
        self.eps = eps
    def forward(self, x):  # x: [B, N_tokens, D]
        x = torch.clamp(x, min=self.eps).pow(self.p)
        x = x.mean(dim=1)  # mean over tokens
        return x.pow(1.0 / self.p)

def vit_dino_embedder(device="cpu", input_size=224, scales=(1.0, 1 / math.sqrt(2), 0.5)):
    processor = AutoImageProcessor.from_pretrained("facebook/dino-vits16")
    model = AutoModel.from_pretrained("facebook/dino-vits16",
                                      output_hidden_states=False).to(device).eval()
    gem = GeM().to(device)

    @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        embs = []
        for s in scales:
            w, h = img.size
            img_s = img.resize((int(w*s), int(h*s)), Image.BICUBIC)

            # override HF resize to control target side
            inputs = processor(
                images=img_s,
                do_resize=True,
                size={"height": input_size, "width": input_size},
                resample=Image.BICUBIC,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = model(**inputs)  # last_hidden_state: [B, 1+N, D]
            tokens = out.last_hidden_state[:, 1:, :]  # drop CLS, keep patches
            pooled = gem(tokens)                      # [B, D]
            embs.append(torch.nn.functional.normalize(pooled, p=2, dim=-1))
        emb = torch.nn.functional.normalize(torch.stack(embs, dim=0).mean(0), p=2, dim=-1)  # [B, D]
        return emb[0].cpu()
    return encode


class GeMFeatureExtractor:
    """Drop-in replacement for DINOFeatureExtractor that uses multi-scale GeM pooling."""

    def __init__(
        self,
        device: str = "cpu",
        input_size: int = 224,
        scales: Sequence[float] = (1.0, 1 / math.sqrt(2), 0.5),
    ) -> None:
        self.device = device
        self.encoder = vit_dino_embedder(device=device, input_size=input_size, scales=scales)

    def reset(self) -> None:
        """Compatibility shim – GeM encoder is frozen, so nothing to reset."""
        return

    def fine_tune(self, image_paths: Sequence[Path]) -> None:
        """No-op to mirror the old DINO API."""
        _ = image_paths
        return

    def extract(self, image: Image.Image) -> np.ndarray:
        embedding = self.encoder(image.convert("RGB"))
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        return np.asarray(embedding, dtype=np.float32)
