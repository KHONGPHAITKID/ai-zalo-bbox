from __future__ import annotations

from pathlib import Path
from typing import Sequence, Protocol

import io
import math
import random
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from similarity.extractor import FeatureExtractor

# ---------- implementation ----------
class DinoV2Extractor(FeatureExtractor):
    """
    Few-shot friendly extractor:
      - Backbone: DINOv2 ViT-S/14 (frozen)
      - Head: 2-layer MLP -> 256-D, L2-normalized
      - Fine-tuning: instance-discrimination InfoNCE on your reference images
    """

    def __init__(
        self,
        device: str | None = None,
        out_dim: int = 256,
        img_size: int = 224,
        pad_context_px: int = 16,
        tta: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dim = out_dim
        self.img_size = img_size
        self.pad_context_px = pad_context_px
        self.tta = tta

        # Backbone: DINOv2 ViT-S/14 from timm (no classifier)
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,   # returns features
            img_size=img_size,
        ).to(self.device).eval()

        # Feature dim of ViT-S/14
        feat_dim = self.backbone.num_features

        # Small projector head (trained; backbone frozen)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.out_dim),
        ).to(self.device)

        # Keep an init copy for reset()
        self._init_state = {
            k: v.detach().cpu().clone() for k, v in self.projector.state_dict().items()
        }

        # Transforms
        self.base_eval_tf = transforms.Compose([
            PadSquare(self.pad_context_px),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(*imagenet_norm()),
        ])
        # Strong but realistic augs for UAV crops
        self.train_tf = transforms.Compose([
            PadSquare(self.pad_context_px),
            transforms.RandomResizedCrop(self.img_size, scale=(0.4, 1.0), antialias=True),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.RandomApply([SmallRotate()], p=0.5),
            transforms.RandomHorizontalFlip(),
            RandomDownUpScale(min_side=64),  # simulate low-res bbox then upscale
            transforms.ToTensor(),
            transforms.Normalize(*imagenet_norm()),
        ])

        # Lightweight TTA for extract()
        self.tta_tfs = [
            transforms.Compose([
                PadSquare(self.pad_context_px),
                transforms.Resize(self.img_size, antialias=True),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(*imagenet_norm()),
            ]),
            transforms.Compose([
                PadSquare(self.pad_context_px),
                transforms.Resize(self.img_size, antialias=True),
                transforms.CenterCrop(self.img_size),
                transforms.functional.hflip,
                transforms.ToTensor(),
                transforms.Normalize(*imagenet_norm()),
            ]),
        ] if self.tta else [self.base_eval_tf]

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    # ----- interface methods -----

    def reset(self) -> None:
        self.projector.load_state_dict({k: v.clone() for k, v in self._init_state.items()})
        self.projector.to(self.device).train(False)

    def fine_tune(
        self,
        image_paths: Sequence[Path],
        epochs: int = 10,
        batch_size: int = 64,
        lr_head: float = 1e-3,
        temperature: float = 0.07,
        seed: int = 42,
    ) -> None:
        """
        Contrastive finetune on your reference images.
        Treats each image as an instance ID; builds two augmented views → InfoNCE.
        Negatives come from other images in the batch.
        """
        if len(image_paths) == 0:
            return  # nothing to do

        rng = torch.Generator().manual_seed(seed)
        ds = TwoViewImageDataset([Path(p) for p in image_paths], self.train_tf)
        # Ensure sufficient batch negatives; fall back to small batch if needed
        bs = min(batch_size, max(8, 2 * math.ceil(len(ds) / 4)))

        use_cuda = isinstance(self.device, str) and self.device.startswith("cuda")
        dl = DataLoader(
            ds,
            batch_size=bs,
            shuffle=True,
            num_workers=0,  # avoid spawning new interpreter processes on macOS
            pin_memory=use_cuda,
            generator=rng,
            drop_last=len(ds) >= bs,
        )

        self.projector.train(True)
        opt = torch.optim.AdamW(self.projector.parameters(), lr=lr_head, weight_decay=1e-4)

        for epoch in range(epochs):
            for (x1, x2) in dl:
                x = torch.cat([x1, x2], dim=0).to(self.device, non_blocking=True)  # B*2,C,H,W
                with torch.no_grad():
                    feats = self.backbone(x)  # (2B, F)
                z = self.projector(feats)    # (2B, D)
                z = F.normalize(z, dim=-1)

                loss = info_nce_pairwise(z, temperature=temperature)  # positives are (0<->B, 1<->B+1, ...)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        self.projector.train(False)

    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Returns a 256-D L2-normalized embedding as np.ndarray.
        """
        self.backbone.eval(); self.projector.eval()
        embs = []
        for tf in self.tta_tfs:
            x = tf(image).unsqueeze(0).to(self.device)
            feat = self.backbone(x)               # (1, F)
            z = self.projector(feat)              # (1, D)
            z = F.normalize(z, dim=-1)
            embs.append(z.squeeze(0))
        z_mean = torch.stack(embs, dim=0).mean(0)
        z_mean = F.normalize(z_mean, dim=-1)
        return z_mean.detach().cpu().numpy()


# ---------- helpers ----------

def imagenet_norm():
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    return mean, std


class PadSquare:
    """Pad (with context) to roughly square before resize/crop, preserving aspect."""
    def __init__(self, pad_px: int = 16, fill: int = 114):
        self.pad_px = pad_px
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        img = ImageOps.expand(img, border=self.pad_px, fill=self.fill)
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        dw = (side - w) // 2
        dh = (side - h) // 2
        return ImageOps.expand(img, border=(dw, dh, side - w - dw, side - h - dh), fill=self.fill)


class SmallRotate:
    """±30° continuous rotation with expand=False to keep crop tight."""
    def __init__(self, max_deg: int = 30):
        self.max_deg = max_deg

    def __call__(self, img: Image.Image) -> Image.Image:
        deg = random.uniform(-self.max_deg, self.max_deg)
        return img.rotate(deg, resample=Image.BICUBIC)


class RandomDownUpScale:
    """
    Simulate low-res crops: downscale to a random side (>= min_side), then upscale back.
    """
    def __init__(self, min_side: int = 64):
        self.min_side = min_side

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        target_side = random.randint(self.min_side, min(w, h))
        if target_side >= min(w, h):
            return img
        # keep aspect
        if w < h:
            new_w = target_side
            new_h = int(h * (target_side / w))
        else:
            new_h = target_side
            new_w = int(w * (target_side / h))
        img_small = img.resize((new_w, new_h), resample=Image.BOX)  # antialiased downscale
        return img_small.resize((w, h), resample=Image.BICUBIC)


class TwoViewImageDataset(Dataset):
    """
    Given a list of image paths, returns two differently augmented views for InfoNCE.
    """
    def __init__(self, paths: Sequence[Path], tf):
        self.paths = [Path(p) for p in paths]
        self.tf = tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        v1 = self.tf(img)
        v2 = self.tf(img)
        return v1, v2


def info_nce_pairwise(z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    z: (2B, D) where positives are (i, i^B) i.e., pair (0,B), (1,B+1), ...
    """
    z = F.normalize(z, dim=-1)
    B2 = z.size(0)
    assert B2 % 2 == 0
    B = B2 // 2
    sim = (z @ z.t()) / temperature  # (2B,2B)
    # mask out self-similarity
    mask = torch.eye(B2, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # positives index: for i in [0..B-1]: pos of i is i+B; for i in [B..2B-1]: pos is i-B
    pos_idx = torch.arange(B2, device=z.device)
    pos_idx = torch.where(pos_idx < B, pos_idx + B, pos_idx - B)

    # logits & labels for cross-entropy
    labels = pos_idx
    loss = F.cross_entropy(sim, labels)
    return loss


# ---------- example usage ----------
if __name__ == "__main__":
    # Instantiate
    extractor = DinoV2Extractor(device=None, out_dim=256, img_size=224, tta=True)

    # Fine-tune on a handful of reference images (paths to your object crops)
    refs = [Path("../data/debug/img_1.jpg"), Path("../data/debug/img_2.jpg"), Path("../data/debug/img_3.jpg")]
    extractor.fine_tune(refs)  # you can pass epochs=5 for speed

    # Extract an embedding from a candidate bbox crop
    img = Image.open("../data/debug/detection_5.jpg").convert("RGB")
    emb = extractor.extract(img)  # np.ndarray shape (256,)
    print(emb.shape, np.linalg.norm(emb))
