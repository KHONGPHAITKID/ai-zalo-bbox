import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import timm

class RMACFeatureExtractor:
    def __init__(self, device="cuda", backbone_name="mobilenetv3_large_100", out_dim=512, rmac_levels=(1,2,3)):
        self.device = device
        self.model = timm.create_model(backbone_name, features_only=True, pretrained=True).to(device).eval()
        self.rmac_levels = rmac_levels
        self.pca = None  # optional: fit on your `fine_tune` set to reduce to out_dim
        self.out_dim = out_dim

    def reset(self): pass

    @torch.no_grad()
    def fine_tune(self, image_paths):
        # Optional: fit PCA-whitening on descriptors extracted from these paths
        # (Collect many RMAC vectors, compute mean+proj; store self.pca)
        pass

    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        x = self._preprocess(image).to(self.device)
        feats = self.model(x)[-1]            # [B, C, H, W]
        desc = self._rmac(feats)             # [B, C]
        desc = F.normalize(desc, dim=1)
        v = desc[0].detach().cpu().numpy()
        if self.pca is not None:
            v = self._apply_pca(v)           # L2 after PCA
        return v

    def _preprocess(self, pil, size=352):
        pil = pil.convert("RGB")
        w, h = pil.size
        s = size / max(w, h)
        nw, nh = int(w*s), int(h*s)
        pil = pil.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114,114,114))
        canvas.paste(pil, ((size-nw)//2, (size-nh)//2))
        x = torch.from_numpy(np.array(canvas)).float().permute(2,0,1)/255.0
        x = x.unsqueeze(0)
        return x

    def _gem(self, x, p=3.0, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0/p)

    def _rmac(self, feat):
        # feat: [B, C, H, W]
        B, C, H, W = feat.shape
        regions = []
        for L in self.rmac_levels:
            # LxL grid of regions with ~40–80% overlap
            for i in range(L):
                for j in range(L):
                    h1 = int((i+0.5)*H/L - 0.5*H/L)
                    h2 = int((i+0.5)*H/L + 0.5*H/L)
                    w1 = int((j+0.5)*W/L - 0.5*W/L)
                    w2 = int((j+0.5)*W/L + 0.5*W/L)
                    h1, w1 = max(0,h1), max(0,w1)
                    h2, w2 = min(H,h2), min(W,w2)
                    r = feat[:, :, h1:h2, w1:w2]
                    regions.append(F.normalize(self._gem(r), dim=1))  # [B, C, 1, 1]
        # max over regions
        R = torch.stack(regions, dim=0).squeeze(-1).squeeze(-1)  # [R, B, C]
        return torch.max(R, dim=0).values                         # [B, C]

    def _apply_pca(self, v: np.ndarray) -> np.ndarray:
        # Example: v' = W (v - mu), then L2
        v = (v - self.pca["mu"]) @ self.pca["W"].T
        v = v / (np.linalg.norm(v) + 1e-8)
        return v
