from __future__ import annotations

import glob
import pickle
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_ERROR = exc
else:
    _CV2_ERROR = None
import numpy as np
import torch
import torch.nn.functional as F
try:  # pragma: no cover - optional dependency
    from torchvision.ops import nms
except Exception as exc:  # pragma: no cover
    nms = None  # type: ignore
    _TV_ERROR = exc
else:
    _TV_ERROR = None
try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


class _NullProgress:
    def __init__(self, total: Optional[int] = None, desc: str | None = None, unit: str | None = None) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit

    def update(self, _count: int) -> None:  # pragma: no cover - noop
        pass

    def close(self) -> None:  # pragma: no cover - noop
        pass

try:  # pragma: no cover - heavy optional dependency
    from detectron2.config import get_cfg
    from tools.train_net import Trainer
except Exception:  # pragma: no cover
    get_cfg = None  # type: ignore
    Trainer = None  # type: ignore


def _allow_pickle_globals() -> None:
    try:
        extra = [np.dtype, np.core.multiarray.scalar]  # type: ignore[attr-defined]
        float64_cls = getattr(np, "float64", None)
        dtype_mod = getattr(np, "dtypes", None)
        if dtype_mod is not None:
            float64_dtype_cls = getattr(dtype_mod, "Float64DType", None)
            if float64_dtype_cls is not None:
                extra.append(float64_dtype_cls)
        if float64_cls is not None:
            extra.append(type(np.array([0.0]).dtype))
        torch.serialization.add_safe_globals(extra)
    except Exception:
        pass


class DevitWrapper:
    """Thin wrapper around the official DE-ViT demo utilities."""

    def __init__(
        self,
        device: str = "cpu",
        config_file: str = "configs/open-vocabulary/lvis/vitl.yaml",
        rpn_config_file: str = "configs/RPN/mask_rcnn_R_50_FPN_1x.yaml",
        model_path: str = "weights/trained/open-vocabulary/lvis/vitl_0069999.pth",
        topk: int = 1,
        mask_on: bool = True,
        label_names: Optional[List[str]] = None,
        proto_model: str = "dinov2_vitl14",
    ) -> None:
        if get_cfg is None or Trainer is None:
            raise RuntimeError(
                "DE-ViT dependencies are missing. Make sure detectron2 and the "
                "official DE-ViT repo (tools.train_net) are on PYTHONPATH."
            )
        self.device = torch.device(device)
        self.config_file = config_file
        self.rpn_config_file = rpn_config_file
        self.model_path = model_path
        self.topk = topk
        self.mask_on = mask_on
        self.model = self._load_model()
        self.label_names = label_names or ["target"]
        self.proto_model = proto_model

    def _load_model(self) -> torch.nn.Module:
        cfg = get_cfg()
        cfg.merge_from_file(self.config_file)
        cfg.DE.OFFLINE_RPN_CONFIG = self.rpn_config_file
        cfg.DE.TOPK = self.topk
        cfg.MODEL.MASK_ON = bool(self.mask_on)
        cfg.MODEL.DEVICE = "cuda" if self.device.type == "cuda" else "cpu"
        cfg.freeze()
        _allow_pickle_globals()
        model = Trainer.build_model(cfg).to(self.device)
        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    def build_prototype(self, support_imgs: List[np.ndarray]) -> torch.Tensor:
        dinov2 = torch.hub.load("facebookresearch/dinov2", self.proto_model).to(self.device)
        dinov2.eval()
        patch_size = getattr(getattr(dinov2, "patch_embed", None), "patch_size", (14, 14))
        patch_h, patch_w = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        feats: List[torch.Tensor] = []
        with torch.no_grad():
            for img in support_imgs:
                t = self._to_tensor_rgb(img).to(self.device)
                t = self._pad_to_patch_multiple(t, patch_h, patch_w)
                feat = dinov2(t)
                if feat.ndim == 4:
                    feat = feat.mean(dim=(2, 3))
                elif feat.ndim == 3:
                    feat = feat.mean(dim=1)
                feats.append(feat.squeeze(0))
        proto = torch.stack(feats, dim=0).mean(dim=0, keepdim=True)
        proto = torch.nn.functional.normalize(proto, dim=1)
        return proto

    def set_category_space(self, prototypes: torch.Tensor, label_names: Optional[List[str]] = None) -> None:
        if label_names is not None:
            self.label_names = label_names
        if prototypes.shape[0] < 2:
            blank = torch.zeros((1, prototypes.shape[1]), device=prototypes.device)
            prototypes = torch.cat([prototypes, blank], dim=0)
            if len(self.label_names) < 2:
                self.label_names = self.label_names + ["blank"]
        self.model.label_names = self.label_names
        self.model.test_class_weight = prototypes.to(self.device)

    def detect(self, frames: List[np.ndarray], conf_thr: float = 0.25) -> List[Tuple[np.ndarray, np.ndarray]]:
        outputs: List[Tuple[np.ndarray, np.ndarray]] = []
        with torch.no_grad():
            for frame in frames:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = image_rgb.shape[:2]
                batch_input = {
                    "image": torch.as_tensor(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).to(self.device),
                    "height": h,
                    "width": w,
                }
                result = self.model([batch_input])[0]
                inst = result["instances"]
                scores = inst.scores.detach().cpu()
                keep = scores >= conf_thr
                boxes = inst.pred_boxes.tensor[keep].detach().cpu().numpy() if keep.any() else np.zeros((0, 4), dtype=np.float32)
                scores_np = scores[keep].numpy() if keep.any() else np.zeros((0,), dtype=np.float32)
                outputs.append((boxes, scores_np))
        return outputs

    @staticmethod
    def _to_tensor_rgb(img: np.ndarray) -> torch.Tensor:
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        rgb = img[:, :, ::-1]
        tensor = torch.from_numpy(rgb.copy()).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor

    @staticmethod
    def _pad_to_patch_multiple(tensor: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        _, _, h, w = tensor.shape
        pad_h = (patch_h - (h % patch_h)) % patch_h
        pad_w = (patch_w - (w % patch_w)) % patch_w
        if pad_h == 0 and pad_w == 0:
            return tensor
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")


@dataclass
class Track:
    id: int
    box: np.ndarray
    score: float
    feat: np.ndarray
    last_frame: int
    age: int = 0
    hits: int = 1


class SimpleTracker:
    """Greedy IoU + cosine tracker to keep IDs stable."""

    def __init__(self, iou_thr: float = 0.3, feat_w: float = 0.3, max_age: int = 30) -> None:
        self.iou_thr = iou_thr
        self.feat_w = feat_w
        self.max_age = max_age
        self.tracks: List[Track] = []
        self.next_id = 1

    @staticmethod
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        xa1, ya1, xa2, ya2 = a
        xb1, yb1, xb2, yb2 = b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = max(0.0, (xa2 - xa1)) * max(0.0, (ya2 - ya1))
        area_b = max(0.0, (xb2 - xb1)) * max(0.0, (yb2 - yb1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-6
        nb = np.linalg.norm(b) + 1e-6
        return float(np.dot(a, b) / (na * nb))

    def _match(self, boxes: np.ndarray, feats: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        matches: List[Tuple[int, int]] = []
        used_tracks, used_dets = set(), set()
        for ti, track in enumerate(self.tracks):
            best_det, best_cost = -1, -1.0
            for di in range(len(boxes)):
                if di in used_dets:
                    continue
                iou_val = self.iou(track.box, boxes[di])
                cos_val = self.cosine(track.feat, feats[di])
                cost = (1 - self.feat_w) * iou_val + self.feat_w * cos_val
                if cost > best_cost:
                    best_cost = cost
                    best_det = di
            if best_det >= 0 and best_cost >= self.iou_thr:
                matches.append((ti, best_det))
                used_tracks.add(ti)
                used_dets.add(best_det)
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        unmatched_dets = [i for i in range(len(boxes)) if i not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    def step(self, frame_idx: int, boxes: np.ndarray, scores: np.ndarray, feats: np.ndarray) -> List[Track]:
        for track in self.tracks:
            track.age += 1
        if len(boxes) == 0:
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return self.tracks

        matches, unmatched_tracks, unmatched_dets = self._match(boxes, feats)
        for ti, di in matches:
            track = self.tracks[ti]
            track.box = boxes[di]
            track.score = float(scores[di])
            track.feat = feats[di]
            track.last_frame = frame_idx
            track.age = 0
            track.hits += 1

        for di in unmatched_dets:
            self.tracks.append(
                Track(
                    id=self.next_id,
                    box=boxes[di],
                    score=float(scores[di]),
                    feat=feats[di],
                    last_frame=frame_idx,
                    age=0,
                    hits=1,
                )
            )
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return self.tracks


@dataclass
class DetResult:
    frame: int
    tid: int
    score: float
    box: Tuple[float, float, float, float]


def _expand_support_paths(sources: str | Path | Sequence[str | Path]) -> List[Path]:
    if isinstance(sources, (str, Path)):
        items: Sequence[str | Path] = [sources]
    else:
        items = sources
    paths: List[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            jpgs = sorted(path.glob("*.jpg"))
            pngs = sorted(path.glob("*.png"))
            paths.extend(jpgs + pngs)
        else:
            globbed = sorted(Path(p) for p in glob.glob(str(path)))
            if globbed:
                paths.extend(globbed)
            elif path.exists():
                paths.append(path)
    return paths


def read_supports(sources: str | Path | Sequence[str | Path], max_side: int = 640) -> List[np.ndarray]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to read support images") from _CV2_ERROR
    paths = _expand_support_paths(sources)
    imgs: List[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(1.0, max_side / float(max(h, w))) if max(h, w) > max_side else 1.0
        if scale != 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        imgs.append(img)
    if not imgs:
        raise RuntimeError(f"No support images found for {sources}")
    return imgs


def _draw_box(canvas: np.ndarray, tr: Track) -> None:
    x1, y1, x2, y2 = map(int, tr.box)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"ID {tr.id} {tr.score:.2f}"
    cv2.putText(
        canvas,
        label,
        (x1, max(0, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        lineType=cv2.LINE_AA,
    )


def process_batch(
    frame_buf: List[np.ndarray],
    overlay_buf: List[np.ndarray],
    start_idx: int,
    devit: DevitWrapper,
    conf_thr: float,
    iou_nms: float,
    tracker: SimpleTracker,
    all_rows: List[DetResult],
    writer: cv2.VideoWriter,
) -> None:
    if nms is None:
        raise RuntimeError("torchvision.ops.nms is required for the DE-ViT pipeline") from _TV_ERROR
    detections = devit.detect(frame_buf, conf_thr=conf_thr)
    for i, (boxes, scores) in enumerate(detections):
        frame_id = start_idx + i
        if len(boxes) > 0:
            keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_nms).cpu().numpy()
            boxes = boxes[keep]
            scores = scores[keep]
        feats = np.zeros((len(boxes), 1), dtype=np.float32)
        tracks = tracker.step(frame_id, boxes, scores, feats)
        canvas = overlay_buf[i]
        for tr in tracks:
            if tr.last_frame != frame_id:
                continue
            _draw_box(canvas, tr)
            all_rows.append(DetResult(frame=frame_id, tid=tr.id, score=tr.score, box=tuple(float(x) for x in tr.box)))
        writer.write(canvas)


def run_devit_pipeline(
    video_path: str,
    supports: str | Path | Sequence[str | Path],
    out_dir: str,
    *,
    device: str = "cuda:0",
    conf_thr: float = 0.25,
    iou_nms: float = 0.5,
    batch_size: int = 8,
    iou_track: float = 0.3,
    feat_w: float = 0.0,
    max_age: int = 30,
    overlay_fps: Optional[float] = None,
    overlay_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    config_file: str = "configs/open-vocabulary/lvis/vitl.yaml",
    rpn_config_file: str = "configs/RPN/mask_rcnn_R_50_FPN_1x.yaml",
    model_path: str = "weights/trained/open-vocabulary/lvis/vitl_0069999.pth",
    label_name: str = "target",
    topk: int = 1,
    mask_on: bool = True,
    proto_model: str = "dinov2_vitl14",
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    overlay_path = overlay_path or os.path.join(out_dir, "overlay.mp4")
    csv_path = csv_path or os.path.join(out_dir, "detections.csv")

    if cv2 is None:
        raise RuntimeError("OpenCV is required for the DE-ViT pipeline") from _CV2_ERROR
    support_imgs = read_supports(supports)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if overlay_fps is None or overlay_fps <= 0:
        overlay_fps = fps if fps and fps > 0 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, overlay_fps, (width, height))

    devit = DevitWrapper(
        device=device,
        config_file=config_file,
        rpn_config_file=rpn_config_file,
        model_path=model_path,
        topk=topk,
        mask_on=mask_on,
        label_names=[label_name],
        proto_model=proto_model,
    )
    proto = devit.build_prototype(support_imgs)
    devit.set_category_space(proto, [label_name])

    tracker = SimpleTracker(iou_thr=iou_track, feat_w=feat_w, max_age=max_age)

    frame_buf: List[np.ndarray] = []
    overlay_buf: List[np.ndarray] = []
    all_rows: List[DetResult] = []
    frame_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    progress_cls = tqdm if tqdm is not None else _NullProgress
    progress = progress_cls(total=total_frames, desc="Processing", unit="frame")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if frame_buf:
                    process_batch(
                        frame_buf,
                        overlay_buf,
                        frame_idx - len(frame_buf),
                        devit,
                        conf_thr,
                        iou_nms,
                        tracker,
                        all_rows,
                        writer,
                    )
                    frame_buf.clear()
                    overlay_buf.clear()
                break
            frame_buf.append(frame)
            overlay_buf.append(frame.copy())
            frame_idx += 1
            if len(frame_buf) >= batch_size:
                process_batch(
                    frame_buf,
                    overlay_buf,
                    frame_idx - len(frame_buf),
                    devit,
                    conf_thr,
                    iou_nms,
                    tracker,
                    all_rows,
                    writer,
                )
                frame_buf.clear()
                overlay_buf.clear()
            progress.update(1)
    finally:
        progress.close()
        cap.release()
        writer.release()

    with open(csv_path, "w", newline="") as f:
        f.write("frame,id,score,x1,y1,x2,y2\n")
        for row in all_rows:
            x1, y1, x2, y2 = row.box
            f.write(
                f"{row.frame},{row.tid},{row.score:.4f},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}\n"
            )

    return overlay_path, csv_path
