from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .config import DEFAULT_CONFIG_PATH, PipelineConfig, load_config
from .embed import build_embedder
from .gallery import Gallery, MemoryBank, build_reference_gallery
from .match import SimilarityScorer, conf_fuse
from .post import TemporalSmoother
from .proposals import build_proposal_generator
from .track import (
    KalmanFilter,
    OStrackFallback,
    build_cost_matrix,
    hungarian_with_gates,
    predict_tracks,
    select_main_track,
)
from .types import Box, Track
from .util import build_frame_iterator, crop_from_boxes


def _load_reference_paths(reference_root: str | Path | Sequence[str]) -> List[Path]:
    if isinstance(reference_root, (str, Path)):
        root = Path(reference_root)
        if root.is_dir():
            return sorted(list(root.glob("*.jpg"))) or sorted(list(root.glob("*.png")))
        if root.is_file():
            return [root]
        return [Path(p) for p in sorted(Path().glob(str(root)))]
    return [Path(p) for p in reference_root]


def _new_track(box: Box, next_id: int, kalman: KalmanFilter) -> Track:
    track = Track(id=next_id, last_box=box, last_conf=box.score_app)
    track.kf = kalman.initiate(box)
    track.history.append(box)
    return track


def _update_track(track: Track, box: Box, kalman: KalmanFilter) -> None:
    if track.kf is None:
        track.kf = kalman.initiate(box)
    else:
        track.kf = kalman.update(track.kf, box)
    track.last_box = box
    track.history.append(box)


def run_pipeline(
    video_path: str,
    references_root: str,
    output_path: Optional[str] = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> List[Dict[str, object]]:
    cfg = load_config(config_path)
    proposal_generator = build_proposal_generator(cfg.proposals)
    embedder = build_embedder(cfg.embedder, cfg.runtime)
    reference_paths = _load_reference_paths(references_root)
    if not reference_paths:
        raise FileNotFoundError(f"No reference images found under {references_root}")
    gallery = build_reference_gallery(reference_paths, cfg.gallery, embedder, cfg.runtime)
    memory = MemoryBank(cfg.gallery.memory_max, cfg.gallery.memory_add_sim_cap)
    similarity = SimilarityScorer(cfg.similarity)
    kalman = KalmanFilter()
    tracker = OStrackFallback(cfg.tracker)
    smoother = TemporalSmoother(cfg.post)
    frame_iter = build_frame_iterator(video_path, cfg.video.fps_override)

    tracks: List[Track] = []
    next_id = 1
    outputs: List[Dict[str, object]] = []

    for frame_idx, frame in frame_iter:
        proposals = proposal_generator.generate(frame)
        if cfg.proposals.topk_embed and len(proposals) > cfg.proposals.topk_embed:
            proposals = sorted(proposals, key=lambda b: b.score_det, reverse=True)[: cfg.proposals.topk_embed]
        crops, boxes = crop_from_boxes(frame, proposals)
        embeddings = embedder.encode(crops, cfg.runtime.batch_embed) if crops else np.zeros((0, 1), dtype=np.float32)
        gallery_with_memory = gallery.with_memory(memory)
        sim_result = similarity.score(embeddings, gallery_with_memory.embeddings)

        for i, box in enumerate(boxes):
            if i < len(sim_result.sim01):
                box.score_app = float(sim_result.sim01[i])
                box.crop_idx = i
        adaptive_gate = max(cfg.assigner.sim_gate, similarity.threshold())
        negatives = [score for score in sim_result.sim01 if score < adaptive_gate]
        similarity.update_background(negatives)

        candidates = [b for b in boxes if b.score_app >= adaptive_gate or b.score_det >= cfg.proposals.conf_thres + 0.05]

        for track in tracks:
            track.time_since_update += 1

        predict_tracks(tracks, kalman)
        cost_matrix, ious = build_cost_matrix(tracks, candidates, cfg.assigner, kalman)
        matches, unmatched_tracks, unmatched_candidates = hungarian_with_gates(
            cost_matrix,
            ious,
            tracks,
            candidates,
            cfg.assigner,
            sim_gate=adaptive_gate,
        )

        for ti, ci in matches:
            track = tracks[ti]
            cand = candidates[ci]
            conf = conf_fuse(cand.score_app, cand.score_det)
            cand.score_app = conf
            _update_track(track, cand, kalman)
            track.last_conf = conf
            track.state = "Confirmed" if track.state == "Tentative" and conf >= cfg.assigner.new_track_sim else track.state
            track.time_since_update = 0

        for idx in unmatched_tracks:
            track = tracks[idx]
            if track.time_since_update > cfg.assigner.max_age:
                track.state = "Lost"

        for ci in unmatched_candidates:
            cand = candidates[ci]
            if cand.score_app >= cfg.assigner.new_track_sim:
                track = _new_track(cand, next_id, kalman)
                tracks.append(track)
                next_id += 1

        main = select_main_track(tracks)
        if main and main.time_since_update >= cfg.tracker.gap_trigger and main.last_box is not None:
            tracker.update_template(frame, main.last_box)
            trk_box, trk_conf = tracker.track(frame)
            if trk_conf >= 0.5:
                fallback_box = Box(xyxy=trk_box, score_det=0.0, score_app=trk_conf)
                _update_track(main, fallback_box, kalman)
                main.last_conf = 0.5 * main.last_conf + 0.5 * trk_conf
                main.time_since_update = 0

        smoothed_box, _ = smoother.update(main.last_box if main else None)
        present = main is not None and main.time_since_update <= cfg.post.track_only_max
        output_entry: Dict[str, object] = {
            "frame": int(frame_idx),
            "present": bool(present),
            "conf": float(main.last_conf if main else 0.0),
            "bbox": smoothed_box.tolist() if smoothed_box is not None else None,
        }
        outputs.append(output_entry)

        # if present and main and main.last_conf >= 0.7 and main.last_box and main.last_box.crop_idx is not None:
        #     embedding = embeddings[main.last_box.crop_idx : main.last_box.crop_idx + 1]
        #     memory.maybe_add(embedding.squeeze(0))
        break

    if output_path:
        Path(output_path).write_text(json.dumps(outputs, indent=2))

    return outputs
