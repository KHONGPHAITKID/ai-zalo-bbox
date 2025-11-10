from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..config import AssignerConfig
from ..types import Box, Track
from ..util.boxes import iou_matrix
from .kalman import KalmanFilter


def ensure_kf(track: Track, kalman: KalmanFilter) -> None:
    if track.kf is None:
        if track.last_box is None:
            return
        track.kf = kalman.initiate(track.last_box)


def predict_tracks(tracks: Sequence[Track], kalman: KalmanFilter) -> None:
    for track in tracks:
        ensure_kf(track, kalman)
        if track.kf is not None:
            track.kf = kalman.predict(track.kf)


def build_cost_matrix(
    tracks: Sequence[Track],
    candidates: Sequence[Box],
    cfg: AssignerConfig,
    kalman: KalmanFilter | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not tracks or not candidates:
        empty = np.zeros((len(tracks), len(candidates)), dtype=np.float32)
        return empty, empty
    kalman = kalman or KalmanFilter()
    track_boxes = []
    for track in tracks:
        if track.kf is not None:
            track_boxes.append(kalman.to_xyxy(track.kf))
        elif track.last_box is not None:
            track_boxes.append(track.last_box.xyxy)
        else:
            track_boxes.append(np.zeros(4, dtype=np.float32))

    candidate_arrays = [cand.xyxy for cand in candidates]
    ious = iou_matrix(track_boxes, candidate_arrays)
    costs = np.zeros_like(ious)
    for t_idx, track in enumerate(tracks):
        for c_idx, cand in enumerate(candidates):
            app_cost = 1.0 - cand.score_app
            costs[t_idx, c_idx] = cfg.alpha_iou * (1 - ious[t_idx, c_idx]) + cfg.beta_app * app_cost
    return costs, ious


def hungarian_with_gates(
    costs: np.ndarray,
    ious: np.ndarray,
    tracks: Sequence[Track],
    candidates: Sequence[Box],
    cfg: AssignerConfig,
    sim_gate: float | None = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    matches: List[Tuple[int, int]] = []
    unmatched_tracks = list(range(len(tracks)))
    unmatched_cands = list(range(len(candidates)))
    if costs.size == 0:
        return matches, unmatched_tracks, unmatched_cands

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        row_ind, col_ind = linear_sum_assignment(costs)
        pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
    except Exception:
        pairs = _greedy_assign(costs)

    used_tracks = set()
    used_cands = set()
    sim_threshold = sim_gate if sim_gate is not None else cfg.sim_gate
    for ti, ci in pairs:
        cand = candidates[ci]
        track = tracks[ti]
        iou = ious[ti, ci] if ious.size else 0.0
        if cand.score_app < sim_threshold:
            continue
        if iou < cfg.iou_gate:
            continue
        if costs[ti, ci] > cfg.cost_gate:
            continue
        matches.append((ti, ci))
        used_tracks.add(ti)
        used_cands.add(ci)

    unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
    unmatched_cands = [i for i in range(len(candidates)) if i not in used_cands]
    return matches, unmatched_tracks, unmatched_cands


def _greedy_assign(costs: np.ndarray) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    taken_tracks: set[int] = set()
    taken_cands: set[int] = set()
    flat = np.argsort(costs, axis=None)
    for idx in flat:
        ti, ci = divmod(idx, costs.shape[1])
        if ti in taken_tracks or ci in taken_cands:
            continue
        pairs.append((ti, ci))
        taken_tracks.add(ti)
        taken_cands.add(ci)
    return pairs


def select_main_track(tracks: Sequence[Track]) -> Track | None:
    confirmed = [t for t in tracks if t.state == "Confirmed"]
    if confirmed:
        return max(confirmed, key=lambda t: (t.last_conf, -t.time_since_update))
    tentative = [t for t in tracks if t.state == "Tentative"]
    return tentative[0] if tentative else None
