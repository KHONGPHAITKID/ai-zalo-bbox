from __future__ import annotations

import numpy as np

from ..types import Box, KalmanState


class KalmanFilter:
    ndim, dt = 4, 1.0

    def __init__(self) -> None:
        self._motion_mat = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        self._update_mat = np.zeros((self.ndim, 2 * self.ndim), dtype=np.float32)
        self._update_mat[: self.ndim, : self.ndim] = np.eye(self.ndim, dtype=np.float32)
        self._std_pos = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self._std_vel = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    def initiate(self, box: Box) -> KalmanState:
        x1, y1, x2, y2 = box.xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        area = w * h
        ratio = w / max(1e-6, h)
        mean = np.array([cx, cy, area, ratio, 0, 0, 0, 0], dtype=np.float32)
        covariance = np.eye(2 * self.ndim, dtype=np.float32)
        return KalmanState(state=mean, covariance=covariance)

    def predict(self, state: KalmanState) -> KalmanState:
        process_noise = np.diag(np.concatenate([self._std_pos, self._std_vel]) ** 2)
        mean = self._motion_mat @ state.state
        covariance = self._motion_mat @ state.covariance @ self._motion_mat.T + process_noise
        return KalmanState(mean, covariance)

    def project(self, state: KalmanState) -> tuple[np.ndarray, np.ndarray]:
        measurement_noise = np.diag(self._std_pos ** 2)
        mean = self._update_mat @ state.state
        covariance = self._update_mat @ state.covariance @ self._update_mat.T + measurement_noise
        return mean, covariance

    def update(self, state: KalmanState, box: Box) -> KalmanState:
        measurement = self._xyxy_to_measurement(box.xyxy)
        mean, covariance = self.project(state)
        kalman_gain = state.covariance @ self._update_mat.T @ np.linalg.inv(covariance)
        innovation = measurement - mean
        new_mean = state.state + kalman_gain @ innovation
        new_cov = (np.eye(len(state.state)) - kalman_gain @ self._update_mat) @ state.covariance
        return KalmanState(new_mean.astype(np.float32), new_cov.astype(np.float32))

    def to_xyxy(self, state: KalmanState) -> np.ndarray:
        cx, cy, area, ratio = state.state[:4]
        w = np.sqrt(area * ratio)
        h = area / np.maximum(w, 1e-6)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

    @staticmethod
    def _xyxy_to_measurement(box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        area = w * h
        ratio = w / max(h, 1e-6)
        return np.array([cx, cy, area, ratio], dtype=np.float32)
