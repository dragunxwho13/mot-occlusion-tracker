"""Constant-velocity Kalman filter in (cx, cy, aspect, height) space.

State vector (8-dim):  [cx, cy, a, h, vcx, vcy, va, vh]
Measurement (4-dim):   [cx, cy, a, h]   (a = w / h)

This is the parameterisation used by DeepSORT. Process/measurement noise is
scaled by the box height, so a large (near) person is allowed to move more
pixels per frame than a small (far) one -- important for ceiling / oblique
cameras where scale varies a lot across the frame.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

# 0.95 quantile of the chi-square distribution, indexed by degrees of freedom.
CHI2INV95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}


class KalmanFilter:
    def __init__(self, std_weight_position: float = 1.0 / 20, std_weight_velocity: float = 1.0 / 160):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

    # ------------------------------------------------------------------ init
    def initiate(self, measurement: np.ndarray):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    # --------------------------------------------------------------- predict
    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        h = mean[3]
        std_pos = [self._std_weight_position * h, self._std_weight_position * h, 1e-2, self._std_weight_position * h]
        std_vel = [self._std_weight_velocity * h, self._std_weight_velocity * h, 1e-5, self._std_weight_velocity * h]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def multi_predict(self, means: np.ndarray, covariances: np.ndarray):
        """Vectorised predict for N tracks (means: Nx8, covariances: Nx8x8)."""
        if len(means) == 0:
            return means, covariances
        h = means[:, 3]
        std_pos = np.stack([self._std_weight_position * h, self._std_weight_position * h,
                            1e-2 * np.ones_like(h), self._std_weight_position * h], axis=1)
        std_vel = np.stack([self._std_weight_velocity * h, self._std_weight_velocity * h,
                            1e-5 * np.ones_like(h), self._std_weight_velocity * h], axis=1)
        sqr = np.square(np.concatenate([std_pos, std_vel], axis=1))
        motion_cov = np.zeros((len(means), 8, 8))
        idx = np.arange(8)
        motion_cov[:, idx, idx] = sqr
        means = means @ self._motion_mat.T
        covariances = self._motion_mat @ covariances @ self._motion_mat.T + motion_cov
        return means, covariances

    # ---------------------------------------------------------------- update
    def project(self, mean: np.ndarray, covariance: np.ndarray, confidence: float = 1.0):
        h = mean[3]
        std = [self._std_weight_position * h, self._std_weight_position * h, 1e-1, self._std_weight_position * h]
        # Lower-confidence detections are trusted less (NSA-Kalman style).
        std = [(1.0 - confidence) * 2 * s + s for s in std]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, confidence: float = 1.0):
        projected_mean, projected_cov = self.project(mean, covariance, confidence)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), (covariance @ self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    # ---------------------------------------------------------------- gating
    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray,
                        only_position: bool = False):
        """Squared Mahalanobis distance between a track and each measurement (Nx4)."""
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)


# ---------------------------------------------------------------- box utils
def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
    ret = np.asarray(tlwh, dtype=float).copy()
    ret[..., :2] += ret[..., 2:] / 2
    ret[..., 2] /= ret[..., 3]
    return ret


def xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
    ret = np.asarray(xyah, dtype=float).copy()
    ret[..., 2] *= ret[..., 3]
    ret[..., :2] -= ret[..., 2:] / 2
    return ret


def tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
    ret = np.asarray(tlwh, dtype=float).copy()
    ret[..., 2:] += ret[..., :2]
    return ret


def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
    ret = np.asarray(tlbr, dtype=float).copy()
    ret[..., 2:] -= ret[..., :2]
    return ret
