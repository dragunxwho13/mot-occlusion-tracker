"""Camera motion compensation (CMC).

Several MOT17 sequences (05, 10, 11, 13) are filmed from a moving camera. A
constant-velocity Kalman filter cannot tell camera motion from person motion,
so predictions drift and boxes get mis-associated. We estimate a global
similarity transform between consecutive frames from sparse optical flow on
background corners (person boxes masked out) and warp every track's state
with it before association. A ceiling camera is static, so this is a no-op
there -- the estimated transform is ~identity.
"""
from __future__ import annotations

import cv2
import numpy as np


class CameraMotionCompensator:
    def __init__(self, downscale: int = 2, max_corners: int = 1000):
        self.downscale = downscale
        self.max_corners = max_corners
        self.prev_gray = None
        self.prev_pts = None

    def estimate(self, frame: np.ndarray, det_tlbr: np.ndarray | None = None) -> np.ndarray:
        """Return 2x3 affine mapping previous-frame coords -> current-frame coords."""
        H = np.eye(2, 3, dtype=np.float64)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            gray = cv2.resize(gray, (gray.shape[1] // self.downscale, gray.shape[0] // self.downscale))
        mask = np.full_like(gray, 255)
        h, w = gray.shape
        mask[: int(0.02 * h)] = 0
        mask[int(0.98 * h):] = 0
        if det_tlbr is not None:
            for x1, y1, x2, y2 in (np.asarray(det_tlbr) / self.downscale).astype(int):
                mask[max(0, y1): max(0, y2), max(0, x1): max(0, x2)] = 0
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners, qualityLevel=0.01,
                                      minDistance=1, blockSize=3, mask=mask)
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_gray, self.prev_pts = gray, pts
            return H
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
        good_prev = self.prev_pts[status.ravel() == 1]
        good_next = nxt[status.ravel() == 1]
        if len(good_prev) >= 10:
            M, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC)
            if M is not None:
                H = M.astype(np.float64)
                H[:, 2] *= self.downscale
        self.prev_gray, self.prev_pts = gray, pts
        return H


def apply_cmc(means: np.ndarray, covs: np.ndarray, H: np.ndarray):
    """Warp Kalman states (xyah) with a 2x3 similarity transform, in place-safe copy."""
    if len(means) == 0:
        return means, covs
    R = H[:, :2]
    t = H[:, 2]
    scale = np.sqrt(abs(np.linalg.det(R)))
    means = means.copy()
    covs = covs.copy()
    means[:, :2] = means[:, :2] @ R.T + t
    means[:, 4:6] = means[:, 4:6] @ R.T
    means[:, 3] *= scale
    means[:, 7] *= scale
    R8 = np.eye(8)
    R8[:2, :2] = R
    R8[4:6, 4:6] = R
    covs = R8 @ covs @ R8.T
    return means, covs
