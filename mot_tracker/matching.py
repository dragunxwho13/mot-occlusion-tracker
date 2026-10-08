"""Cost matrices and Hungarian assignment."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

INF_COST = 1e5


def iou_matrix(a_tlbr: np.ndarray, b_tlbr: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of boxes in (x1, y1, x2, y2)."""
    a = np.asarray(a_tlbr, dtype=float).reshape(-1, 4)
    b = np.asarray(b_tlbr, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def linear_assignment(cost: np.ndarray, thresh: float):
    """Hungarian assignment; pairs with cost > thresh are rejected.

    Returns (matches [K x 2], unmatched_rows, unmatched_cols).
    """
    if cost.size == 0:
        return np.empty((0, 2), dtype=int), list(range(cost.shape[0])), list(range(cost.shape[1]))
    c = cost.copy()
    c[c > thresh] = INF_COST
    rows, cols = linear_sum_assignment(c)
    matches, um_r, um_c = [], set(range(cost.shape[0])), set(range(cost.shape[1]))
    for r, col in zip(rows, cols):
        if c[r, col] <= thresh:
            matches.append((r, col))
            um_r.discard(r)
            um_c.discard(col)
    return np.asarray(matches, dtype=int).reshape(-1, 2), sorted(um_r), sorted(um_c)


def embedding_distance(track_feats: np.ndarray, det_feats: np.ndarray) -> np.ndarray:
    """Cosine distance in [0, 2]; features are assumed L2-normalised."""
    if len(track_feats) == 0 or len(det_feats) == 0:
        return np.zeros((len(track_feats), len(det_feats)))
    return np.clip(1.0 - track_feats @ det_feats.T, 0.0, 2.0)
