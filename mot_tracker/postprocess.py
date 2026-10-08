"""Offline post-processing of finished tracks."""
from __future__ import annotations

import numpy as np


def interpolate_gaps(rows: np.ndarray, max_gap: int = 20) -> np.ndarray:
    """Linearly fill short gaps inside each track.

    When a person is occluded and later re-identified with the same ID, the
    frames in between have no box. Filling them recovers false negatives
    during the occlusion (raises MOTA / IDF1). Only usable offline (it needs
    the re-appearance), so results are reported with and without it.

    rows: MOT result array (frame, id, x, y, w, h, score, ...).
    """
    if len(rows) == 0:
        return rows
    out = [rows]
    for tid in np.unique(rows[:, 1]):
        tr = rows[rows[:, 1] == tid]
        tr = tr[np.argsort(tr[:, 0])]
        frames = tr[:, 0].astype(int)
        for a, b in zip(range(len(tr) - 1), range(1, len(tr))):
            gap = frames[b] - frames[a]
            if 1 < gap <= max_gap + 1:
                for f in range(frames[a] + 1, frames[b]):
                    t = (f - frames[a]) / gap
                    box = tr[a, 2:6] * (1 - t) + tr[b, 2:6] * t
                    new = tr[a].copy()
                    new[0], new[2:6], new[6] = f, box, -1  # score -1 marks interpolated
                    out.append(new[None])
    res = np.concatenate(out)
    return res[np.lexsort((res[:, 1], res[:, 0]))]


def drop_short_tracks(rows: np.ndarray, min_len: int = 0) -> np.ndarray:
    if min_len <= 1 or len(rows) == 0:
        return rows
    ids, counts = np.unique(rows[:, 1], return_counts=True)
    keep = set(ids[counts >= min_len])
    return rows[np.isin(rows[:, 1], list(keep))]
