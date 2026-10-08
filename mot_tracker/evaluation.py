"""CLEAR-MOT / identity metrics following the MOTChallenge protocol.

Protocol (same as the official devkit / TrackEval for MOT17 & MOT20):
  * Only GT rows with class == 1 (pedestrian) and consider-flag == 1 count.
  * Distractor classes (2 person-on-vehicle, 7 static person, 8 distractor,
    12 reflection) are "don't care": a tracker box matched (IoU >= 0.5) to one
    of them is removed before scoring, so it is neither a TP nor an FP.
  * A tracker box is a TP if IoU >= 0.5 with a GT box (Hungarian per frame).

Metrics are computed with py-motmetrics: MOTA, MOTP, IDF1, ID switches,
fragmentations, MT/ML, FP, FN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import motmetrics as mm
from scipy.optimize import linear_sum_assignment

from .matching import iou_matrix

DISTRACTOR_CLASSES = {2, 7, 8, 12}

METRICS = ["mota", "motp", "idf1", "idp", "idr", "recall", "precision", "num_switches",
           "num_fragmentations", "mostly_tracked", "partially_tracked", "mostly_lost",
           "num_false_positives", "num_misses", "num_unique_objects", "num_objects"]

NAMES = {"mota": "MOTA", "motp": "MOTP", "idf1": "IDF1", "idp": "IDP", "idr": "IDR", "recall": "Rcll",
         "precision": "Prcn", "num_switches": "IDSW", "num_fragmentations": "Frag",
         "mostly_tracked": "MT", "partially_tracked": "PT", "mostly_lost": "ML",
         "num_false_positives": "FP", "num_misses": "FN", "num_unique_objects": "GT_IDs",
         "num_objects": "GT_boxes"}


def _tlbr(rows):
    b = rows[:, 2:6].copy()
    b[:, 2:] += b[:, :2]
    return b


def evaluate_sequence(gt: np.ndarray, res: np.ndarray, iou_thresh: float = 0.5):
    """Return (motmetrics accumulator, per-frame cleaned data dict)."""
    acc = mm.MOTAccumulator(auto_id=False)
    frames = sorted(set(gt[:, 0].astype(int)) | set(res[:, 0].astype(int) if len(res) else []))
    has_class = gt.shape[1] >= 8
    cleaned = {}
    for f in frames:
        g = gt[gt[:, 0] == f]
        r = res[res[:, 0] == f] if len(res) else np.zeros((0, 7))
        if has_class:
            cls = g[:, 7].astype(int)
            valid = (g[:, 6] == 1) & (cls == 1)
            distr = np.isin(cls, list(DISTRACTOR_CLASSES))
        else:
            valid = g[:, 6] == 1
            distr = np.zeros(len(g), bool)
        # --- remove tracker boxes that sit on distractors
        if len(r) and distr.any():
            cand = valid | distr
            gc = g[cand]
            iou = iou_matrix(_tlbr(gc), _tlbr(r))
            cost = np.where(iou >= iou_thresh - 1e-9, 1 - iou, 1e6)
            ri, ci = linear_sum_assignment(cost)
            drop = [c for rr, c in zip(ri, ci) if cost[rr, c] < 1e6 and distr[cand][rr]]
            r = np.delete(r, drop, axis=0)
        gv = g[valid]
        dist = 1 - iou_matrix(_tlbr(gv), _tlbr(r)) if len(gv) and len(r) else np.zeros((len(gv), len(r)))
        dist = np.where(dist > 1 - iou_thresh, np.nan, dist)
        acc.update(gv[:, 1].astype(int).tolist(), r[:, 1].astype(int).tolist(), dist, frameid=f)
        cleaned[f] = (gv, r)
    return acc, cleaned


def compute_metrics(accs: list, names: list[str]) -> pd.DataFrame:
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=METRICS, names=names, generate_overall=True)
    return summary.rename(columns=NAMES)


def format_table(df: pd.DataFrame) -> str:
    """Markdown table with percentages for ratio metrics."""
    pct = ["MOTA", "MOTP", "IDF1", "IDP", "IDR", "Rcll", "Prcn"]
    out = df.copy()
    for c in pct:
        if c in out:
            # motmetrics MOTP is a distance (1 - IoU); report as IoU overlap like the benchmark.
            vals = (1 - out[c]) if c == "MOTP" else out[c]
            out[c] = (vals * 100).map(lambda v: f"{v:.1f}")
    for c in out.columns:
        if c not in pct:
            out[c] = out[c].map(lambda v: f"{int(v)}")
    out.index.name = "Sequence"
    return out.to_markdown()
