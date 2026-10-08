"""Failure analysis: why did each ID switch happen?

For every SWITCH event reported by the evaluator (GT person g changes from
tracker ID h_old to h_new at frame f) we look back to the last frame g was
correctly tracked (f_prev) and measure:

  gap          frames g went unmatched before the switch (track lost)
  min_vis      lowest GT visibility of g in [f_prev, f]  (MOT17 gives 0..1)
  crowd_iou    highest IoU between g and any other person in [f_prev, f]
  speed        centre displacement per frame / box height around f
  transfer     h_new (or h_old) was attached to a *different* person g2 --
               i.e. two identities got exchanged, not just a new ID spawned
  app_sim      appearance similarity of g and g2 (same embedder as tracker)
  new_track    h_new was born at f (the old track had died -> fragmentation)

Flags -> causes (multi-label, plus one primary cause by priority):
  occlusion          min_vis < 0.5  or  crowd_iou > 0.3  or  gap > 0
  long_occlusion     gap > LOST buffer (track already deleted, new ID inevitable)
  fast_motion        speed > 0.08 box-heights / frame (~ 3x normal walking)
  similar_appearance transfer and app_sim > 0.85
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .appearance import HistogramEmbedder
from .matching import iou_matrix

VIS_T, CROWD_T, SPEED_T, SIM_T = 0.5, 0.3, 0.08, 0.85


@dataclass
class SwitchCase:
    seq: str
    frame: int
    gt_id: int
    old_hid: int
    new_hid: int
    prev_frame: int
    gap: int
    min_vis: float
    crowd_iou: float
    other_gt: int
    speed: float
    transfer: bool
    app_sim: float
    new_track: bool
    occlusion: bool = False
    long_occlusion: bool = False
    fast_motion: bool = False
    similar_appearance: bool = False
    primary: str = ""


def _gt_index(gt: np.ndarray):
    valid = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)] if gt.shape[1] >= 8 else gt
    by_frame = {int(f): valid[valid[:, 0] == f] for f in np.unique(valid[:, 0])}
    by_id = {int(i): valid[valid[:, 1] == i] for i in np.unique(valid[:, 1])}
    return by_frame, by_id


def _box(row):
    return np.array([row[2], row[3], row[2] + row[4], row[3] + row[5]])


def analyse_switches(seq_name: str, gt: np.ndarray, events: pd.DataFrame, read_frame=None,
                     max_age: int = 30) -> list[SwitchCase]:
    by_frame, by_id = _gt_index(gt)
    ev = events[events["Type"].isin(["MATCH", "SWITCH"])].copy()
    ev = ev.dropna(subset=["OId", "HId"])
    ev["OId"] = ev["OId"].astype(int)
    ev["HId"] = ev["HId"].astype(int)
    ev = ev.sort_values("FrameId")
    first_seen_h = ev.groupby("HId")["FrameId"].min().to_dict()
    # (frame, hid) -> oid  for transfer detection
    h_to_o = {}
    for r in ev.itertuples():
        h_to_o.setdefault(r.HId, []).append((r.FrameId, r.OId))
    embedder = HistogramEmbedder() if read_frame is not None else None

    cases = []
    last_match: dict[int, tuple[int, int]] = {}
    for r in ev.itertuples():
        g, h, f = r.OId, r.HId, int(r.FrameId)
        if r.Type == "SWITCH" and g in last_match:
            f_prev, h_old = last_match[g]
            track = by_id[g]
            win = track[(track[:, 0] >= f_prev) & (track[:, 0] <= f)]
            min_vis = float(win[:, 8].min()) if gt.shape[1] >= 9 and len(win) else 1.0
            # crowding: max IoU with other people over the window
            crowd, other = 0.0, -1
            for row in win:
                others = by_frame.get(int(row[0]))
                if others is None:
                    continue
                others = others[others[:, 1] != g]
                if len(others):
                    ious = iou_matrix(_box(row)[None], np.array([_box(o) for o in others]))[0]
                    k = int(ious.argmax())
                    if ious[k] > crowd:
                        crowd, other = float(ious[k]), int(others[k, 1])
            # speed around the switch (centre displacement / height per frame)
            seg = track[(track[:, 0] >= f - 3) & (track[:, 0] <= f)]
            speed = 0.0
            if len(seg) >= 2:
                c = seg[:, 2:4] + seg[:, 4:6] / 2
                d = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(seg[1:, 0] - seg[:-1, 0], 1)
                speed = float(np.max(d / seg[1:, 5]))
            # identity transfer: did h_new previously belong to another person?
            prev_owners = [o for (ff, o) in h_to_o.get(h, []) if ff < f and o != g]
            transfer = bool(prev_owners)
            g2 = prev_owners[-1] if prev_owners else other
            # appearance similarity between g and g2 at the switch frame
            sim = float("nan")
            if embedder is not None and g2 in by_id:
                rows_f = by_frame.get(f, np.zeros((0, 9)))
                a = rows_f[rows_f[:, 1] == g]
                b = rows_f[rows_f[:, 1] == g2]
                if len(a) == 0 or len(b) == 0:
                    tb = by_id[g2]
                    b = tb[np.argsort(np.abs(tb[:, 0] - f))][:1]
                if len(a) and len(b):
                    if int(b[0, 0]) == f:
                        img = read_frame(f)
                        fa, fb = embedder(img, np.array([_box(a[0]), _box(b[0])]))
                    else:
                        fa = embedder(read_frame(f), np.array([_box(a[0])]))[0]
                        fb = embedder(read_frame(int(b[0, 0])), np.array([_box(b[0])]))[0]
                    sim = float(fa @ fb)
            gap = f - f_prev - 1
            c = SwitchCase(seq_name, f, g, h_old, h, f_prev, gap, min_vis, crowd, int(g2), speed, transfer,
                           sim, first_seen_h.get(h, f) == f)
            c.occlusion = min_vis < VIS_T or crowd > CROWD_T or gap > 0
            c.long_occlusion = gap > max_age
            c.fast_motion = speed > SPEED_T
            c.similar_appearance = transfer and not np.isnan(sim) and sim > SIM_T
            if c.long_occlusion:
                c.primary = "long occlusion (> buffer)"
            elif c.occlusion and c.similar_appearance:
                c.primary = "occlusion + similar appearance"
            elif c.occlusion:
                c.primary = "occlusion"
            elif c.similar_appearance:
                c.primary = "similar appearance"
            elif c.fast_motion:
                c.primary = "fast motion"
            else:
                c.primary = "other (detector jitter / missed det)"
            cases.append(c)
        last_match[g] = (f, h)
    return cases


def occlusion_recovery(events: pd.DataFrame, bins=(1, 5, 15, 30, 60, 10_000)) -> pd.DataFrame:
    """For every interval where a GT person was not tracked (missed) and then
    tracked again, was the ID kept (recovered) or changed (switched)?

    Bucketed by the length of the gap -- this directly measures "re-associate
    to the same ID on reappearance rather than spawning a new one".
    """
    ev = events[events["Type"].isin(["MATCH", "SWITCH"])].sort_values("FrameId")
    rows = []
    state: dict[int, tuple[int, int]] = {}
    for r in ev.itertuples():
        g, h, f = int(r.OId), int(r.HId), int(r.FrameId)
        if g in state:
            f_prev, h_prev = state[g]
            gap = f - f_prev - 1
            if gap >= 1:
                rows.append((gap, h == h_prev))
        state[g] = (f, h)
    df = pd.DataFrame(rows, columns=["gap", "same_id"])
    if df.empty:
        return pd.DataFrame(columns=["gap (frames)", "events", "same ID kept", "recovery %"])
    labels = [f"{a}-{b - 1}" if b < 10_000 else f"{a}+" for a, b in zip(bins[:-1], bins[1:])]
    df["bucket"] = pd.cut(df["gap"], bins=list(bins), right=False, labels=labels)
    out = df.groupby("bucket", observed=False).agg(events=("same_id", "size"), kept=("same_id", "sum"))
    out["recovery %"] = (100 * out["kept"] / out["events"].clip(lower=1)).round(1)
    out = out.reset_index().rename(columns={"bucket": "gap (frames)", "kept": "same ID kept"})
    return out


def cases_to_frame(cases: list[SwitchCase]) -> pd.DataFrame:
    return pd.DataFrame([asdict(c) for c in cases])
