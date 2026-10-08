"""Fast unit tests (no images, no weights):  python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot_tracker import OcclusionAwareTracker, TrackerConfig  # noqa: E402
from mot_tracker.matching import iou_matrix, linear_assignment  # noqa: E402
from mot_tracker.postprocess import interpolate_gaps  # noqa: E402


def box(cx, cy=300, w=50, h=120, s=0.9):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, s]


def run(cfg, frames):
    trk = OcclusionAwareTracker(cfg)
    out = []
    for dets in frames:
        res = trk.update(np.array(dets, dtype=float).reshape(-1, 5))
        out.append({tid: tlwh for tid, tlwh, _ in res})
    return out, trk


def test_iou_and_hungarian():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    assert np.isclose(iou_matrix(a, a), np.eye(2)).all()
    m, ur, uc = linear_assignment(1 - iou_matrix(a, a[::-1]), 0.5)
    assert sorted(map(tuple, m.tolist())) == [(0, 1), (1, 0)] and not ur and not uc


def occluded_walk(gap):
    """One person walking right at 4 px/frame, invisible for `gap` frames."""
    frames = []
    for f in range(60):
        frames.append([] if 20 <= f < 20 + gap else [box(100 + 4 * f)])
    return frames


def test_reacquires_same_id_after_occlusion():
    out, trk = run(TrackerConfig.preset("byte"), occluded_walk(15))
    ids_before = set(out[19])
    ids_after = set(out[-1])
    assert ids_before == ids_after and len(ids_before) == 1
    assert trk.reid_events and trk.reid_events[0].gap == 15


def test_sort_spawns_new_id_after_occlusion():
    out, _ = run(TrackerConfig.preset("sort"), occluded_walk(15))
    assert set(out[19]) != set(out[-1])


def test_low_score_detections_keep_track_alive():
    frames = [[box(100 + 4 * f, s=0.9 if not 20 <= f < 30 else 0.3)] for f in range(40)]
    out, trk = run(TrackerConfig.preset("byte"), frames)
    assert all(len(o) == 1 for o in out[25:]) and set(out[19]) == set(out[-1])
    assert not trk.reid_events  # never even lost


def test_crossing_people_keep_ids():
    frames = []
    for f in range(80):
        frames.append([box(100 + 5 * f, cy=300), box(500 - 5 * f, cy=310)])
    out, _ = run(TrackerConfig.preset("byte"), frames)
    a0 = min(out[2], key=lambda t: out[2][t][0])  # left person at start
    a_end = max(out[-1], key=lambda t: out[-1][t][0])  # should now be on the right
    assert a0 == a_end


def test_interpolation_fills_gap():
    rows = np.array([[1, 7, 0, 0, 10, 10, 1], [5, 7, 40, 0, 10, 10, 1]], float)
    filled = interpolate_gaps(rows, 10)
    assert len(filled) == 5 and np.isclose(filled[2, 2], 20)


def test_evaluation_ignores_distractors_and_counts_switch():
    from mot_tracker.evaluation import evaluate_sequence, compute_metrics
    # GT: person 1 (class 1) + static person 2 (class 7, distractor) on frames 1-4
    gt = []
    for f in range(1, 5):
        gt.append([f, 1, 100, 100, 50, 120, 1, 1, 1.0])
        gt.append([f, 2, 400, 100, 50, 120, 0, 7, 1.0])
    gt = np.array(gt, float)
    res = []
    for f in range(1, 5):
        res.append([f, 10 if f < 3 else 11, 100, 100, 50, 120, 1, -1, -1, -1])   # ID switch at frame 3
        res.append([f, 20, 400, 100, 50, 120, 1, -1, -1, -1])                  # box on the distractor
    acc, _ = evaluate_sequence(gt, np.array(res, float))
    m = compute_metrics([acc], ["t"]).loc["t"]
    assert m["FP"] == 0 and m["FN"] == 0 and m["IDSW"] == 1


def test_public_detections_reader(tmp_path):
    from mot_tracker.detector import PublicDetections
    f = tmp_path / "det.txt"
    f.write_text("1,-1,10,20,30,40,0.9,-1,-1,-1\n2,-1,12,20,30,40,0.4,-1,-1,-1\n")
    pub = PublicDetections(f)
    assert np.allclose(pub.get(1), [[10, 20, 40, 60, 0.9]]) and len(pub.get(3)) == 0
