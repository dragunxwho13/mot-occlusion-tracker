#!/usr/bin/env python
"""Run the tracker on MOTChallenge sequences and write result files.

Examples
--------
  # Full tracker, YOLO detections, all MOT17 train sequences
  python scripts/track.py --data data/MOT17/train --preset full --out results/full

  # Classic SORT baseline on the same (cached) detections
  python scripts/track.py --data data/MOT17/train --preset sort --out results/sort

  # Use MOT17's official public FRCNN detections instead of YOLO
  python scripts/track.py --data data/MOT17/train --detector public --preset full --out results/full_public
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot_tracker.data import list_sequences  # noqa: E402
from mot_tracker.pipeline import run_experiment  # noqa: E402
from mot_tracker.tracker import TrackerConfig  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="split dir, e.g. data/MOT17/train")
    p.add_argument("--seqs", nargs="*", help="substring filter, e.g. MOT17-04 MOT17-09")
    p.add_argument("--det-suffix", default="FRCNN", help="which MOT17 copy to use (DPM/FRCNN/SDP)")
    p.add_argument("--out", required=True)
    p.add_argument("--preset", default="full", choices=["sort", "sort_buffer", "byte", "byte_cmc", "full"])
    p.add_argument("--appearance", default="hist", choices=["hist", "resnet", "none"])
    p.add_argument("--detector", default="yolo", choices=["yolo", "public"])
    p.add_argument("--weights", default="yolo11m.pt", help="any Ultralytics COCO checkpoint")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None, help="cpu / 0 / mps (auto if unset)")
    p.add_argument("--cache", default="cache/dets")
    p.add_argument("--max-age", type=int, default=None, help="override LOST buffer (frames @30fps)")
    p.add_argument("--appearance-thresh", type=float, default=None)
    p.add_argument("--interp-gap", type=int, default=20)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    seqs = list_sequences(a.data, a.det_suffix, a.seqs)
    if not seqs:
        sys.exit(f"no sequences found in {a.data}")
    cfg = TrackerConfig.preset(a.preset, max_age=a.max_age, appearance_thresh=a.appearance_thresh)
    if a.appearance == "none":
        cfg.use_appearance = False
    det_kwargs = dict(detector=a.detector, weights=a.weights, imgsz=a.imgsz, cache_dir=a.cache, device=a.device)
    print(f"[{a.preset}] {len(seqs)} sequences -> {a.out}")
    run_experiment(seqs, a.out, cfg, a.appearance, det_kwargs, a.interp_gap,
                   device=a.device if a.device not in (None, "") else "cpu")


if __name__ == "__main__":
    main()
