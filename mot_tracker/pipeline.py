"""Glue: detections (cached) -> appearance -> CMC -> tracker -> MOT rows."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .appearance import build_embedder
from .cmc import CameraMotionCompensator
from .data import MOTSequence, write_results
from .detector import YOLODetector, PublicDetections
from .postprocess import interpolate_gaps
from .tracker import OcclusionAwareTracker, TrackerConfig


def detection_cache_path(cache_dir: Path, seq: MOTSequence, detector: str, weights: str, imgsz: int) -> Path:
    tag = f"public" if detector == "public" else f"{Path(weights).stem}_{imgsz}"
    return Path(cache_dir) / tag / f"{seq.name}.npy"


def get_detections(seq: MOTSequence, detector: str = "yolo", weights: str = "yolo11m.pt", imgsz: int = 1280,
                   cache_dir: str | Path = "cache/dets", device: str | None = None, model=None) -> dict[int, np.ndarray]:
    """Return {frame: Nx5} detections, running the detector once and caching to disk.

    Every tracker variant in the ablation consumes the *same* detections, so
    differences in metrics are attributable to the tracker alone.
    """
    if detector == "public":
        pub = PublicDetections(seq.det_file)
        return {i: pub.get(i) for i in range(1, seq.seq_length + 1)}

    path = detection_cache_path(Path(cache_dir), seq, detector, weights, imgsz)
    if path.exists():
        arr = np.load(path)
    else:
        model = model or YOLODetector(weights=weights, imgsz=imgsz, device=device)
        rows = []
        t0 = time.time()
        for i in tqdm(range(1, seq.seq_length + 1), desc=f"detect {seq.name}", leave=False):
            d = model(seq.read(i))
            rows.append(np.hstack([np.full((len(d), 1), i), d]))
        arr = np.concatenate(rows) if rows else np.zeros((0, 6))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        print(f"  detector: {seq.seq_length / (time.time() - t0):.1f} FPS on {seq.name}")
    return {i: arr[arr[:, 0] == i, 1:6] for i in range(1, seq.seq_length + 1)}


def run_sequence(seq: MOTSequence, dets: dict[int, np.ndarray], cfg: TrackerConfig, appearance: str = "hist",
                 device: str = "cpu"):
    tracker = OcclusionAwareTracker(cfg, frame_rate=seq.frame_rate)
    embedder = build_embedder(appearance, device) if cfg.use_appearance else None
    cmc = CameraMotionCompensator() if cfg.use_cmc else None
    need_img = embedder is not None or cmc is not None
    rows = []
    t0 = time.time()
    for i in tqdm(range(1, seq.seq_length + 1), desc=f"track  {seq.name}", leave=False):
        d = dets.get(i, np.zeros((0, 5)))
        img = seq.read(i) if need_img else None
        H = cmc.estimate(img, d[d[:, 4] >= cfg.low_thresh, :4]) if cmc else None
        feats = None
        if embedder is not None:
            keep = d[:, 4] >= cfg.low_thresh
            feats = np.zeros((len(d), embedder.dim), dtype=np.float32)
            feats[keep] = embedder(img, d[keep, :4])
        for tid, tlwh, score in tracker.update(d, feats, H):
            rows.append((i, tid, *tlwh, score))
    fps = seq.seq_length / max(time.time() - t0, 1e-9)
    return rows, tracker.reid_events, fps


def run_experiment(seqs: list[MOTSequence], out_dir: str | Path, cfg: TrackerConfig, appearance: str,
                   det_kwargs: dict, interp_gap: int = 20, device: str = "cpu"):
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "data_interp").mkdir(parents=True, exist_ok=True)
    summary = {"config": cfg.to_dict(), "appearance": appearance if cfg.use_appearance else "none",
               "detector": det_kwargs, "sequences": {}}
    for seq in seqs:
        dets = get_detections(seq, **det_kwargs)
        rows, reid, fps = run_sequence(seq, dets, cfg, appearance, device)
        write_results(out_dir / "data" / f"{seq.name}.txt", rows)
        arr = np.array(rows, dtype=float).reshape(-1, 7)
        interp = interpolate_gaps(arr, interp_gap)
        write_results(out_dir / "data_interp" / f"{seq.name}.txt", [tuple(r[:7]) for r in interp])
        summary["sequences"][seq.name] = {
            "tracker_fps": round(fps, 1),
            "num_ids": int(len(np.unique(arr[:, 1]))) if len(arr) else 0,
            "reid_events": len(reid),
            "reid_by_appearance": sum(e.via == "appearance" for e in reid),
            "reid_gaps": [e.gap for e in reid],
        }
        with open(out_dir / f"reid_events_{seq.name}.json", "w") as f:
            json.dump([asdict(e) for e in reid], f)
        print(f"  {seq.name}: {summary['sequences'][seq.name]['num_ids']} IDs, "
              f"{len(reid)} re-associations after loss, tracker {fps:.0f} FPS")
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
