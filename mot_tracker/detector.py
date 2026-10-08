"""Per-frame person detection with a pretrained (COCO) detector.

No detector is trained here -- the brief requires a pretrained model. We use
Ultralytics YOLO (COCO weights, class 0 = person). Detections are returned
down to a low confidence (0.1) because the tracker's second association
stage deliberately consumes low-score boxes, which are typically partially
occluded people.

`PublicDetections` replays MOT17's provided det.txt (DPM / FRCNN / SDP) so
the tracker can also be evaluated on the benchmark's official detections,
separating tracker quality from detector quality.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class YOLODetector:
    def __init__(self, weights: str = "yolo11m.pt", conf: float = 0.1, iou: float = 0.7,
                 imgsz: int = 1280, device: str | None = None, min_box_area: float = 100.0):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.min_box_area = min_box_area

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Return Nx5 array [x1, y1, x2, y2, score] for people in a BGR frame."""
        res = self.model.predict(frame, classes=[0], conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 5))
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()[:, None]
        dets = np.hstack([xyxy, conf])
        wh = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        return dets[wh >= self.min_box_area]


class PublicDetections:
    """MOT17 det/det.txt: frame, -1, x, y, w, h, score, ..."""

    def __init__(self, det_file: str | Path, min_score: float | None = None):
        raw = np.loadtxt(det_file, delimiter=",", ndmin=2)
        self.by_frame: dict[int, np.ndarray] = {}
        for f in np.unique(raw[:, 0]).astype(int):
            r = raw[raw[:, 0] == f]
            d = np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5], r[:, 6]], axis=1)
            if min_score is not None:
                d = d[d[:, 4] >= min_score]
            self.by_frame[int(f)] = d
        # DPM scores are unbounded; squash all detectors to [0, 1] for the thresholds.
        all_s = np.concatenate([d[:, 4] for d in self.by_frame.values()]) if self.by_frame else np.zeros(1)
        self.normalise = all_s.max() > 1.0 or all_s.min() < 0.0
        if self.normalise:
            lo, hi = np.percentile(all_s, 1), np.percentile(all_s, 99)
            for f, d in self.by_frame.items():
                d[:, 4] = np.clip((d[:, 4] - lo) / (hi - lo + 1e-9), 0, 1)

    def get(self, frame_idx: int) -> np.ndarray:
        return self.by_frame.get(frame_idx, np.zeros((0, 5))).copy()
