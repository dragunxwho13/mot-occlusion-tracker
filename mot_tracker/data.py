"""MOTChallenge sequence I/O (MOT17 / MOT20 layout)."""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MOTSequence:
    root: Path
    name: str
    frame_rate: int
    seq_length: int
    width: int
    height: int
    img_dir: Path
    img_ext: str

    @classmethod
    def load(cls, root: str | Path) -> "MOTSequence":
        root = Path(root)
        ini = configparser.ConfigParser()
        ini.read(root / "seqinfo.ini")
        s = ini["Sequence"]
        return cls(root=root, name=s.get("name", root.name), frame_rate=int(s.get("frameRate", 30)),
                   seq_length=int(s["seqLength"]), width=int(s["imWidth"]), height=int(s["imHeight"]),
                   img_dir=root / s.get("imDir", "img1"), img_ext=s.get("imExt", ".jpg"))

    def frame_path(self, i: int) -> Path:
        return self.img_dir / f"{i:06d}{self.img_ext}"

    def read(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.frame_path(i)))
        if img is None:
            raise FileNotFoundError(self.frame_path(i))
        return img

    @property
    def gt_file(self) -> Path:
        return self.root / "gt" / "gt.txt"

    @property
    def det_file(self) -> Path:
        return self.root / "det" / "det.txt"


def list_sequences(split_dir: str | Path, detector_suffix: str = "FRCNN", only: list[str] | None = None):
    """List sequences in e.g. MOT17/train.

    MOT17 ships every sequence three times (-DPM, -FRCNN, -SDP) with identical
    images and GT, differing only in public detections. We keep one copy so
    each video is counted once. MOT20 has no suffix.
    """
    split_dir = Path(split_dir)
    seqs = sorted(p for p in split_dir.iterdir() if (p / "seqinfo.ini").exists())
    has_suffix = any(p.name.endswith(("-DPM", "-FRCNN", "-SDP")) for p in seqs)
    if has_suffix:
        seqs = [p for p in seqs if p.name.endswith(f"-{detector_suffix}")]
    if only:
        seqs = [p for p in seqs if any(o in p.name for o in only)]
    return [MOTSequence.load(p) for p in seqs]


def load_gt(gt_file: str | Path) -> np.ndarray:
    """gt.txt columns: frame, id, x, y, w, h, consider, class, visibility."""
    return np.loadtxt(gt_file, delimiter=",", ndmin=2)


def write_results(path: str | Path, rows: list[tuple]):
    """rows: (frame, id, x, y, w, h, score) -> MOTChallenge result file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fr, tid, x, y, w, h, s in rows:
            f.write(f"{int(fr)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.3f},-1,-1,-1\n")


def load_results(path: str | Path) -> np.ndarray:
    if Path(path).stat().st_size == 0:
        return np.zeros((0, 10))
    return np.loadtxt(path, delimiter=",", ndmin=2)
