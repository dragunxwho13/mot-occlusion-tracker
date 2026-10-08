#!/usr/bin/env python
"""Render tracked trajectories over a sample clip.

  python scripts/visualize.py --seq data/MOT17/train/MOT17-04-FRCNN \
      --results results/full/data_interp/MOT17-04-FRCNN.txt --start 1 --end 300 --out results/viz

Produces <seq>_tracks.mp4 (boxes + IDs + trajectory tails; dashed boxes =
positions inferred while the person was occluded), <seq>_tracks.gif (small,
for the README) and <seq>_trajectories.png (all paths on one frame).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot_tracker.data import MOTSequence, load_results  # noqa: E402
from mot_tracker.visualize import render_video, plot_trajectories  # noqa: E402


def visualize(seq_dir, results_file, out_dir, start=1, end=300, gif=True, title=""):
    seq = MOTSequence.load(seq_dir)
    rows = load_results(results_file)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mp4 = render_video(seq, rows, out / f"{seq.name}_tracks.mp4", start, end,
                       gif_path=(out / f"{seq.name}_tracks.gif") if gif else None, title=title)
    png = plot_trajectories(seq, rows, out / f"{seq.name}_trajectories.png", start, end)
    print(f"wrote {mp4} and {png}")
    return mp4, png


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out", default="results/viz")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=300)
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args()
    visualize(a.seq, a.results, a.out, a.start, a.end, not a.no_gif)


if __name__ == "__main__":
    main()
