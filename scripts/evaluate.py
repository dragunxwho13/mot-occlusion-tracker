#!/usr/bin/env python
"""Evaluate MOTChallenge result files against ground truth (MOTA, IDF1, IDSW...).

  python scripts/evaluate.py --gt data/MOT17/train --results results/full/data

Writes metrics.md / metrics.csv next to the result folder plus events_<seq>.csv
(every per-frame match / miss / FP / SWITCH event) used by the failure analysis.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot_tracker.data import list_sequences, load_gt, load_results  # noqa: E402
from mot_tracker.evaluation import evaluate_sequence, compute_metrics, format_table  # noqa: E402


def evaluate_dir(gt_root, results_dir, det_suffix="FRCNN", seq_filter=None, save=True, verbose=True,
                 max_frames=None):
    results_dir = Path(results_dir)
    seqs = [s for s in list_sequences(gt_root, det_suffix, seq_filter) if (results_dir / f"{s.name}.txt").exists()]
    if not seqs:
        raise SystemExit(f"no result files in {results_dir} matching sequences in {gt_root}")
    accs, names = [], []
    for s in seqs:
        gt = load_gt(s.gt_file)
        if max_frames:
            gt = gt[gt[:, 0] <= max_frames]
        acc, _ = evaluate_sequence(gt, load_results(results_dir / f"{s.name}.txt"))
        accs.append(acc)
        names.append(s.name)
        if save:
            ev = acc.mot_events.reset_index()
            ev.to_csv(results_dir.parent / f"events_{results_dir.name}_{s.name}.csv", index=False)
    df = compute_metrics(accs, names)
    if save:
        df.to_csv(results_dir.parent / f"metrics_{results_dir.name}.csv")
        (results_dir.parent / f"metrics_{results_dir.name}.md").write_text(format_table(df) + "\n")
    if verbose:
        print(format_table(df))
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True, help="split dir with GT, e.g. data/MOT17/train")
    p.add_argument("--results", required=True, help="folder of <seq>.txt result files")
    p.add_argument("--det-suffix", default="FRCNN")
    p.add_argument("--seqs", nargs="*")
    p.add_argument("--max-frames", type=int, default=None, help="only score the first N frames")
    a = p.parse_args()
    evaluate_dir(a.gt, a.results, a.det_suffix, a.seqs, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
