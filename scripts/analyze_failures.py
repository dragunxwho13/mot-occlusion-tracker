#!/usr/bin/env python
"""Classify every ID switch (occlusion / fast motion / similar appearance / ...)
and measure how often occlusion gaps are bridged with the same ID.

  python scripts/analyze_failures.py --gt data/MOT17/train --exp results/full
  python scripts/analyze_failures.py --gt data/MOT17/train --exp results/full --compare results/sort_buffer

Needs events_*.csv from scripts/evaluate.py (run automatically if missing).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot_tracker.analysis import analyse_switches, occlusion_recovery, cases_to_frame  # noqa: E402
from mot_tracker.data import list_sequences, load_gt, load_results  # noqa: E402
from mot_tracker.visualize import switch_gallery  # noqa: E402

CAUSE_ORDER = ["occlusion", "occlusion + similar appearance", "long occlusion (> buffer)",
               "similar appearance", "fast motion", "other (detector jitter / missed det)"]


def analyse_experiment(gt_root, exp_dir, variant="data", det_suffix="FRCNN", seq_filter=None, max_frames=None,
                       out_dir=None, gallery=True, max_age=30):
    exp_dir = Path(exp_dir)
    out_dir = Path(out_dir or exp_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = list_sequences(gt_root, det_suffix, seq_filter)
    all_cases, rec = [], []
    for s in seqs:
        ev_file = exp_dir / f"events_{variant}_{s.name}.csv"
        if not ev_file.exists():
            from evaluate import evaluate_dir  # noqa: E402
            evaluate_dir(gt_root, exp_dir / variant, det_suffix, seq_filter, verbose=False, max_frames=max_frames)
        ev = pd.read_csv(ev_file)
        gt = load_gt(s.gt_file)
        if max_frames:
            gt = gt[gt[:, 0] <= max_frames]
        cases = analyse_switches(s.name, gt, ev, read_frame=s.read, max_age=max_age)
        all_cases += cases
        r = occlusion_recovery(ev)
        r.insert(0, "seq", s.name)
        rec.append(r)
        if gallery and cases:
            rows = load_results(exp_dir / variant / f"{s.name}.txt")
            # most informative first: identity exchanges between two people, then by gap
            ordered = sorted(cases, key=lambda c: (not c.transfer, -c.crowd_iou))
            switch_gallery(s, ordered, rows, out_dir / f"switches_{s.name}.png", n=4)
    df = cases_to_frame(all_cases)
    df.to_csv(out_dir / "id_switch_cases.csv", index=False)
    rec_df = pd.concat(rec) if rec else pd.DataFrame()
    rec_all = (rec_df.groupby("gap (frames)", observed=False)[["events", "same ID kept"]].sum().reset_index()
               if len(rec_df) else rec_df)
    if len(rec_all):
        rec_all["recovery %"] = (100 * rec_all["same ID kept"] / rec_all["events"].clip(lower=1)).round(1)
    rec_all.to_csv(out_dir / "occlusion_recovery.csv", index=False)

    summary = {"num_switches": int(len(df))}
    if len(df):
        prim = df["primary"].value_counts().reindex(CAUSE_ORDER).fillna(0).astype(int)
        summary["primary"] = prim.to_dict()
        summary["flags"] = {k: int(df[k].sum()) for k in ["occlusion", "long_occlusion", "fast_motion",
                                                         "similar_appearance", "transfer", "new_track"]}
        summary["per_seq"] = df.groupby("seq")["primary"].value_counts().unstack(fill_value=0).to_dict("index")
        summary["median_gap"] = float(df["gap"].median())
        summary["median_min_vis"] = float(df["min_vis"].median())
    with open(out_dir / "failure_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=int)
    return df, rec_all, summary


def plot_causes(summary, out_png, title="ID-switch causes"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prim = summary.get("primary", {})
    labels = [k for k in CAUSE_ORDER if prim.get(k, 0) > 0]
    vals = [prim[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(labels[::-1], vals[::-1], color="#4C78A8")
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2, f"{v} ({100 * v / sum(vals):.0f}%)",
                va="center", fontsize=9)
    ax.set_xlabel("number of ID switches (primary cause)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_recovery(recs: dict, out_png):
    """recs: {label: recovery dataframe}"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = list(recs)
    w = 0.8 / max(len(names), 1)
    cols = ["#BAB0AC", "#F58518", "#4C78A8", "#54A24B", "#E45756"]
    for k, n in enumerate(names):
        r = recs[n]
        if not len(r):
            continue
        x = np.arange(len(r))
        ax.bar(x + k * w, r["recovery %"], w, label=n, color=cols[k % len(cols)])
        ax.set_xticks(x + w * (len(names) - 1) / 2, [str(g) for g in r["gap (frames)"]])
    ax.set_ylim(0, 105)
    ax.set_xlabel("length of the gap the person was not tracked (frames)")
    ax.set_ylabel("% re-associated to same ID")
    ax.set_title("Occlusion recovery: same ID kept after a tracking gap")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--exp", required=True, help="experiment folder, e.g. results/full")
    p.add_argument("--variant", default="data", choices=["data", "data_interp"])
    p.add_argument("--compare", nargs="*", default=[], help="other experiment folders for the recovery plot")
    p.add_argument("--det-suffix", default="FRCNN")
    p.add_argument("--seqs", nargs="*")
    p.add_argument("--max-frames", type=int)
    a = p.parse_args()
    df, rec, summ = analyse_experiment(a.gt, a.exp, a.variant, a.det_suffix, a.seqs, a.max_frames)
    out = Path(a.exp) / "analysis"
    plot_causes(summ, out / "id_switch_causes.png")
    recs = {}
    for c in a.compare:
        _, r, _ = analyse_experiment(a.gt, c, a.variant, a.det_suffix, a.seqs, a.max_frames, gallery=False)
        recs[Path(c).name] = r
    recs[Path(a.exp).name] = rec
    plot_recovery(recs, out / "occlusion_recovery.png")
    print(json.dumps(summ, indent=2, default=int))
    print(rec.to_markdown(index=False) if len(rec) else "no gaps")


if __name__ == "__main__":
    main()
