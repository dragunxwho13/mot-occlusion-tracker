#!/usr/bin/env python
"""One command to reproduce everything: detections -> ablation of tracker
variants -> metrics -> failure analysis -> visualisations -> REPORT.md.

  python scripts/run_all.py --data data/MOT17/train
  python scripts/run_all.py --data data/MOT17/train --quick     # 3 seqs x 300 frames, ~10 min on CPU
  python scripts/run_all.py --data data/SYNTH/train --out results_synth --weights yolo11n.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mot_tracker.data import list_sequences  # noqa: E402
from mot_tracker.pipeline import run_experiment, get_detections  # noqa: E402
from mot_tracker.tracker import TrackerConfig  # noqa: E402
from mot_tracker.evaluation import format_table  # noqa: E402
from evaluate import evaluate_dir  # noqa: E402
from analyze_failures import analyse_experiment, plot_causes, plot_recovery  # noqa: E402
from visualize import visualize  # noqa: E402
from make_report import build_report  # noqa: E402

VARIANTS = [
    # (row label, preset, result sub-folder)
    ("SORT (IoU + Hungarian, no memory)", "sort", "data"),
    ("+ 30-frame lost-track buffer", "sort_buffer", "data"),
    ("+ low-score det. association (BYTE)", "byte", "data"),
    ("+ camera-motion compensation", "byte_cmc", "data"),
    ("+ appearance re-ID of lost tracks (full)", "full", "data"),
    ("+ gap interpolation (offline)", "full", "data_interp"),
]
KEY_COLS = ["MOTA", "IDF1", "IDSW", "Frag", "FP", "FN", "MT", "ML", "Rcll", "Prcn"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="split dir, e.g. data/MOT17/train")
    p.add_argument("--out", default="results")
    p.add_argument("--seqs", nargs="*")
    p.add_argument("--det-suffix", default="FRCNN")
    p.add_argument("--weights", default="yolo11m.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None)
    p.add_argument("--appearance", default="hist", choices=["hist", "resnet"])
    p.add_argument("--max-frames", type=int, default=None, help="truncate every sequence (faster runs)")
    p.add_argument("--quick", action="store_true", help="MOT17-02/04/09, first 300 frames")
    p.add_argument("--public", action="store_true", help="also run the full tracker on MOT17 public FRCNN dets")
    p.add_argument("--viz-seq", default=None, help="sequence for the trajectory video (default: first)")
    p.add_argument("--viz-start", type=int, default=1)
    p.add_argument("--viz-end", type=int, default=300)
    a = p.parse_args()
    if a.quick:
        a.seqs = a.seqs or ["MOT17-02", "MOT17-04", "MOT17-09"]
        a.max_frames = a.max_frames or 300

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seqs = list_sequences(a.data, a.det_suffix, a.seqs)
    if not seqs:
        sys.exit(f"no sequences in {a.data}")
    if a.max_frames:
        for s in seqs:
            s.seq_length = min(s.seq_length, a.max_frames)
    print(f"{len(seqs)} sequences: {[s.name for s in seqs]}")
    t_start = time.time()

    # 1) detections once (cached) -------------------------------------------------
    det_kwargs = dict(detector="yolo", weights=a.weights, imgsz=a.imgsz, cache_dir=str(out / "cache_dets"),
                      device=a.device)
    model = None
    for s in seqs:
        from mot_tracker.pipeline import detection_cache_path
        if not detection_cache_path(out / "cache_dets", s, "yolo", a.weights, a.imgsz).exists():
            if model is None:
                from mot_tracker.detector import YOLODetector
                model = YOLODetector(a.weights, imgsz=a.imgsz, device=a.device)
            get_detections(s, model=model, **det_kwargs)

    # 2) tracker variants ------------------------------------------------------------
    run_info = {}
    for preset in dict.fromkeys(v[1] for v in VARIANTS):
        cfg = TrackerConfig.preset(preset)
        print(f"\n== {preset} ==")
        run_info[preset] = run_experiment(seqs, out / preset, cfg, a.appearance, det_kwargs,
                                          device=a.device or "cpu")
    if a.public and all(s.det_file.exists() for s in seqs):
        print("\n== full (public FRCNN detections) ==")
        pub_kwargs = dict(detector="public")
        run_info["full_public"] = run_experiment(seqs, out / "full_public", TrackerConfig.preset("full"),
                                                 a.appearance, pub_kwargs, device=a.device or "cpu")

    # 3) evaluation -----------------------------------------------------------------
    names = [s.name for s in seqs]
    rows = {}
    per_seq = {}
    labels = list(VARIANTS) + ([("full tracker, MOT17 public FRCNN dets", "full_public", "data")]
                               if "full_public" in run_info else [])
    for label, preset, sub in labels:
        df = evaluate_dir(a.data, out / preset / sub, a.det_suffix, names, verbose=False,
                          max_frames=a.max_frames)
        rows[label] = df.loc["OVERALL"]
        per_seq[f"{preset}/{sub}"] = df
    abl = pd.DataFrame(rows).T[KEY_COLS + ["MOTP", "GT_IDs"]]
    abl.to_csv(out / "ablation.csv")
    (out / "ablation.md").write_text(format_table(abl[KEY_COLS]) + "\n")
    main_key = "full/data_interp"
    (out / "per_sequence.md").write_text(format_table(per_seq[main_key][KEY_COLS + ["MOTP"]]) + "\n")
    (out / "per_sequence_online.md").write_text(format_table(per_seq["full/data"][KEY_COLS + ["MOTP"]]) + "\n")
    print("\nABLATION (OVERALL)\n" + format_table(abl[KEY_COLS]))

    # 4) failure analysis -------------------------------------------------------------
    print("\n== failure analysis ==")
    _, rec_full, summ_full = analyse_experiment(a.data, out / "full", "data", a.det_suffix, names, a.max_frames)
    _, rec_sb, summ_sb = analyse_experiment(a.data, out / "sort_buffer", "data", a.det_suffix, names,
                                            a.max_frames, gallery=False)
    _, rec_sort, summ_sort = analyse_experiment(a.data, out / "sort", "data", a.det_suffix, names,
                                                a.max_frames, gallery=False)
    plot_causes(summ_full, out / "full" / "analysis" / "id_switch_causes.png",
                "Full tracker: primary cause of each ID switch")
    plot_causes(summ_sort, out / "sort" / "analysis" / "id_switch_causes.png", "SORT: primary cause of each ID switch")
    plot_recovery({"SORT": rec_sort, "SORT + buffer": rec_sb, "full tracker": rec_full},
                  out / "occlusion_recovery.png")

    # 5) visualisation ------------------------------------------------------------------
    viz_seq = next((s for s in seqs if a.viz_seq and a.viz_seq in s.name), seqs[0])
    end = min(a.viz_end, viz_seq.seq_length)
    visualize(viz_seq.root, out / "full" / "data_interp" / f"{viz_seq.name}.txt", out / "viz", a.viz_start, end,
              title="full tracker")
    visualize(viz_seq.root, out / "sort" / "data" / f"{viz_seq.name}.txt", out / "viz_sort", a.viz_start, end,
              gif=False, title="SORT baseline")

    summary = dict(
        data=str(a.data), sequences=names, max_frames=a.max_frames, weights=a.weights, imgsz=a.imgsz,
        appearance=a.appearance, runtime_min=round((time.time() - t_start) / 60, 1),
        tracker_fps={k: v["sequences"] for k, v in run_info.items()},
        failure={"full": summ_full, "sort_buffer": summ_sb, "sort": summ_sort},
        recovery={"full": rec_full.to_dict("records"), "sort_buffer": rec_sb.to_dict("records"),
                  "sort": rec_sort.to_dict("records")},
        viz_seq=viz_seq.name, viz_frames=[a.viz_start, end],
    )
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    report = build_report(out)
    try:  # also write the submission copy at the repo root, with links into the results folder
        rel = out.resolve().relative_to(ROOT)
        build_report(out, ROOT / "REPORT.md", link_prefix=f"{rel.as_posix()}/")
        gif = out / "viz" / f"{viz_seq.name}_tracks.gif"
        if gif.exists():
            (ROOT / "docs").mkdir(exist_ok=True)
            (ROOT / "docs" / "demo.gif").write_bytes(gif.read_bytes())
    except ValueError:
        pass
    print(f"\nDone in {summary['runtime_min']} min. Report: {report} (copy: REPORT.md)")


if __name__ == "__main__":
    main()
