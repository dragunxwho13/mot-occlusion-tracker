#!/usr/bin/env python
"""Generate the 1-2 page write-up (REPORT.md) from the numbers in a results folder.

  python scripts/make_report.py --results results

Every number in the report is read from files produced by run_all.py, so the
write-up can never drift from the actual results.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

APPROACH = """\
## 1. Approach

**Pipeline.** `frame -> YOLO11 (COCO-pretrained, person class, conf >= 0.1) -> appearance embedding -> camera-motion estimate -> tracker -> MOT result file`.
No detector is trained; detections are computed once and cached, so every tracker variant below sees *identical* boxes and metric differences are due to the tracker alone.

**Tracker** (`mot_tracker/tracker.py`), built up from SORT:

| Component | What it does | Why |
|---|---|---|
| Kalman filter, state `(cx, cy, a, h, v...)` | constant-velocity motion model; noise scaled by box height | predicts where a person is while unseen |
| Hungarian assignment | optimal one-to-one det/track matching per frame (SciPy) | the SORT core |
| Lost-track buffer (30 frames @ 30 fps) | an unmatched track is kept as LOST and keeps being predicted instead of being deleted | lets a briefly occluded person get the *same* ID back |
| Two-stage association (BYTE) | high-score dets matched first; low-score dets (0.1-0.5) then matched to the remaining tracks by IoU | a half-occluded person gets a low detector score -- using it keeps the track alive through the occlusion |
| Appearance re-ID | part-based HSV colour histogram (head/torso/legs), EMA-smoothed per track. Visible tracks: cost = mean(IoU dist, appearance dist) for nearby boxes. LOST tracks: cost = 0.8 x appearance + 0.2 x normalised Mahalanobis, inside the gate | when a LOST track has drifted off the person, IoU is 0 -- appearance lets it be re-associated to its old ID; the motion term breaks ties between look-alikes |
| Mahalanobis gate (chi^2 95%, inflated x4 for LOST tracks) | appearance matches are only allowed where the motion model says the person could be | stops two similar-looking people on opposite sides of the frame from swapping |
| Camera-motion compensation | global similarity transform from sparse optical flow on background corners, applied to track states | MOT17-05/10/11/13 are filmed from a moving camera |
| Gap interpolation (offline only) | linear boxes for gaps <= 20 frames inside a track | recovers the frames during an occlusion once the ID is re-acquired |

**Evaluation.** MOTChallenge protocol re-implemented on `py-motmetrics`: IoU >= 0.5, only class-1 pedestrians with the consider flag count, tracker boxes on distractor classes (static person, reflection, person on vehicle) are ignored. MOT17 test GT is private, so all numbers are on the **MOT17 train** split; because the detector is COCO-pretrained and the tracker has no learned parameters, train is not a seen split for any part of the system.
"""

DECISIONS = """\
## 2. Key decisions

1. **Pretrained YOLO11 instead of the MOT17 public detections.** Modern COCO detectors are far stronger than DPM/FRCNN public boxes; detection quality bounds MOTA. `--public` reruns the full tracker on the official FRCNN detections for a tracker-only comparison.
2. **Keep low-confidence detections.** The detector runs at conf 0.1; the tracker decides what to do with weak boxes (only extend existing tracks, never start new ones). This is the single most effective occlusion trick.
3. **Appearance only where motion agrees, and never alone.** Pure appearance matching swaps IDs between people in similar clothing, and taking `min(IoU cost, appearance cost)` (BoT-SORT style) makes two look-alikes equally cheap so the Hungarian tie-break is arbitrary -- this produced a twin swap in testing. So appearance is *averaged* with IoU for visible tracks, mixed with a Mahalanobis term for lost tracks, and only allowed for pairs already plausible by motion (IoU distance < 0.5, or inside the inflated Kalman gate).
4. **Don't learn appearance from occluded crops.** A track's appearance template is not updated when its detection overlaps another detection (IoU > 0.3), otherwise the template absorbs the occluder's colours and re-ID then fails.
5. **Colour-histogram embedding instead of a deep re-ID net.** No extra weights or GPU, and for re-acquiring someone after 1-2 s clothing colour is the dominant cue. A ResNet-18 backend is included (`--appearance resnet`) as an alternative.
6. **Ablation on shared detections** so each row adds exactly one idea.
"""


def _pct(x):
    return f"{100 * float(x):.1f}"


def build_report(results_dir, out_path=None, title=None, link_prefix=""):
    """link_prefix: prepended to image/file links (use 'results/' when writing to the repo root)."""
    r = Path(results_dir)
    L = link_prefix
    summ = json.loads((r / "summary.json").read_text())
    abl = pd.read_csv(r / "ablation.csv", index_col=0)
    labels = list(abl.index)
    sort_row, full_row = abl.loc[labels[0]], abl.loc["+ appearance re-ID of lost tracks (full)"]
    best_row = abl.loc["+ gap interpolation (offline)"]
    fail = summ["failure"]["full"]
    fail_sort = summ["failure"]["sort"]
    data_name = Path(summ["data"]).parent.name or summ["data"]
    synthetic = "SYNTH" in summ["data"].upper()

    def abl_md():
        t = abl[["MOTA", "IDF1", "IDSW", "Frag", "FP", "FN", "MT", "ML"]].copy()
        for c in ["MOTA", "IDF1"]:
            t[c] = t[c].map(_pct)
        for c in ["IDSW", "Frag", "FP", "FN", "MT", "ML"]:
            t[c] = t[c].astype(int)
        t.index.name = "Tracker variant (same detections)"
        return t.to_markdown()

    idsw_red = 100 * (1 - full_row["IDSW"] / max(sort_row["IDSW"], 1))
    lines = []
    lines.append(f"# {title or 'Multi-Object Tracking Under Occlusion'} — write-up\n")
    scope = (f"Data: **{data_name}** `{summ['data']}`, sequences: {', '.join(summ['sequences'])}"
             + (f" (first {summ['max_frames']} frames each)" if summ.get("max_frames") else "")
             + f". Detector: `{summ['weights']}` @ {summ['imgsz']} px. Appearance: `{summ['appearance']}`.")
    lines.append(scope + "\n")
    if synthetic:
        lines.append("> **Note:** these numbers come from the scripted synthetic smoke-test sequence "
                     "(`tools/make_synthetic_sequence.py`), not from MOT17. Run `scripts/run_all.py --data "
                     "data/MOT17/train` to regenerate this report on the benchmark.\n")
    lines.append(APPROACH)
    lines.append(DECISIONS)
    lines.append("## 3. Results\n")
    lines.append("### 3.1 Ablation (all sequences combined)\n")
    lines.append(abl_md() + "\n")
    lines.append(
        f"Going from plain SORT to the full online tracker changes MOTA {_pct(sort_row['MOTA'])} -> "
        f"{_pct(full_row['MOTA'])}, IDF1 {_pct(sort_row['IDF1'])} -> {_pct(full_row['IDF1'])} and ID switches "
        f"{int(sort_row['IDSW'])} -> {int(full_row['IDSW'])} ({abs(idsw_red):.0f}% {'fewer' if idsw_red >= 0 else 'more'}). "
        f"With offline gap interpolation: "
        f"MOTA {_pct(best_row['MOTA'])}, IDF1 {_pct(best_row['IDF1'])}. "
        "MOTA is dominated by FN (missed people), which is a detector property; IDF1 and IDSW are what the "
        "tracker controls, and they are the metrics to read for the occlusion problem.\n")
    if (r / "per_sequence.md").exists():
        lines.append("### 3.2 Per sequence (full tracker + interpolation)\n")
        lines.append((r / "per_sequence.md").read_text() + "\n")

    lines.append("### 3.3 Does the tracker survive occlusion?\n")
    rec = {k: pd.DataFrame(v) for k, v in summ["recovery"].items()}
    if all(len(v) for v in rec.values()):
        def cell(d, k):
            e, kept = int(d["events"][k]), int(d["same ID kept"][k])
            return "–" if e == 0 else f"{kept}/{e} ({100 * kept / e:.0f}%)"
        t = pd.DataFrame({"gap (frames)": rec["full"]["gap (frames)"],
                          "SORT": [cell(rec["sort"], k) for k in range(len(rec["full"]))],
                          "SORT + buffer": [cell(rec["sort_buffer"], k) for k in range(len(rec["full"]))],
                          "full tracker": [cell(rec["full"], k) for k in range(len(rec["full"]))]})
        lines.append("Every time a ground-truth person stopped being tracked and was later tracked again, "
                      "did they get their *old* ID back? Cells: gaps where the same ID was kept / all gaps of that "
                      "length (`gap` = consecutive frames without a matched box).\n")
        lines.append(t.to_markdown(index=False) + "\n")
        lines.append(f"![recovery]({L}occlusion_recovery.png)\n")

    lines.append("## 4. Failure analysis: where do ID switches happen?\n")
    lines.append("Each ID switch of the full tracker is matched back to the last frame the person was tracked "
                 "correctly and tagged (see `mot_tracker/analysis.py`): **occlusion** if GT visibility dropped "
                 "below 0.5, the person overlapped another person (IoU > 0.3) or there was a tracking gap; "
                 "**long occlusion** if the gap exceeded the 30-frame buffer; **fast motion** if the centre moved "
                 "> 0.08 box-heights/frame; **similar appearance** if the new ID previously belonged to "
                 "a different person whose colour histogram is > 0.85 cosine-similar.\n")
    n = fail.get("num_switches", 0)
    if n:
        prim = fail["primary"]
        t = pd.DataFrame({"full tracker": prim, "SORT": fail_sort.get("primary", {})}).fillna(0).astype(int)
        t = t[(t.T != 0).any()]
        t.loc["total"] = t.sum()
        t.index.name = "primary cause"
        lines.append(t.to_markdown() + "\n")
        fl = fail["flags"]
        top = max(prim, key=prim.get)
        lines.append(
            f"Of {n} switches, {fl['occlusion']} ({100 * fl['occlusion'] / n:.0f}%) involve occlusion, "
            f"{fl['fast_motion']} ({100 * fl['fast_motion'] / n:.0f}%) fast motion and "
            f"{fl['similar_appearance']} ({100 * fl['similar_appearance'] / n:.0f}%) an identity exchange "
            f"between similar-looking people (flags overlap). {fl['transfer']} are *exchanges* between two "
            f"people and {fl['new_track']} are a new ID spawned for the same person. The most common primary "
            f"cause is **{top}**; the median switch follows a {fail['median_gap']:.0f}-frame gap with minimum "
            f"visibility {fail['median_min_vis']:.2f}.\n")
        lines.append(f"![causes]({L}full/analysis/id_switch_causes.png)\n")
        gal = sorted((r / "full" / "analysis").glob("switches_*.png"))
        if gal:
            lines.append("Examples (left: last correct frame, right: the switch) — "
                         + ", ".join(f"[{g.stem}]({L}full/analysis/{g.name})" for g in gal) + ".\n")
    else:
        lines.append("The full tracker produced no ID switches on this data.\n")

    interp = []
    if n:
        fl, prim = fail["flags"], fail["primary"]
        if fl["occlusion"] >= n / 2:
            interp.append(f"Occlusion is the dominant failure mode ({fl['occlusion']}/{n} switches).")
        if prim.get("long occlusion (> buffer)", 0):
            interp.append(f"{prim['long occlusion (> buffer)']} switches follow occlusions longer than the 30-frame "
                          "buffer: the track had already been deleted, so a new ID was unavoidable for an online "
                          "tracker with this buffer (a longer buffer trades these for more false re-identifications).")
        crossing = prim.get("occlusion", 0) + prim.get("occlusion + similar appearance", 0)
        if crossing:
            interp.append(f"{crossing} happen while people overlap each other: when two people cross, both Kalman "
                          "predictions sit on the same detections and the detector often returns one merged box.")
        interp.append(f"Fast motion is the primary cause of {prim.get('fast motion', 0)} switches"
                      + (" -- at 30 fps a walking person moves well under a tenth of their height per frame, so "
                         "the motion model rarely loses them; CMC handles the moving-camera case."
                         if prim.get('fast motion', 0) <= n / 5 else "."))
        sim_n = fl["similar_appearance"]
        interp.append(f"Similar-looking people are involved in {sim_n} switch(es)"
                      + (" -- the residual failure a colour histogram cannot resolve." if sim_n else "."))
    interp.append("The natural next step is a learned re-ID embedding (e.g. OSNet trained on Market-1501) "
                  "plugged into `appearance.py`, which would also allow a longer LOST buffer without more false "
                  "re-identifications.")
    lines.append("**Interpretation.** " + " ".join(interp) + "\n")

    lines.append("## 5. Visualisation\n")
    v = summ["viz_seq"]
    lines.append(f"`{L}viz/{v}_tracks.mp4` — frames {summ['viz_frames'][0]}-{summ['viz_frames'][1]} of {v}: boxes, IDs "
                 "and trajectory tails; dashed boxes are positions filled in while the person was occluded. "
                 f"`{L}viz_sort/{v}_tracks.mp4` is the SORT baseline on the same clip for comparison.\n")
    lines.append(f"![trajectories]({L}viz/{v}_trajectories.png)\n")
    lines.append(f"\n_Pipeline runtime: {summ['runtime_min']} min (excluding detection if it was cached)._\n")

    out_path = Path(out_path or r / "REPORT.md")
    out_path.write_text("\n".join(lines))
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    print(build_report(a.results, a.out))


if __name__ == "__main__":
    main()
