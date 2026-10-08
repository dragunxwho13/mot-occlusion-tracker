#!/usr/bin/env python
"""Build a small *synthetic* MOT-format sequence for smoke-testing the pipeline.

This is NOT a replacement for MOT17 -- it is a 20 s scripted "store aisle"
clip so the whole pipeline (detector -> tracker -> evaluation -> failure
analysis -> visualisation) can be exercised in a couple of minutes without
downloading the 5.5 GB benchmark. Real people are cut out of the two sample
photos that ship with Ultralytics (instance segmentation) and walked along
scripted paths that create each failure mode on purpose:

  * two people crossing at the same depth           -> person/person occlusion
  * a person walking behind a pillar                 -> full occlusion (~1 s)
  * a person behind a shelf for ~2.5 s               -> long occlusion
  * two *identical-looking* people crossing          -> similar appearance
  * a person running                                 -> fast motion
  * a small group walking together                   -> crowding

Output layout matches MOTChallenge (seqinfo.ini, img1/, gt/gt.txt with
visibility), so every script in this repo runs on it unchanged.

  python tools/make_synthetic_sequence.py --out data/SYNTH/train/SYNTH-01
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, N_FRAMES = 1280, 720, 30, 600


def extract_sprites(seg_weights: str):
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(seg_weights)
    assets = Path(ultralytics.__file__).parent / "assets"
    sprites = []
    for name in ["bus.jpg", "zidane.jpg"]:
        img = cv2.imread(str(assets / name))
        r = model.predict(img, classes=[0], conf=0.5, retina_masks=True, verbose=False)[0]
        if r.masks is None:
            continue
        for m, box in zip(r.masks.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            # skip people cut by the image border (they look truncated)
            if x1 <= 2 or x2 >= img.shape[1] - 2 or y2 >= img.shape[0] - 2:
                continue
            mask = (m[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            rgba = np.dstack([img[y1:y2, x1:x2], mask])
            sprites.append(rgba)
    return sprites


def recolor(sprite, hue_shift=0, sat=1.0):
    """Change clothing colour so one cut-out can play several different people."""
    out = sprite.copy()
    hsv = cv2.cvtColor(sprite[..., :3], cv2.COLOR_BGR2HSV).astype(np.int32)
    skin = (hsv[..., 0] <= 20) & (hsv[..., 1] > 40) & (hsv[..., 1] < 180) & (hsv[..., 2] > 80)
    hsv[..., 0] = np.where(skin, hsv[..., 0], (hsv[..., 0] + hue_shift) % 180)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    out[..., :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def make_background(rng):
    bg = np.full((H, W, 3), (196, 200, 204), np.uint8)
    for y in range(0, H, 48):  # floor tiles with mild perspective
        cv2.line(bg, (0, y), (W, y), (178, 182, 186), 1)
    for x in range(-W, 2 * W, 80):
        cv2.line(bg, (x, H), (W // 2 + (x - W // 2) // 3, 0), (178, 182, 186), 1)
    # back shelves (static background)
    for x in range(30, W, 250):
        cv2.rectangle(bg, (x, 40), (x + 190, 150), (90, 110, 140), -1)
        for k in range(4):
            cv2.rectangle(bg, (x + 8 + 45 * k, 55), (x + 40 + 45 * k, 140),
                          tuple(int(c) for c in rng.integers(60, 230, 3)), -1)
    return cv2.GaussianBlur(bg, (3, 3), 0)


# Floor-standing occluders: (x1, y1, x2, y2, colour). y2 is the occluder's front
# edge on the floor: people whose feet are above it (further away) are hidden
# behind it, people whose feet are below it walk in front of it.
OCCLUDERS = [
    (600, 120, 680, 470, (70, 75, 85)),     # pillar in the middle
    (60, 230, 420, 480, (60, 95, 150)),     # shelf unit, left
]


def walkers():
    """Scripted people: sprite index, hue shift, start frame, waypoints (foot x, foot y), speed px/frame."""
    return [
        # 1-2: cross each other at the same depth (person/person occlusion)
        dict(s=0, hue=0, t0=0, path=[(-60, 520), (1340, 540)], v=3.2),
        dict(s=1, hue=60, t0=40, path=[(1340, 545), (-60, 525)], v=3.0),
        # 3: walks behind the pillar (~1 s full occlusion)
        dict(s=2, hue=100, t0=60, path=[(950, 380), (300, 395)], v=2.4),
        # 4: walks behind the left display for a long time (~2.5 s)
        dict(s=3, hue=20, t0=90, path=[(500, 430), (30, 440), (30, 650)], v=1.6),
        # 5-6: identical-looking twins crossing (similar appearance)
        dict(s=1, hue=0, t0=230, path=[(-60, 650), (1340, 640)], v=3.4),
        dict(s=1, hue=0, t0=250, path=[(1340, 655), (-60, 650)], v=3.6),
        # 7: runner (fast motion) crossing the scene behind others
        dict(s=4, hue=140, t0=330, path=[(1340, 455), (-80, 455)], v=11.0),
        # 8-10: small group walking together
        dict(s=0, hue=120, t0=380, path=[(-60, 600), (1340, 560)], v=2.6),
        dict(s=2, hue=30, t0=385, path=[(-110, 615), (1300, 575)], v=2.6),
        dict(s=3, hue=80, t0=392, path=[(-150, 590), (1260, 550)], v=2.6),
    ]


def positions(wk, n_frames):
    """Foot position per frame (None when not yet started / finished)."""
    pts = np.array(wk["path"], float)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg_len.sum()
    out = {}
    for f in range(n_frames):
        d = (f - wk["t0"]) * wk["v"]
        if d < 0 or d > total:
            continue
        k = int(np.searchsorted(np.cumsum(seg_len), d, side="right"))
        k = min(k, len(seg_len) - 1)
        d0 = d - (np.cumsum(seg_len)[k - 1] if k > 0 else 0)
        p = pts[k] + (pts[k + 1] - pts[k]) * d0 / seg_len[k]
        # small gait bob
        out[f] = (p[0], p[1] + 2.0 * np.sin(f * 0.6 + wk["t0"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/SYNTH/train/SYNTH-01")
    ap.add_argument("--seg-weights", default="yolo11m-seg.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "img1").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(exist_ok=True)
    sprites = extract_sprites(a.seg_weights)
    if len(sprites) < 2:
        raise SystemExit("could not extract person sprites")
    print(f"extracted {len(sprites)} person cut-outs")
    bg = make_background(rng)
    wks = walkers()
    paths = [positions(w, N_FRAMES) for w in wks]
    looks = [recolor(sprites[w["s"] % len(sprites)], w["hue"]) for w in wks]

    gt_rows = []
    for f in range(N_FRAMES):
        frame = bg.copy()
        owner = np.full((H, W), -1, np.int32)
        placed = []
        # far (small foot y) first so nearer people occlude farther ones
        order = sorted([i for i in range(len(wks)) if f in paths[i]], key=lambda i: paths[i][f][1])
        # interleave occluders with people by depth (foot / front-edge y)
        items = [(paths[i][f][1], "p", i) for i in order] + [(o[3], "o", k) for k, o in enumerate(OCCLUDERS)]
        for _, kind, i in sorted(items):
            if kind == "o":
                x1, y1, x2, y2, col = OCCLUDERS[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), 2)
                owner[y1:y2, x1:x2] = -1
                continue
            fx, fy = paths[i][f]
            h = int(140 + (fy - 350) * 0.35)  # perspective: lower in frame -> taller
            sp = looks[i]
            w = int(sp.shape[1] * h / sp.shape[0])
            sp = cv2.resize(sp, (w, h), interpolation=cv2.INTER_AREA)
            x1, y1 = int(fx - w / 2), int(fy - h)
            X1, Y1, X2, Y2 = max(0, x1), max(0, y1), min(W, x1 + w), min(H, y1 + h)
            if X2 <= X1 or Y2 <= Y1:
                continue
            crop = sp[Y1 - y1:Y2 - y1, X1 - x1:X2 - x1]
            alpha = crop[..., 3:4].astype(np.float32) / 255
            roi = frame[Y1:Y2, X1:X2].astype(np.float32)
            frame[Y1:Y2, X1:X2] = (alpha * crop[..., :3] + (1 - alpha) * roi).astype(np.uint8)
            owner[Y1:Y2, X1:X2][crop[..., 3] > 127] = i
            placed.append((i, x1, y1, w, h, int((sp[..., 3] > 127).sum())))
        noise = rng.normal(0, 3, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / "img1" / f"{f + 1:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        for (i, x1, y1, w, h, full_px) in placed:
            vis = float((owner == i).sum()) / max(full_px, 1)
            # clip box to image like MOT annotations of people at the border
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(W, x1 + w), min(H, y1 + h)
            if cx2 - cx1 < 10:
                continue
            gt_rows.append(f"{f + 1},{i + 1},{cx1},{cy1},{cx2 - cx1},{cy2 - cy1},1,1,{min(vis, 1.0):.3f}")
    (out / "gt" / "gt.txt").write_text("\n".join(gt_rows) + "\n")
    (out / "seqinfo.ini").write_text(
        f"[Sequence]\nname={out.name}\nimDir=img1\nframeRate={FPS}\nseqLength={N_FRAMES}\n"
        f"imWidth={W}\nimHeight={H}\nimExt=.jpg\n")
    print(f"wrote {N_FRAMES} frames, {len(gt_rows)} GT boxes, {len(wks)} identities -> {out}")


if __name__ == "__main__":
    main()
