"""Drawing helpers: tracked video with trajectory tails, trajectory plot, switch gallery."""
from __future__ import annotations

import colorsys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


def color_for_id(tid: int) -> tuple[int, int, int]:
    h = (tid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _label(img, text, org, color, scale=0.6):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - bl - 2), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - bl), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)


def render_video(seq, rows: np.ndarray, out_path: str | Path, start: int = 1, end: int | None = None,
                 tail: int = 45, scale: float = 0.6, gif_path: str | Path | None = None, gif_every: int | None = None,
                 gif_width: int = 480, gif_max_frames: int = 110, title: str = ""):
    """Draw tracked boxes, IDs and fading trajectory tails (foot points)."""
    end = min(end or seq.seq_length, seq.seq_length)
    if gif_every is None:  # keep the README gif small (~a few MB)
        gif_every = max(2, int(np.ceil((end - start + 1) / gif_max_frames)))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    W, H = int(seq.width * scale), int(seq.height * scale)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), seq.frame_rate, (W, H))
    tails: dict[int, deque] = defaultdict(lambda: deque(maxlen=tail))
    seen = set()
    gif_frames = []
    for f in range(start, end + 1):
        img = seq.read(f)
        cur = rows[rows[:, 0] == f]
        active = set()
        for r in cur:
            tid = int(r[1])
            x, y, w, h = r[2:6]
            tails[tid].append((x + w / 2, y + h))
            active.add(tid)
            seen.add(tid)
        overlay = img.copy()
        for tid, pts in tails.items():
            if tid not in active or len(pts) < 2:
                continue
            p = np.array(pts, dtype=np.int32)
            col = color_for_id(tid)
            for k in range(1, len(p)):
                th = max(1, int(1 + 4 * k / len(p)))
                cv2.line(overlay, tuple(p[k - 1]), tuple(p[k]), col, th, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
        for r in cur:
            tid = int(r[1])
            x, y, w, h = r[2:6].astype(int)
            col = color_for_id(tid)
            dashed = r[6] < 0  # interpolated box (person occluded, position inferred)
            if dashed:
                for xx in range(x, x + w, 12):
                    cv2.line(img, (xx, y), (min(xx + 6, x + w), y), col, 2)
                    cv2.line(img, (xx, y + h), (min(xx + 6, x + w), y + h), col, 2)
                for yy in range(y, y + h, 12):
                    cv2.line(img, (x, yy), (x, min(yy + 6, y + h)), col, 2)
                    cv2.line(img, (x + w, yy), (x + w, min(yy + 6, y + h)), col, 2)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), col, 3)
            _label(img, f"{tid}", (x, y), col, 0.8)
        hud = f"{title}  frame {f}/{end}   in view: {len(active)}   unique IDs so far: {len(seen)}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 44), (20, 20, 20), -1)
        cv2.putText(img, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        vw.write(small)
        if gif_path is not None and (f - start) % gif_every == 0:
            gh = int(H * gif_width / W)
            gif_frames.append(cv2.cvtColor(cv2.resize(small, (gif_width, gh), interpolation=cv2.INTER_AREA),
                                           cv2.COLOR_BGR2RGB))
    vw.release()
    if gif_path is not None and gif_frames:
        from PIL import Image

        ims = [Image.fromarray(g).quantize(colors=96, method=Image.Quantize.MEDIANCUT) for g in gif_frames]
        ims[0].save(gif_path, save_all=True, append_images=ims[1:], loop=0,
                    duration=int(1000 * gif_every / seq.frame_rate), optimize=True)
    return out_path


def plot_trajectories(seq, rows: np.ndarray, out_path: str | Path, start: int = 1, end: int | None = None,
                      min_len: int = 10, title: str = ""):
    """All foot-point trajectories of a clip drawn over its middle frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    end = min(end or seq.seq_length, seq.seq_length)
    sel = rows[(rows[:, 0] >= start) & (rows[:, 0] <= end)]
    bg = cv2.cvtColor(seq.read((start + end) // 2), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 12 * seq.height / seq.width))
    ax.imshow((bg * 0.45).astype(np.uint8))
    n = 0
    for tid in np.unique(sel[:, 1]).astype(int):
        tr = sel[sel[:, 1] == tid]
        if len(tr) < min_len:
            continue
        tr = tr[np.argsort(tr[:, 0])]
        px, py = tr[:, 2] + tr[:, 4] / 2, tr[:, 3] + tr[:, 5]
        b, g, r = color_for_id(tid)
        c = (r / 255, g / 255, b / 255)
        ax.plot(px, py, "-", color=c, lw=2, alpha=0.95)
        ax.plot(px[0], py[0], "o", color=c, ms=5)
        if len(px) > 2:
            ax.annotate("", xy=(px[-1], py[-1]), xytext=(px[-3], py[-3]),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
        ax.text(px[-1] + 4, py[-1] - 4, str(tid), color=c, fontsize=9, weight="bold")
        n += 1
    ax.set_title(title or f"{seq.name} frames {start}-{end}: {n} trajectories (dot = start, arrow = end)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def switch_gallery(seq, cases, rows: np.ndarray, out_path: str | Path, n: int = 6, pad: float = 1.2):
    """For n ID-switch cases, show the scene before (last correct frame) and at the switch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = cases[:n]
    if not cases:
        return None
    fig, axes = plt.subplots(len(cases), 2, figsize=(9, 3.3 * len(cases)))
    axes = np.atleast_2d(axes)
    for k, c in enumerate(cases):
        for j, f in enumerate([c.prev_frame, c.frame]):
            img = seq.read(f)
            cur = rows[rows[:, 0] == f]
            # crop region around the involved tracks
            sub = cur[np.isin(cur[:, 1], [c.old_hid, c.new_hid])]
            if len(sub) == 0:
                sub = cur
            x1, y1 = sub[:, 2].min(), sub[:, 3].min()
            x2, y2 = (sub[:, 2] + sub[:, 4]).max(), (sub[:, 3] + sub[:, 5]).max()
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * (1 + pad), (y2 - y1) * (1 + pad * 0.5)
            w = max(w, h * 1.3)
            X1, Y1 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
            X2, Y2 = int(min(seq.width, cx + w / 2)), int(min(seq.height, cy + h / 2))
            for r in cur:
                tid = int(r[1])
                x, y, ww, hh = r[2:6].astype(int)
                thick = 4 if tid in (c.old_hid, c.new_hid) else 1
                cv2.rectangle(img, (x, y), (x + ww, y + hh), color_for_id(tid), thick)
                _label(img, str(tid), (x, y), color_for_id(tid), 0.7)
            crop = cv2.cvtColor(img[Y1:Y2, X1:X2], cv2.COLOR_BGR2RGB)
            ax = axes[k, j]
            ax.imshow(crop)
            ax.axis("off")
            lab = "before" if j == 0 else "switch"
            ax.set_title(f"{lab}: frame {f}" + (f"  (ID {c.old_hid} -> {c.new_hid})" if j else ""), fontsize=9)
        axes[k, 0].text(0, -0.08, f"{c.primary} | gap={c.gap} vis={c.min_vis:.2f} crowd IoU={c.crowd_iou:.2f}"
                        f" speed={c.speed:.3f} sim={c.app_sim:.2f}", transform=axes[k, 0].transAxes,
                        fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path
