"""OcclusionAwareTracker: SORT -> DeepSORT/ByteTrack-style multi-object tracker.

Per frame:

  0. Predict every track with its Kalman filter; warp predictions with the
     estimated camera motion (CMC).
  1. HIGH-confidence detections  vs  confirmed tracks (tracked + LOST)
       visible tracks: cost = mean(IoU distance, appearance distance) for
                       nearby boxes, else IoU distance
       LOST tracks:    cost = 0.8 * appearance + 0.2 * normalised Mahalanobis,
                       only inside the (inflated) Kalman gate
     Appearance lets a LOST track (occluded, no IoU overlap any more) be
     re-associated to the same ID when the person re-appears.
  2. LOW-confidence detections vs still-unmatched *tracked* tracks, IoU only
     (ByteTrack). Partially occluded people get low detector scores; using
     them keeps the track alive instead of letting it die mid-occlusion.
  3. Remaining high detections vs tentative tracks (IoU).
  4. Unmatched tracked tracks -> LOST (kept for `max_age` frames, predicted
     forward). Unmatched high detections -> new tentative tracks.

With `preset="sort"` the tracker degrades to classic SORT (IoU-only Hungarian,
tracks deleted after 1 missed frame) which is used as the ablation baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import IntEnum

import numpy as np

from .kalman import KalmanFilter, CHI2INV95, tlwh_to_xyah, xyah_to_tlwh, tlwh_to_tlbr
from .matching import iou_matrix, linear_assignment, embedding_distance
from .cmc import apply_cmc


class TrackState(IntEnum):
    TENTATIVE = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5        # detections >= this go to stage 1
    low_thresh: float = 0.1         # detections in [low, high) go to stage 2
    new_track_thresh: float = 0.6   # min score to start a new track
    match_thresh: float = 0.8       # stage-1 max cost (1 - IoU)
    low_match_thresh: float = 0.5   # stage-2 max cost
    tentative_match_thresh: float = 0.7
    max_age: int = 30               # frames a LOST track survives (scaled by fps)
    n_init: int = 2                 # consecutive hits to confirm a track
    use_low_dets: bool = True       # ByteTrack second stage
    use_appearance: bool = True
    appearance_thresh: float = 0.25  # max cosine distance to accept an appearance match
    proximity_thresh: float = 0.5    # tracked tracks: appearance only used if 1-IoU < this
    reid_gate: bool = True           # LOST tracks: appearance used only inside Kalman gate
    gate_inflation: float = 4.0      # multiply chi2 gate (occlusion -> sloppy motion)
    motion_weight: float = 0.2       # LOST re-ID cost = 0.8 * appearance + 0.2 * normalised Mahalanobis
    ema_alpha: float = 0.9           # appearance feature smoothing
    feat_update_max_overlap: float = 0.3  # skip feature update if crop overlaps another det
    use_cmc: bool = True
    freeze_size_when_lost: bool = True

    @classmethod
    def preset(cls, name: str, **overrides) -> "TrackerConfig":
        presets = {
            # Classic SORT (Bewley et al. 2016): IoU Hungarian, no memory.
            "sort": dict(use_low_dets=False, use_appearance=False, use_cmc=False, max_age=1,
                         n_init=3, new_track_thresh=0.5, match_thresh=0.7, reid_gate=False),
            # SORT + track buffer: occluded tracks survive, but re-association is IoU-only.
            "sort_buffer": dict(use_low_dets=False, use_appearance=False, use_cmc=False, max_age=30,
                                n_init=3, new_track_thresh=0.5, match_thresh=0.7),
            # + ByteTrack low-score association.
            "byte": dict(use_appearance=False, use_cmc=False),
            # + camera-motion compensation.
            "byte_cmc": dict(use_appearance=False, use_cmc=True),
            # Full: + appearance re-identification of LOST tracks.
            "full": dict(),
        }
        if name not in presets:
            raise ValueError(f"unknown preset {name}; choose from {list(presets)}")
        cfg = cls(**presets[name])
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    def to_dict(self):
        return asdict(self)


class Track:
    _next_id = 1

    def __init__(self, tlwh, score, feat, kf: KalmanFilter, frame_id: int):
        self.kf = kf
        self.mean, self.cov = kf.initiate(tlwh_to_xyah(tlwh))
        self.score = score
        self.feat = feat
        self.state = TrackState.TENTATIVE
        self.track_id = 0
        self.hits = 1
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.time_since_update = 0

    def activate(self, frame_id):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.state = TrackState.TRACKED
        self.frame_id = frame_id

    @property
    def tlwh(self):
        return xyah_to_tlwh(self.mean[:4])

    @property
    def tlbr(self):
        return tlwh_to_tlbr(self.tlwh)

    def update(self, tlwh, score, feat, frame_id, update_feat: bool, ema_alpha: float):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, tlwh_to_xyah(tlwh), score)
        self.score = score
        if feat is not None and update_feat:
            if self.feat is None:
                self.feat = feat
            else:
                f = ema_alpha * self.feat + (1 - ema_alpha) * feat
                self.feat = f / (np.linalg.norm(f) + 1e-9)
        self.hits += 1
        self.time_since_update = 0
        self.frame_id = frame_id


@dataclass
class ReIDEvent:
    frame: int
    track_id: int
    gap: int          # frames the track was lost
    via: str          # "iou" or "appearance"


class OcclusionAwareTracker:
    def __init__(self, config: TrackerConfig | None = None, frame_rate: int = 30):
        self.cfg = config or TrackerConfig()
        self.kf = KalmanFilter()
        self.tracks: list[Track] = []
        self.frame_id = 0
        # Lost buffer is defined in frames at 30 fps; scale for other frame rates.
        self.max_age = max(1, int(round(self.cfg.max_age * frame_rate / 30.0)))
        self.reid_events: list[ReIDEvent] = []
        Track._next_id = 1

    # ----------------------------------------------------------------- utils
    def _predict(self, tracks: list[Track], H: np.ndarray | None):
        if not tracks:
            return
        means = np.stack([t.mean.copy() for t in tracks])
        covs = np.stack([t.cov for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.TRACKED:
                means[i, 7] = 0.0  # don't extrapolate height while unobserved
                if self.cfg.freeze_size_when_lost:
                    means[i, 6] = 0.0
        means, covs = self.kf.multi_predict(means, covs)
        if H is not None:
            means, covs = apply_cmc(means, covs, H)
        for i, t in enumerate(tracks):
            t.mean, t.cov = means[i], covs[i]
            t.time_since_update += 1

    def _cost(self, tracks: list[Track], det_tlbr, det_xyah, det_feats, use_app: bool):
        iou_cost = 1.0 - iou_matrix(np.array([t.tlbr for t in tracks]).reshape(-1, 4), det_tlbr)
        if not use_app or det_feats is None or len(tracks) == 0 or len(det_tlbr) == 0:
            return iou_cost, iou_cost, None
        tfeats = np.stack([t.feat if t.feat is not None else np.zeros(det_feats.shape[1]) for t in tracks])
        emb = embedding_distance(tfeats, det_feats)
        emb_gated = emb.copy()
        for i, t in enumerate(tracks):
            if t.feat is None:
                emb_gated[i] = 1.0
                continue
            if t.state == TrackState.TRACKED:
                # Visible track: appearance only refines nearby candidates. Averaging with
                # the IoU cost (rather than taking the minimum) keeps geometry decisive
                # when two candidates look alike.
                ok = (iou_cost[i] <= self.cfg.proximity_thresh) & (emb[i] <= self.cfg.appearance_thresh)
                emb_gated[i] = np.where(ok, 0.5 * iou_cost[i] + 0.5 * emb[i], 1.0)
                continue
            elif self.cfg.reid_gate:
                # LOST track: allow re-ID anywhere inside the (inflated) Kalman gate.
                gate = CHI2INV95[2] * self.cfg.gate_inflation
                gd = self.kf.gating_distance(t.mean, t.cov, det_xyah, only_position=True)
                # Small motion term breaks ties between look-alikes: the candidate
                # closer to the predicted position wins when appearance can't decide.
                ok = (gd <= gate) & (emb[i] <= self.cfg.appearance_thresh)
                mixed = (1 - self.cfg.motion_weight) * emb[i] + self.cfg.motion_weight * np.minimum(gd / gate, 1.0)
                emb_gated[i] = np.where(ok, np.minimum(mixed, self.cfg.appearance_thresh), 1.0)
        emb_gated[emb_gated > self.cfg.appearance_thresh] = 1.0
        fused = np.minimum(iou_cost, emb_gated)
        return fused, iou_cost, emb_gated

    # ------------------------------------------------------------------ main
    def update(self, dets: np.ndarray, feats: np.ndarray | None = None, H: np.ndarray | None = None):
        """dets: Nx5 [x1, y1, x2, y2, score]; feats: NxD L2-normalised or None.

        Returns list of (track_id, tlwh, score) for tracks visible this frame.
        """
        cfg = self.cfg
        self.frame_id += 1
        dets = np.asarray(dets, dtype=float).reshape(-1, 5)
        scores = dets[:, 4]
        if feats is None or not cfg.use_appearance:
            feats = None

        high = scores >= cfg.high_thresh
        low = (scores >= cfg.low_thresh) & ~high
        if not cfg.use_low_dets:
            low[:] = False
        hi_idx, lo_idx = np.where(high)[0], np.where(low)[0]

        # Pairwise det overlap -> don't learn appearance from occluded crops.
        det_overlap = iou_matrix(dets[:, :4], dets[:, :4])
        np.fill_diagonal(det_overlap, 0.0)
        max_overlap = det_overlap.max(axis=1) if len(dets) > 1 else np.zeros(len(dets))

        all_tlwh = dets[:, :4].copy()
        all_tlwh[:, 2:] -= all_tlwh[:, :2]
        all_xyah = tlwh_to_xyah(all_tlwh) if len(dets) else np.zeros((0, 4))

        confirmed = [t for t in self.tracks if t.state in (TrackState.TRACKED, TrackState.LOST)]
        tentative = [t for t in self.tracks if t.state == TrackState.TENTATIVE]
        self._predict(self.tracks, H if cfg.use_cmc else None)

        def _apply(track: Track, di: int, via: str):
            was_lost = track.state == TrackState.LOST
            gap = track.time_since_update
            f = feats[di] if feats is not None else None
            track.update(all_tlwh[di], scores[di], f, self.frame_id,
                         update_feat=(max_overlap[di] < cfg.feat_update_max_overlap and scores[di] >= cfg.high_thresh),
                         ema_alpha=cfg.ema_alpha)
            if was_lost:
                self.reid_events.append(ReIDEvent(self.frame_id, track.track_id, gap - 1, via))
            track.state = TrackState.TRACKED

        # ---- stage 1: high dets vs confirmed (tracked + lost)
        d_feats = feats[hi_idx] if feats is not None else None
        cost, iou_cost, _ = self._cost(confirmed, dets[hi_idx, :4], all_xyah[hi_idx], d_feats, cfg.use_appearance)
        matches, um_trk, um_det = linear_assignment(cost, cfg.match_thresh)
        for ti, di in matches:
            via = "iou" if iou_cost[ti, di] <= cost[ti, di] + 1e-9 else "appearance"
            _apply(confirmed[ti], hi_idx[di], via)
        rem_hi = hi_idx[um_det]
        rem_trk = [confirmed[i] for i in um_trk]

        # ---- stage 2: low dets vs still-unmatched *tracked* tracks (IoU only)
        r_tracked = [t for t in rem_trk if t.state == TrackState.TRACKED]
        if len(lo_idx) and r_tracked:
            c2 = 1.0 - iou_matrix(np.array([t.tlbr for t in r_tracked]), dets[lo_idx, :4])
            m2, um2, _ = linear_assignment(c2, cfg.low_match_thresh)
            for ti, di in m2:
                _apply(r_tracked[ti], lo_idx[di], "iou")
            r_tracked = [r_tracked[i] for i in um2]
        for t in r_tracked:
            t.state = TrackState.LOST

        # ---- stage 3: tentative tracks vs remaining high dets
        if tentative and len(rem_hi):
            c3 = 1.0 - iou_matrix(np.array([t.tlbr for t in tentative]), dets[rem_hi, :4])
            m3, um3, umd3 = linear_assignment(c3, cfg.tentative_match_thresh)
            for ti, di in m3:
                t = tentative[ti]
                _apply(t, rem_hi[di], "iou")
                t.state = TrackState.TENTATIVE
                if t.hits >= cfg.n_init:
                    t.activate(self.frame_id)
            for ti in um3:
                tentative[ti].state = TrackState.REMOVED
            rem_hi = rem_hi[umd3]
        else:
            for t in tentative:
                t.state = TrackState.REMOVED

        # ---- births
        for di in rem_hi:
            if scores[di] < cfg.new_track_thresh:
                continue
            f = feats[di] if feats is not None else None
            t = Track(all_tlwh[di], scores[di], f, self.kf, self.frame_id)
            if cfg.n_init <= 1 or self.frame_id == 1:
                t.activate(self.frame_id)
            self.tracks.append(t)

        # ---- deaths
        for t in self.tracks:
            if t.state == TrackState.LOST and t.time_since_update > self.max_age:
                t.state = TrackState.REMOVED
        self.tracks = [t for t in self.tracks if t.state != TrackState.REMOVED]

        return [(t.track_id, t.tlwh.copy(), t.score) for t in self.tracks
                if t.state == TrackState.TRACKED and t.time_since_update == 0 and t.track_id > 0]
