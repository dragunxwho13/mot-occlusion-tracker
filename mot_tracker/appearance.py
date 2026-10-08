"""Appearance embeddings used to re-identify people after occlusion.

Two backends:

* ``hist``   -- part-based HSV colour histogram (3 horizontal stripes: head /
               torso / legs). No weights, runs anywhere, surprisingly strong
               for short-term re-ID because clothing colour is the dominant
               cue over a few seconds.
* ``resnet`` -- ImageNet-pretrained ResNet-18 global-pooled features from
               torchvision (downloads ~45 MB the first time). Not a re-ID
               model, but a reasonable generic descriptor.

Every embedding is L2-normalised so cosine distance = 1 - dot product.
"""
from __future__ import annotations

import numpy as np
import cv2


def _crop(img: np.ndarray, tlbr: np.ndarray, shrink_w: float = 0.0) -> np.ndarray | None:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = tlbr
    if shrink_w > 0:
        dw = (x2 - x1) * shrink_w
        x1, x2 = x1 + dw, x2 - dw
    x1, y1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
    x2, y2 = int(min(W, np.ceil(x2))), int(min(H, np.ceil(y2)))
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    return img[y1:y2, x1:x2]


class HistogramEmbedder:
    name = "hist"

    def __init__(self, h_bins: int = 16, s_bins: int = 8, stripes: int = 3):
        self.h_bins, self.s_bins, self.stripes = h_bins, s_bins, stripes
        self.dim = h_bins * s_bins * stripes

    def __call__(self, img: np.ndarray, boxes_tlbr: np.ndarray) -> np.ndarray:
        feats = np.zeros((len(boxes_tlbr), self.dim), dtype=np.float32)
        if len(boxes_tlbr) == 0:
            return feats
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i, box in enumerate(boxes_tlbr):
            # Drop 15% on each side: background leaks into person boxes there.
            crop = _crop(hsv, box, shrink_w=0.15)
            if crop is None:
                continue
            h = crop.shape[0]
            parts = []
            for s in range(self.stripes):
                stripe = crop[int(s * h / self.stripes): int((s + 1) * h / self.stripes)]
                # Ignore very dark / desaturated pixels whose hue is noise.
                mask = ((stripe[..., 2] > 30)).astype(np.uint8)
                hist = cv2.calcHist([stripe], [0, 1], mask, [self.h_bins, self.s_bins], [0, 180, 0, 256])
                hist = hist.flatten()
                hist /= hist.sum() + 1e-9
                parts.append(np.sqrt(hist))  # Hellinger mapping
            f = np.concatenate(parts)
            feats[i] = f / (np.linalg.norm(f) + 1e-9)
        return feats


class ResNetEmbedder:
    name = "resnet"

    def __init__(self, device: str = "cpu"):
        import torch
        import torchvision

        self.torch = torch
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.device = device
        self.dim = 512
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, img: np.ndarray, boxes_tlbr: np.ndarray) -> np.ndarray:
        feats = np.zeros((len(boxes_tlbr), self.dim), dtype=np.float32)
        crops, idx = [], []
        for i, box in enumerate(boxes_tlbr):
            c = _crop(img, box)
            if c is None:
                continue
            c = cv2.resize(c, (64, 128))[:, :, ::-1].astype(np.float32) / 255.0
            crops.append(((c - self.mean) / self.std).transpose(2, 0, 1))
            idx.append(i)
        if not crops:
            return feats
        with self.torch.no_grad():
            x = self.torch.from_numpy(np.stack(crops)).to(self.device)
            out = self.model(x).cpu().numpy()
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        feats[idx] = out
        return feats


def build_embedder(name: str, device: str = "cpu"):
    if name in (None, "none"):
        return None
    if name == "hist":
        return HistogramEmbedder()
    if name == "resnet":
        device = f"cuda:{device}" if str(device).isdigit() else (device or "cpu")
        return ResNetEmbedder(device)
    raise ValueError(f"unknown appearance backend: {name}")
