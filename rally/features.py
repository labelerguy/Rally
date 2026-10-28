"""Clothing appearance features. No jerseys at practice, so we identify people by what they wear."""
from __future__ import annotations

import cv2
import numpy as np

# COCO keypoint indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16

H_BINS, S_BINS, V_BINS = 12, 3, 3
REGION_DIM = H_BINS * S_BINS * V_BINS
N_REGIONS = 3  # shirt, shorts, shoes
FEAT_DIM = REGION_DIM * N_REGIONS


def _hist(hsv_patch: np.ndarray) -> np.ndarray | None:
    if hsv_patch.size == 0 or hsv_patch.shape[0] < 3 or hsv_patch.shape[1] < 3:
        return None
    h = cv2.calcHist([hsv_patch], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256]).flatten()
    s = h.sum()
    if s <= 0:
        return None
    return h / s


def _regions(box, kp):
    """Return (shirt, shorts, shoes) boxes in pixel coords. Uses pose when confident, else box fractions.
    Shoes matter: half a team can wear the same practice shirt and black shorts, but shoes differ."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pts_ok = kp is not None and min(kp[L_SH, 2], kp[R_SH, 2], kp[L_HIP, 2], kp[R_HIP, 2]) > 0.5
    if pts_ok:
        sh_y = (kp[L_SH, 1] + kp[R_SH, 1]) / 2
        hip_y = (kp[L_HIP, 1] + kp[R_HIP, 1]) / 2
        xs = [kp[L_SH, 0], kp[R_SH, 0], kp[L_HIP, 0], kp[R_HIP, 0]]
        tx1, tx2 = min(xs), max(xs)
        pad = 0.1 * (tx2 - tx1)
        torso_len = max(hip_y - sh_y, 4)
        shirt = (tx1 + pad, sh_y + 0.1 * torso_len, tx2 - pad, hip_y - 0.05 * torso_len)
        knee_ok = min(kp[L_KN, 2], kp[R_KN, 2]) > 0.4
        kn_y = (kp[L_KN, 1] + kp[R_KN, 1]) / 2 if knee_ok else hip_y + 0.6 * torso_len
        shorts = (tx1 + pad, hip_y + 0.05 * torso_len, tx2 - pad, hip_y + 0.6 * max(kn_y - hip_y, 4))
    else:
        shirt = (x1 + 0.3 * w, y1 + 0.22 * h, x2 - 0.3 * w, y1 + 0.45 * h)
        shorts = (x1 + 0.3 * w, y1 + 0.5 * h, x2 - 0.3 * w, y1 + 0.62 * h)
    shoes = (x1 + 0.1 * w, y2 - max(4.0, 0.08 * h), x2 - 0.1 * w, y2)
    return shirt, shorts, shoes


def clothing_hist(frame: np.ndarray, box, kp, min_h: int = 34) -> np.ndarray | None:
    """Concatenated HSV histograms of shirt, shorts and shoes. None if the person is too small."""
    if box[3] - box[1] < min_h:
        return None
    H, W = frame.shape[:2]
    out = []
    for (a, b, c, d) in _regions(box, kp):
        a, b, c, d = int(max(0, a)), int(max(0, b)), int(min(W, c)), int(min(H, d))
        if c - a < 3 or d - b < 3:
            return None
        hsv = cv2.cvtColor(frame[b:d, a:c], cv2.COLOR_BGR2HSV)
        hh = _hist(hsv)
        if hh is None:
            return None
        out.append(hh)
    return np.concatenate(out).astype(np.float32)


REGION_WEIGHTS = (0.5, 0.2, 0.3)  # shirt, shorts, shoes


def hellinger(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two clothing descriptors, 0 (same) to 1 (nothing shared)."""
    d = 0.0
    for r, wgt in enumerate(REGION_WEIGHTS):
        ra, rb = a[r * REGION_DIM:(r + 1) * REGION_DIM], b[r * REGION_DIM:(r + 1) * REGION_DIM]
        d += wgt * np.sqrt(max(0.0, 1 - float(np.sum(np.sqrt(ra * rb)))))
    return float(d)


def pairwise_hellinger(X: np.ndarray) -> np.ndarray:
    """All-pairs version of hellinger() for an (n, FEAT_DIM) matrix. Same result, thousands of times faster."""
    S = np.sqrt(np.clip(X, 0, None)).astype(np.float64)
    D = np.zeros((len(X), len(X)))
    for r, wgt in enumerate(REGION_WEIGHTS):
        R = S[:, r * REGION_DIM:(r + 1) * REGION_DIM]
        D += wgt * np.sqrt(np.clip(1 - R @ R.T, 0, None))
    return D.astype(np.float32)


def crop_person(frame: np.ndarray, box, out_h: int = 160) -> np.ndarray | None:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = 0.15 * (x2 - x1), 0.08 * (y2 - y1)
    a, b = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    c, d = int(min(W, x2 + pw)), int(min(H, y2 + ph))
    if c - a < 4 or d - b < 8:
        return None
    crop = frame[b:d, a:c]
    scale = out_h / crop.shape[0]
    return cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), out_h), interpolation=cv2.INTER_CUBIC)
