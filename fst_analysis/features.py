"""Forced swim test: per-frame movement features for the two rats in a side-view video.

Pipeline per video (two cylinders side by side, one white rat in each):
  1. Sample frames at ~10 fps, downscaled.
  2. Rat mask = white, unsaturated pixels that are clearly brighter than that pixel's
     usual (30th-percentile) brightness over the session, so constant glare and the
     cylinder edges drop out.
  3. Split left/right at the column the rats never occupy (the gap between cylinders).
  4. For each side and frame, two movement measures over a 0.2 s lag:
       - flow_resid: dense optical flow inside the rat, minus the rat's median flow
         (removes passive drift), mean magnitude divided by sqrt(rat area) so it does
         not depend on how large the rat appears;
       - shape_change: 1 - overlap of the rat silhouette with itself 0.2 s earlier.
"""

import cv2
import numpy as np

SAMPLE_FPS = 10.0
LONG_SIDE = 512


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sc = LONG_SIDE / max(w, h)
    size = (int(round(w * sc)), int(round(h * sc)))
    step = max(1, int(round(fps / SAMPLE_FPS)))
    grays, whites, idx = [], [], []
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            whites.append(((hsv[..., 2] > 150) & (hsv[..., 1] < 60)).astype(np.uint8))
            idx.append(i)
        i += 1
    cap.release()
    return np.stack(grays), np.stack(whites), np.array(idx) / fps, fps / step, fps, i / fps


def rat_masks(G, Wm):
    base = np.percentile(G[::10], 30, axis=0).astype(np.int16)
    R = np.zeros(G.shape, np.uint8)
    k = np.ones((3, 3), np.uint8)
    for i in range(len(G)):
        r = ((Wm[i] > 0) & ((G[i].astype(np.int16) - base) > 40)).astype(np.uint8)
        R[i] = cv2.morphologyEx(r, cv2.MORPH_OPEN, k)
    return R


def split_column(R):
    W = R.shape[2]
    prof = R[::5].mean(0).mean(0).astype(np.float32).reshape(1, -1)
    prof = cv2.GaussianBlur(prof, (31, 1), 0).ravel()
    lo, hi = int(W * 0.3), int(W * 0.7)
    return int(np.argmin(prof[lo:hi])) + lo


def side_features(G, R, x0, x1, lag):
    n = len(G)
    flow_resid = np.full(n, np.nan)
    shape_change = np.full(n, np.nan)
    area = R[:, :, x0:x1].reshape(n, -1).sum(1).astype(float)
    for i in range(lag, n):
        a, b = R[i - lag, :, x0:x1] > 0, R[i, :, x0:x1] > 0
        u = (a | b).sum()
        if u < 50 or area[i] < 50:
            continue
        shape_change[i] = 1 - (a & b).sum() / u
        ys, xs = np.nonzero(a | b)
        y0, y1, xa, xb = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pad = 8
        y0, xa = max(0, y0 - pad), max(0, xa - pad)
        y1, xb = min(G.shape[1], y1 + pad), min(x1 - x0, xb + pad)
        g0 = G[i - lag, y0:y1, x0 + xa:x0 + xb]
        g1 = G[i, y0:y1, x0 + xa:x0 + xb]
        if g0.shape[0] < 16 or g0.shape[1] < 16:
            continue
        fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 2, 15, 3, 5, 1.1, 0)
        m = a[y0:y1, xa:xb]
        if m.sum() < 30:
            continue
        v = fl[m]
        resid = v - np.median(v, axis=0)
        flow_resid[i] = float(np.mean(np.hypot(resid[:, 0], resid[:, 1]))) / np.sqrt(area[i])
    return {"flow_resid": flow_resid, "shape_change": shape_change, "area": area}


def extract(path):
    G, Wm, t, sfps, vfps, dur = read_frames(path)
    R = rat_masks(G, Wm)
    mid = split_column(R)
    lag = max(1, int(round(0.2 * sfps)))
    out = {"t": t, "sample_fps": sfps, "video_fps": vfps, "duration_s": dur, "split_x": mid, "width": G.shape[2]}
    for side, (a, b) in (("left", (0, mid)), ("right", (mid, G.shape[2]))):
        out[side] = side_features(G, R, a, b, lag)
    return out
