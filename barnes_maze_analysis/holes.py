"""Detect the Barnes maze escape holes around the platform perimeter.

The holes are evenly spaced on a ring, but the camera views the platform at a
slight angle, so in the image the ring is an ellipse (not a circle) and the holes
are perspective-distorted. Strategy:

  1. Detect as many holes as possible (CLAHE + Hough) across the mouse-free
     background and a frame; keep only robust, repeated detections in the rim ring.
  2. Fit an ellipse to those real detections.
  3. Place all N holes on the ellipse at even parametric spacing, snapping each
     slot to a real detection where one exists and filling the gaps from the grid.

This gives accurate positions even for the low-contrast holes Hough misses.
"""

import cv2
import numpy as np


def _detections(images, center, radius, ring_lo=0.62, ring_hi=0.97):
    cx, cy = center
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    dets = []
    for im in images:
        if im is None:
            continue
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(clahe.apply(im), (9, 9), 2)
        for p2 in (16, 18, 20, 22):
            c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.0, 60, param1=50,
                                 param2=p2, minRadius=30, maxRadius=54)
            if c is not None:
                for x, y, _ in c[0]:
                    if ring_lo * radius < np.hypot(x - cx, y - cy) < ring_hi * radius:
                        dets.append((float(x), float(y)))
    return np.array(dets)


def _dedupe(dets, tol=35, min_count=2):
    merged = []
    for p in dets:
        for m in merged:
            if np.hypot(*(p - m[0])) < tol:
                m[0] = (m[0] * m[1] + p) / (m[1] + 1)
                m[1] += 1
                break
        else:
            merged.append([p.copy(), 1])
    return np.array([m[0] for m in merged if m[1] >= min_count])


def detect_holes(images, center, radius, n_holes=18, snap_tol=32):
    """Return an (n_holes, 2) array of hole centers, ordered by ring angle.

    Args:
        images: list of images (grayscale or BGR) — pass the mouse-free
            background plus optionally a frame for more detections.
        center: (cx, cy) platform center (from the fitted circle).
        radius: platform radius in px.
        n_holes: number of holes on the ring (Barnes maze is typically 18 or 20).
        snap_tol: a grid slot within this many px of a real detection uses it.

    Returns:
        holes: (n_holes, 2) float array, and n_snapped (int) for diagnostics.
    """
    pts = _dedupe(_detections(images, center, radius))
    if len(pts) < 5:
        raise ValueError(f"Only {len(pts)} robust hole detections — need >=5 to fit the ring.")

    (ex, ey), (MA, ma), angd = cv2.fitEllipse(pts.astype(np.float32).reshape(-1, 1, 2))
    ec = np.array([ex, ey])
    a, b = MA / 2, ma / 2
    th = np.radians(angd)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def to_t(p):
        v = rot.T @ (np.asarray(p) - ec)
        return np.arctan2(v[1] / b, v[0] / a)

    def to_pt(t):
        return ec + rot @ np.array([a * np.cos(t), b * np.sin(t)])

    ts = np.array([to_t(p) for p in pts])
    phase = np.angle(np.mean(np.exp(1j * n_holes * ts))) / n_holes

    holes, n_snapped = [], 0
    for k in range(n_holes):
        gp = to_pt(phase + k * 2 * np.pi / n_holes)
        d = np.hypot(pts[:, 0] - gp[0], pts[:, 1] - gp[1])
        if d.min() < snap_tol:
            holes.append(pts[np.argmin(d)])
            n_snapped += 1
        else:
            holes.append(gp)

    # order by angle around the platform center for stable indexing
    holes = np.array(holes)
    order = np.argsort([np.arctan2(y - center[1], x - center[0]) for x, y in holes])
    return holes[order], n_snapped


def identify_escape_hole(holes, escape_hint):
    """Index of the hole nearest a hint point (e.g. the visible covered hole)."""
    hint = np.asarray(escape_hint, dtype=float)
    return int(np.argmin([np.hypot(x - hint[0], y - hint[1]) for x, y in holes]))
