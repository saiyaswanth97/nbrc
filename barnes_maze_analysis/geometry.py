"""Barnes maze geometry: circle fitting and quadrant assignment.

The arena is a circular platform. The user annotates 4 points on the circle
edge. From those points we:
  1. Fit the circle (center + radius) via least squares.
  2. Use the 4 points as quadrant *boundaries* — a radial line from the center
     through each point splits the disk into 4 angular sectors (the quadrants).

So click the 4 edge points where you want the quadrant dividing lines to sit
(e.g. the two ends of each dividing diameter).
"""

import cv2
import numpy as np


def fit_circle(points):
    """Least-squares (Kasa) circle fit.

    Args:
        points: [[x,y], ...] — 3 or more points on the circle edge.

    Returns:
        (cx, cy, r) center and radius in the same (pixel) coordinates.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("Need at least 3 points to fit a circle")
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = np.sqrt(max(c + cx ** 2 + cy ** 2, 0.0))
    return float(cx), float(cy), float(r)


def quadrant_boundaries(center, edge_points):
    """Angles (radians, atan2 convention) of each edge point from the center.

    Returned sorted ascending. These are the dividing lines between quadrants.
    """
    cx, cy = center
    angs = [float(np.arctan2(py - cy, px - cx)) for px, py in edge_points]
    return sorted(angs)


def point_quadrant(px, py, center, boundaries):
    """Return the quadrant index (1..N) for a point.

    Quadrant i is the angular sector between the i-th and (i+1)-th sorted
    boundary angles. The last sector wraps around +/-pi. With 4 boundaries this
    yields quadrants 1-4, numbered in increasing angle from the first boundary.
    """
    cx, cy = center
    ang = float(np.arctan2(py - cy, px - cx))
    n = len(boundaries)
    for i in range(n - 1):
        if boundaries[i] <= ang < boundaries[i + 1]:
            return i + 1
    # wrap sector: from last boundary, through +/-pi, to first boundary
    return n


def normalized_radius(px, py, center, radius):
    """Distance from center as a fraction of arena radius (0=center, 1=edge)."""
    cx, cy = center
    if radius <= 0:
        return 0.0
    return float(np.hypot(px - cx, py - cy) / radius)


def make_circle_mask(center, radius, h, w):
    """Binary mask (255 inside circle) of size h x w."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(center[0])), int(round(center[1]))), int(round(radius)), 255, -1)
    return mask


def circle_bounds(center, radius, w, h):
    """Axis-aligned bounding box (x1, y1, x2, y2) of the circle, clipped to frame."""
    cx, cy = center
    x1 = max(0, int(np.floor(cx - radius)))
    y1 = max(0, int(np.floor(cy - radius)))
    x2 = min(w, int(np.ceil(cx + radius)))
    y2 = min(h, int(np.ceil(cy + radius)))
    return x1, y1, x2, y2


def circle_from_edge_points(edge_points, w, h):
    """Convenience: fit circle + boundaries from 4 edge points.

    Returns dict: center, radius, boundaries, bounds.
    """
    cx, cy, r = fit_circle(edge_points)
    center = (cx, cy)
    return {
        "center": center,
        "radius": r,
        "boundaries": quadrant_boundaries(center, edge_points),
        "bounds": circle_bounds(center, r, w, h),
    }
