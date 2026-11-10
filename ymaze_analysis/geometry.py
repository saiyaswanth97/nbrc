"""Y-maze geometry: detect the maze, label the arms, build the tracking mask.

A maze is described by its junction centre, the three arm tips and the arm width,
all in original video pixel coordinates:

    {"center": [x, y], "tips": {"A": [x, y], "B": [x, y], "C": [x, y]}, "width": w}

Arms are labelled by direction so labels match across videos and sessions:
A is the arm pointing most nearly straight up in the image, then B and C follow
clockwise on screen.
"""

import cv2
import numpy as np

ARMS = ("A", "B", "C")


def label_arms(center, tips):
    """Assign A/B/C to 3 tip points: A = most upward arm, then clockwise on screen."""
    c = np.asarray(center, float)
    ang = [np.arctan2(t[1] - c[1], t[0] - c[0]) for t in tips]  # image coords, y down
    up = -np.pi / 2
    # Angle measured clockwise on screen from "up" (atan2 grows clockwise when y points down)
    rel = [(a - up) % (2 * np.pi) for a in ang]
    # A = smallest deviation from up in either direction
    dev = [min(r, 2 * np.pi - r) for r in rel]
    a_idx = int(np.argmin(dev))
    order = sorted(range(3), key=lambda i: (rel[i] - rel[a_idx]) % (2 * np.pi))
    return {lab: [float(tips[i][0]), float(tips[i][1])] for lab, i in zip(ARMS, order)}


def arm_frames(maze):
    """Per arm: unit direction from centre to tip and centre-to-tip length (px)."""
    c = np.asarray(maze["center"], float)
    out = {}
    for lab in ARMS:
        v = np.asarray(maze["tips"][lab], float) - c
        L = float(np.linalg.norm(v))
        out[lab] = (v / L, L)
    return out


def maze_mask(maze, shape, margin=0.35):
    """Binary mask of the Y (three arm rectangles), widened by ``margin`` x arm width
    so a rat rearing against a wall stays inside."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    c = np.asarray(maze["center"], float)
    half = maze["width"] * (0.5 + margin)
    for lab in ARMS:
        tip = np.asarray(maze["tips"][lab], float)
        u = (tip - c) / np.linalg.norm(tip - c)
        n = np.array([-u[1], u[0]])
        end = tip + u * maze["width"] * margin
        start = c - u * half
        poly = np.array([start + n * half, end + n * half, end - n * half, start - n * half])
        cv2.fillPoly(mask, [np.int32(np.round(poly))], 255)
    return mask


def maze_polygon(maze, shape, margin=0.35):
    """Outline of maze_mask as a polygon (list of [x, y]) for the tracker's ROI."""
    mask = maze_mask(maze, shape, margin)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    return cnt.reshape(-1, 2).tolist()


def _fit_from_mask(mask):
    """Fit centre / tips / width to a binary Y-shaped mask. Returns (maze, problems)."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(np.argmax(dt), dt.shape)  # junction = widest point
    r0 = float(dt[cy, cx])

    ys, xs = np.nonzero(mask)
    dx, dy = xs - cx, ys - cy
    rad = np.hypot(dx, dy)
    far = rad > 2.5 * r0
    ang = np.arctan2(dy[far], dx[far])

    # Three dominant directions from a smoothed circular histogram
    bins = 360
    hist, _ = np.histogram(ang, bins=bins, range=(-np.pi, np.pi))
    k = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2)
    hs = np.convolve(np.concatenate([hist[-15:], hist, hist[:15]]), k, mode="same")[15:-15]
    peaks = []
    hs_work = hs.copy()
    for _ in range(3):
        p = int(np.argmax(hs_work))
        if hs_work[p] <= 0:
            break
        peaks.append(-np.pi + (p + 0.5) * 2 * np.pi / bins)
        idx = (np.arange(bins) - p) % bins
        hs_work[(idx < 50) | (idx > bins - 50)] = 0  # suppress +/-50 degrees
    if len(peaks) < 3:
        return None, ["could not find three arms"]

    tips, widths, axes = [], [], []
    for pk in peaks:
        d = np.angle(np.exp(1j * (ang - pk)))
        sel = np.abs(d) < np.radians(25)
        px, py = xs[far][sel].astype(float), ys[far][sel].astype(float)
        pts = np.stack([px, py], 1)
        mean = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
        u = vt[0]
        if np.dot(u, mean - [cx, cy]) < 0:
            u = -u
        proj = (pts - mean) @ u
        perp = (pts - mean) @ np.array([-u[1], u[0]])
        widths.append(float(np.percentile(perp, 97) - np.percentile(perp, 3)))
        axes.append((mean, u))
        tips.append(mean + u * np.percentile(proj, 99.5))

    # Centre = least-squares intersection of the three arm axes
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for mean, u in axes:
        P = np.eye(2) - np.outer(u, u)
        A += P
        b += P @ mean
    center = np.linalg.solve(A, b)

    maze = {"center": [float(center[0]), float(center[1])],
            "tips": label_arms(center, [t.tolist() for t in tips]),
            "width": float(np.median(widths))}
    return maze, validate(maze)


def validate(maze):
    """Shape checks; returns a list of problems (empty = looks like a Y-maze)."""
    problems = []
    frames = arm_frames(maze)
    lengths = [frames[a][1] for a in ARMS]
    if max(lengths) / min(lengths) > 1.6:
        problems.append(f"arm lengths uneven ({', '.join(f'{x:.0f}' for x in lengths)} px)")
    dirs = [frames[a][0] for a in ARMS]
    for i in range(3):
        for j in range(i + 1, 3):
            deg = np.degrees(np.arccos(np.clip(np.dot(dirs[i], dirs[j]), -1, 1)))
            if not 85 <= deg <= 155:
                problems.append(f"arms {ARMS[i]}/{ARMS[j]} at {deg:.0f} deg (expect ~120)")
    ratio = maze["width"] / np.mean(lengths)
    if not 0.08 <= ratio <= 0.45:
        problems.append(f"width/length ratio {ratio:.2f} out of range")
    return problems


def detect_maze(bg_gray, thresholds=(60, 70, 80, 90, 50)):
    """Detect the (dark) maze in a background image. Returns (maze, problems).

    Tries several darkness thresholds and returns the first fit that passes the
    shape checks; otherwise the last fit with its problems listed.
    """
    blur = cv2.GaussianBlur(bg_gray, (9, 9), 0)
    best = (None, ["no dark region found"])
    for t in thresholds:
        th = (blur < t).astype(np.uint8) * 255
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
        if n < 2:
            continue
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.zeros_like(th)
        cnts, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, cnts, -1, 255, -1)  # fill holes (e.g. the rat)
        maze, problems = _fit_from_mask(mask)
        if maze is None:
            continue
        maze["threshold"] = t
        if not problems:
            return maze, []
        best = (maze, problems)
    return best


def warp_maze(maze, H):
    """Map a maze through a homography (for propagating to a shifted camera)."""
    def w(p):
        q = cv2.perspectiveTransform(np.float32([[p]]), H)[0, 0]
        return [float(q[0]), float(q[1])]
    c = w(maze["center"])
    tips = [w(maze["tips"][a]) for a in ARMS]
    scale = np.mean([np.linalg.norm(np.subtract(t, c)) for t in tips]) / np.mean(
        [np.linalg.norm(np.subtract(maze["tips"][a], maze["center"])) for a in ARMS])
    return {"center": c, "tips": label_arms(c, tips), "width": float(maze["width"] * scale)}


def draw_maze(img, maze, entry_frac=None, exit_frac=None, color=(0, 220, 255)):
    """Draw arm outlines, labels and (optionally) entry/exit lines on a BGR image."""
    c = np.asarray(maze["center"], float)
    half = maze["width"] / 2
    for lab in ARMS:
        tip = np.asarray(maze["tips"][lab], float)
        L = np.linalg.norm(tip - c)
        u = (tip - c) / L
        n = np.array([-u[1], u[0]])
        poly = np.array([c + n * half, tip + n * half, tip - n * half, c - n * half])
        cv2.polylines(img, [np.int32(np.round(poly))], True, color, 2)
        for frac, col in ((entry_frac, (0, 200, 0)), (exit_frac, (0, 140, 255))):
            if frac:
                p = c + u * L * frac
                cv2.line(img, tuple(np.int32(p + n * half)), tuple(np.int32(p - n * half)), col, 2)
        lp = tip + u * 30
        cv2.putText(img, lab, (int(lp[0]) - 12, int(lp[1]) + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, color, 3)
    cv2.circle(img, tuple(np.int32(c)), 6, color, -1)
    return img
