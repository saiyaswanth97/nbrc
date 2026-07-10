"""Interactive GUI to pick the 4 circle-edge points and the open (escape) hole.

Uses matplotlib (works with the tkagg backend; OpenCV is headless here).

Flow:
  1. Click 4 points on the platform rim — the circle is fit live and the
     quadrants + perimeter holes are drawn.
  2. Click the OPEN (escape) hole — it snaps to the nearest detected hole,
     is marked ESCAPE, and the target quadrant is derived from it.
  3. Press Enter to save. 'r' resets, 'u' undoes the last edge point.

If hole detection fails (too few detections), you can still press 1-4 to set the
target quadrant manually.
"""

import os
import json

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .geometry import fit_circle, quadrant_boundaries, point_quadrant
from .holes import detect_holes, identify_escape_hole


def _ensure_interactive_backend():
    """Switch to an interactive backend (plotting.py forces Agg at import)."""
    if "inline" in matplotlib.get_backend().lower():
        return
    for backend in ("TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg"):
        try:
            plt.switch_backend(backend)
            return backend
        except Exception:
            continue
    raise RuntimeError(
        "No interactive matplotlib backend available (current: "
        f"{matplotlib.get_backend()}). Install python3-tk or a Qt binding, "
        "or use the non-GUI 'quads' command with --edge-points."
    )


class QuadrantPicker:
    def __init__(self, image_path, video_path=None, n_holes=18, title=None):
        _ensure_interactive_backend()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image_path = image_path
        self.video_path = video_path
        self.n_holes = n_holes
        self.title = title or os.path.basename(image_path)

        self.edge = []
        self.target = None
        self.center = None
        self.radius = None
        self.boundaries = None
        self.holes = None
        self.escape_idx = None
        self._bg = None

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._onclick)
        self.fig.canvas.mpl_connect("key_press_event", self._onkey)
        self._redraw()

    # ---- geometry / detection ----
    def _background(self):
        if self._bg is None and self.video_path and os.path.exists(self.video_path):
            from .tracking import build_background
            cap = cv2.VideoCapture(self.video_path)
            try:
                self._bg = build_background(cap, n_samples=120)
            finally:
                cap.release()
        return self._bg

    def _fit(self):
        cx, cy, r = fit_circle(self.edge)
        self.center = (cx, cy)
        self.radius = r
        self.boundaries = quadrant_boundaries(self.center, self.edge)
        # detect holes (background improves it; falls back to the frame)
        imgs = [im for im in (self._background(), self._frame_gray) if im is not None]
        try:
            self.holes, _ = detect_holes(imgs, self.center, self.radius, n_holes=self.n_holes)
        except Exception as e:
            self.holes = None
            print(f"Hole detection failed ({e}); use keys 1-4 to set target quadrant.")

    def _set_escape(self, px, py):
        self.escape_idx = identify_escape_hole(self.holes, (px, py))
        ex, ey = self.holes[self.escape_idx]
        self.target = point_quadrant(ex, ey, self.center, self.boundaries)

    # ---- events ----
    def _onclick(self, e):
        if e.inaxes != self.ax or e.xdata is None:
            return
        if len(self.edge) < 4:
            self.edge.append([float(e.xdata), float(e.ydata)])
            if len(self.edge) == 4:
                self._fit()
        elif self.holes is not None:
            self._set_escape(e.xdata, e.ydata)
        elif self.center is not None:
            self.target = point_quadrant(e.xdata, e.ydata, self.center, self.boundaries)
        self._redraw()

    def _onkey(self, e):
        if e.key == "r":
            self.edge, self.target = [], None
            self.center = self.radius = self.boundaries = self.holes = self.escape_idx = None
        elif e.key == "u" and self.edge:
            self.edge.pop()
            self.center = self.radius = self.boundaries = self.holes = self.escape_idx = None
            self.target = None
        elif e.key in ("1", "2", "3", "4") and self.center is not None:
            self.target = int(e.key)
        elif e.key in ("enter", "return"):
            plt.close(self.fig)
            return
        self._redraw()

    # ---- drawing ----
    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.img)
        self.ax.axis("off")

        for i, (px, py) in enumerate(self.edge):
            self.ax.scatter(px, py, c="red", s=60, zorder=6)
            self.ax.text(px + 8, py - 8, str(i + 1), color="yellow", fontsize=11, fontweight="bold")

        if self.center is not None:
            cx, cy = self.center
            r = self.radius
            self.ax.add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor="lime", linewidth=2))
            self.ax.scatter(cx, cy, c="lime", s=40, marker="+", zorder=6)

            n = len(self.boundaries)
            for ang in self.boundaries:
                self.ax.plot([cx, cx + r * np.cos(ang)], [cy, cy + r * np.sin(ang)],
                             color="lime", linewidth=1.2)
            for i in range(n):
                a0 = self.boundaries[i]
                a1 = self.boundaries[(i + 1) % n]
                if a1 < a0:
                    a1 += 2 * np.pi
                mid = (a0 + a1) / 2
                lx = cx + 0.55 * r * np.cos(mid)
                ly = cy + 0.55 * r * np.sin(mid)
                is_tgt = (self.target == i + 1)
                self.ax.text(lx, ly, f"Q{i+1}" + ("*" if is_tgt else ""),
                             ha="center", va="center", fontsize=15, fontweight="bold",
                             color=("yellow" if is_tgt else "lime"))

            if self.holes is not None:
                for i, (hx, hy) in enumerate(self.holes):
                    if i == self.escape_idx:
                        continue
                    self.ax.add_patch(plt.Circle((hx, hy), 34, fill=False, edgecolor="cyan", linewidth=1.5))
                if self.escape_idx is not None:
                    ex, ey = self.holes[self.escape_idx]
                    self.ax.add_patch(plt.Circle((ex, ey), 40, fill=False, edgecolor="magenta", linewidth=3))
                    self.ax.text(ex, ey - 48, "ESCAPE", ha="center", color="magenta",
                                 fontsize=11, fontweight="bold")

        # status line
        if len(self.edge) < 4:
            msg = f"Click edge point {len(self.edge)+1}/4 on the platform rim"
        elif self.holes is not None and self.escape_idx is None:
            msg = "Click the OPEN (escape) hole.   Enter=save  r=reset  u=undo"
        elif self.escape_idx is not None:
            msg = f"Escape hole set (target Q{self.target}).   Enter=save  r=reset  u=undo"
        elif self.target is None:
            msg = "Holes not found — press 1-4 to set target quadrant.   Enter=save  r=reset"
        else:
            msg = f"Target Q{self.target}.   Enter=save  r=reset  u=undo"
        self.ax.set_title(f"{self.title}\n{msg}", fontsize=11)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()
        escape_hole = None
        if self.holes is not None and self.escape_idx is not None:
            hx, hy = self.holes[self.escape_idx]
            escape_hole = [round(float(hx), 1), round(float(hy), 1)]
        return {
            "edge_points": [[round(x, 1), round(y, 1)] for x, y in self.edge] if len(self.edge) == 4 else None,
            "escape_hole": escape_hole,
            "target_quadrant": self.target,
            "center": [round(self.center[0], 1), round(self.center[1], 1)] if self.center else None,
            "radius_px": round(self.radius, 1) if self.radius else None,
        }


def pick_and_update_config(image_path, config_path=None, video_file=None, video_path=None, n_holes=18):
    """Launch the picker; optionally write results into a config's matching video entry.

    Returns the picker result dict.
    """
    picker = QuadrantPicker(image_path, video_path=video_path, n_holes=n_holes)
    result = picker.run()

    if result["edge_points"] is None:
        print("No 4 edge points selected — nothing saved.")
        return result

    edge_str = ",".join(str(v) for pt in result["edge_points"] for v in pt)
    print(f"\nedge_points: {edge_str}")
    if result["escape_hole"]:
        print(f"escape_hole: {result['escape_hole'][0]},{result['escape_hole'][1]}")
    print(f"target_quadrant: {result['target_quadrant']}")
    print(f"circle: center={result['center']} radius={result['radius_px']}px")

    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        target_file = video_file or (stem + ".mp4")
        updated = False
        for v in cfg.get("videos", []):
            if v.get("file") == target_file or os.path.splitext(v.get("file", ""))[0] == stem:
                v["edge_points"] = edge_str
                if result["escape_hole"]:
                    v["escape_hole"] = f"{result['escape_hole'][0]},{result['escape_hole'][1]}"
                if result["target_quadrant"] is not None:
                    v["target_quadrant"] = result["target_quadrant"]
                updated = True
                break
        if updated:
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"Updated {config_path} for {target_file}")
        else:
            print(f"WARNING: no video entry matched '{stem}' in {config_path} — not saved.")

    return result
