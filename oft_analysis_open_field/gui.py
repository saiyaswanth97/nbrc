"""Interactive GUI to annotate the arena polygon and floor boundary per video.

Uses matplotlib (tkagg backend) so it works without OpenCV's highgui.

Flow (per video in the config):
  1. Click 4 POLYGON points on the arena wall corners, any order (red).
  2. Click 4 BOUNDARY points on the floor grid corners, any order (lime) —
     the grid is drawn live once all 4 are placed.
  3. Press Enter to save and move to the next video.

Keys: u=undo last point, r=reset, p=copy previous video's annotation,
      n=skip video, q=quit (annotations saved so far are kept).
"""

import os
import json

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
        f"{matplotlib.get_backend()}). Install python3-tk or a Qt binding."
    )


def _parse(s):
    if not s:
        return []
    v = [int(x) for x in s.split(",")]
    return [[v[i], v[i + 1]] for i in range(0, len(v), 2)]


def order_corners(pts):
    """Sort 4 clicked corners into TL, BL, BR, TR regardless of click order."""
    pts = sorted(pts, key=lambda p: p[1])
    tl, tr = sorted(pts[:2], key=lambda p: p[0])
    bl, br = sorted(pts[2:], key=lambda p: p[0])
    return [tl, bl, br, tr]


def _fmt(pts):
    return ",".join(str(int(round(c))) for p in pts for c in p)


def _grid_lines(boundary, rows, cols):
    """Return grid line segments for a TL,BL,BR,TR quad via perspective transform."""
    src = np.float32([[0, 0], [0, 1], [1, 1], [1, 0]])
    M = cv2.getPerspectiveTransform(src, np.float32(boundary))
    lines = []
    for i in range(1, rows):
        y = i / rows
        lines.append(np.float32([[0, y], [1, y]]))
    for j in range(1, cols):
        x = j / cols
        lines.append(np.float32([[x, 0], [x, 1]]))
    return [cv2.perspectiveTransform(l.reshape(-1, 1, 2), M).reshape(-1, 2) for l in lines]


def load_frame(video_path, sample_path=None):
    """Load the sample image if present, otherwise grab the middle frame of the video."""
    if sample_path and os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        if img is not None:
            return img
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame from {video_path}")
    return frame


class ArenaPicker:
    def __init__(self, img_bgr, title, grid="4x4", previous=None, initial=None):
        _ensure_interactive_backend()
        self.img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.title = title
        self.rows, self.cols = [int(x) for x in grid.split("x")]
        self.previous = previous
        self.polygon = [list(p) for p in initial["polygon"]] if initial else []
        self.boundary = [list(p) for p in initial["boundary"]] if initial else []
        self.reviewing = initial is not None
        self.action = None  # "save" | "skip" | "quit"

        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._onclick)
        self.fig.canvas.mpl_connect("key_press_event", self._onkey)
        self._redraw()

    def _done(self):
        return len(self.polygon) == 4 and len(self.boundary) == 4

    def _onclick(self, e):
        if e.inaxes != self.ax or e.xdata is None or e.button != 1:
            return
        # Ignore clicks while a toolbar mode (zoom/pan) is active
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return
        pt = [float(e.xdata), float(e.ydata)]
        if self._done():
            # Both quads complete: a click moves the nearest corner
            pts = self.polygon + self.boundary
            i = int(np.argmin([np.hypot(p[0] - pt[0], p[1] - pt[1]) for p in pts]))
            (self.polygon if i < 4 else self.boundary)[i % 4] = pt
        elif len(self.polygon) < 4:
            self.polygon.append(pt)
        elif len(self.boundary) < 4:
            self.boundary.append(pt)
        self._redraw()

    def _onkey(self, e):
        if e.key == "r":
            self.polygon, self.boundary = [], []
        elif e.key == "u":
            if self.boundary:
                self.boundary.pop()
            elif self.polygon:
                self.polygon.pop()
        elif e.key == "p" and self.previous:
            self.polygon = [list(p) for p in self.previous["polygon"]]
            self.boundary = [list(p) for p in self.previous["boundary"]]
        elif e.key in ("enter", "return") and self._done():
            self.action = "save"
            plt.close(self.fig)
            return
        elif e.key == "n":
            self.action = "skip"
            plt.close(self.fig)
            return
        elif e.key == "q":
            self.action = "quit"
            plt.close(self.fig)
            return
        self._redraw()

    def _draw_quad(self, pts, color, label):
        for i, (px, py) in enumerate(pts):
            self.ax.scatter(px, py, c=color, s=50, zorder=6)
            self.ax.text(px + 10, py - 10, f"{label}{i+1}", color=color,
                         fontsize=10, fontweight="bold")
        if len(pts) > 1:
            closed = pts + [pts[0]] if len(pts) == 4 else pts
            xs, ys = zip(*closed)
            self.ax.plot(xs, ys, color=color, linewidth=1.8)

    def _redraw(self):
        # Preserve zoom across redraws
        lims = (self.ax.get_xlim(), self.ax.get_ylim()) if self.ax.images else None
        self.ax.clear()
        self.ax.imshow(self.img)
        self.ax.axis("off")
        if lims:
            self.ax.set_xlim(lims[0])
            self.ax.set_ylim(lims[1])

        self._draw_quad(self.polygon, "red", "P-")
        self._draw_quad(self.boundary, "lime", "B-")
        if len(self.boundary) == 4:
            for seg in _grid_lines(order_corners(self.boundary), self.rows, self.cols):
                self.ax.plot(seg[:, 0], seg[:, 1], color="lime", linewidth=1)

        keys = "u=undo  r=reset  n=skip  q=quit" + ("  p=copy previous" if self.previous else "")
        if len(self.polygon) < 4:
            msg = f"POLYGON (arena wall corners): click corner {len(self.polygon)+1}/4"
        elif len(self.boundary) < 4:
            msg = f"BOUNDARY (floor grid corners): click corner {len(self.boundary)+1}/4"
        elif self.reviewing:
            msg = "REVIEW — Enter=looks good & next   click=move nearest corner   r=re-click all"
        else:
            msg = "Done — Enter=save & next   click=move nearest corner"
        self.ax.set_title(f"{self.title}\n{msg}\n{keys}", fontsize=11)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()
        return self.action, self.polygon, self.boundary


def annotate_config(config_path, only=None, redo=False):
    """Step through every video in the config, saving polygon/boundary after each one.

    Videos already marked ``"annotated": true`` are skipped unless ``redo`` is set.
    """
    with open(config_path) as f:
        config = json.load(f)

    video_dir = config.get("video_dir", os.path.dirname(os.path.abspath(config_path)))
    grid = config.get("grid", "4x4")
    videos = config.get("videos", [])
    if only:
        only = {os.path.splitext(n)[0] for n in only}
        videos = [v for v in videos if os.path.splitext(v["file"])[0] in only]

    previous = None
    todo = [v for v in videos if redo or not v.get("annotated")]
    print(f"{len(todo)} video(s) to annotate ({len(videos) - len(todo)} already done).")

    for idx, v in enumerate(todo):
        name = os.path.splitext(v["file"])[0]
        video_path = os.path.join(video_dir, v["file"])
        if not os.path.exists(video_path):
            print(f"  SKIP {name} (not found)")
            continue

        img = load_frame(video_path, os.path.join(video_dir, "samples", f"{name}.png"))
        picker = ArenaPicker(img, f"[{idx+1}/{len(todo)}] {v['file']}", grid=grid, previous=previous)
        action, polygon, boundary = picker.run()

        if action == "quit":
            print("Quit.")
            break
        if action != "save":
            print(f"  {name}: skipped")
            continue

        polygon, boundary = order_corners(polygon), order_corners(boundary)
        v["polygon"] = _fmt(polygon)
        v["boundary"] = _fmt(boundary)
        v["annotated"] = True
        previous = {"polygon": polygon, "boundary": boundary}
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  {name}: polygon={v['polygon']}  boundary={v['boundary']}")

    done = sum(1 for v in config.get("videos", []) if v.get("annotated"))
    print(f"\n{done}/{len(config.get('videos', []))} videos annotated in {config_path}")


def review_config(config_path, only=None, all_videos=False):
    """Step through annotated videos showing the saved corners; Enter marks a video verified.

    Clicking moves the nearest corner, 'r' clears for a full re-click. Videos already
    marked ``"verified": true`` are skipped unless ``all_videos`` is set.
    """
    with open(config_path) as f:
        config = json.load(f)

    video_dir = config.get("video_dir", os.path.dirname(os.path.abspath(config_path)))
    grid = config.get("grid", "4x4")
    videos = [v for v in config.get("videos", []) if v.get("annotated")]
    if only:
        only = {os.path.splitext(n)[0] for n in only}
        videos = [v for v in videos if os.path.splitext(v["file"])[0] in only]
    todo = [v for v in videos if all_videos or not v.get("verified")]
    print(f"{len(todo)} video(s) to review ({len(videos) - len(todo)} already verified).")

    for idx, v in enumerate(todo):
        name = os.path.splitext(v["file"])[0]
        video_path = os.path.join(video_dir, v["file"])
        if not os.path.exists(video_path):
            print(f"  SKIP {name} (not found)")
            continue

        initial = {"polygon": _parse(v["polygon"]), "boundary": _parse(v["boundary"])}
        tag = "auto" if v.get("auto") else "manual"
        img = load_frame(video_path, os.path.join(video_dir, "samples", f"{name}.png"))
        picker = ArenaPicker(img, f"[{idx+1}/{len(todo)}] {v['file']} ({tag})", grid=grid, initial=initial)
        action, polygon, boundary = picker.run()

        if action == "quit":
            print("Quit.")
            break
        if action != "save":
            print(f"  {name}: skipped")
            continue

        polygon, boundary = order_corners(polygon), order_corners(boundary)
        new_poly, new_bnd = _fmt(polygon), _fmt(boundary)
        changed = (new_poly, new_bnd) != (v["polygon"], v["boundary"])
        v["polygon"], v["boundary"] = new_poly, new_bnd
        v["verified"] = True
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  {name}: {'CORRECTED' if changed else 'ok'}")

    verified = sum(1 for v in config.get("videos", []) if v.get("verified"))
    print(f"\n{verified}/{len(config.get('videos', []))} videos verified in {config_path}")
