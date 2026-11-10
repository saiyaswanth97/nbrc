"""Review / fix the detected maze per video (matplotlib, tkagg).

Shows the maze outline on the video's background image. Enter = looks good,
click = move the nearest point (centre or an arm tip), +/- = arm width,
r = clear and click centre then the three tips, n = skip, q = quit.
"""

import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

from oft_analysis_open_field.gui import _ensure_interactive_backend
from .geometry import ARMS, label_arms, draw_maze


class MazePicker:
    def __init__(self, img_bgr, title, maze=None, default_width=100.0):
        _ensure_interactive_backend()
        self.img = img_bgr
        self.title = title
        self.width = maze["width"] if maze else default_width
        self.pts = ([maze["center"]] + [maze["tips"][a] for a in ARMS]) if maze else []
        self.action = None
        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._onclick)
        self.fig.canvas.mpl_connect("key_press_event", self._onkey)
        self._redraw()

    def maze(self):
        if len(self.pts) < 4:
            return None
        c = self.pts[0]
        return {"center": [float(c[0]), float(c[1])], "tips": label_arms(c, self.pts[1:4]),
                "width": float(self.width)}

    def _onclick(self, e):
        if e.inaxes != self.ax or e.xdata is None or e.button != 1:
            return
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return
        p = [float(e.xdata), float(e.ydata)]
        if len(self.pts) < 4:
            self.pts.append(p)
        else:
            i = int(np.argmin([np.hypot(q[0] - p[0], q[1] - p[1]) for q in self.pts]))
            self.pts[i] = p
        self._redraw()

    def _onkey(self, e):
        if e.key == "r":
            self.pts = []
        elif e.key in ("+", "="):
            self.width *= 1.05
        elif e.key == "-":
            self.width /= 1.05
        elif e.key in ("enter", "return") and len(self.pts) == 4:
            self.action = "save"
            plt.close(self.fig)
            return
        elif e.key in ("n", "q"):
            self.action = "skip" if e.key == "n" else "quit"
            plt.close(self.fig)
            return
        self._redraw()

    def _redraw(self):
        lims = (self.ax.get_xlim(), self.ax.get_ylim()) if self.ax.images else None
        self.ax.clear()
        img = self.img.copy()
        m = self.maze()
        if m:
            draw_maze(img, m)
        self.ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for i, p in enumerate(self.pts):
            self.ax.scatter(*p, c="red" if i == 0 else "magenta", s=40, zorder=5)
        self.ax.axis("off")
        if lims:
            self.ax.set_xlim(lims[0])
            self.ax.set_ylim(lims[1])
        if len(self.pts) == 0:
            msg = "Click the junction CENTRE"
        elif len(self.pts) < 4:
            msg = f"Click arm TIP {len(self.pts)}/3 (end of an arm, any order)"
        else:
            msg = "Enter=looks good   click=move nearest point   +/-=arm width   r=re-click"
        self.ax.set_title(f"{self.title}\n{msg}\nn=skip  q=quit", fontsize=11)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()
        return self.action, self.maze()


def review_config(config_path, only=None, all_videos=False):
    """Step through videos; Enter marks the maze verified, edits are saved."""
    from .run import video_background
    with open(config_path) as f:
        config = json.load(f)
    video_dir = config["video_dir"]
    videos = config["videos"]
    if only:
        only = {os.path.splitext(n)[0] for n in only}
        videos = [v for v in videos if os.path.splitext(v["file"])[0] in only]
    todo = [v for v in videos if all_videos or not v.get("verified")]
    widths = [v["maze"]["width"] for v in config["videos"] if v.get("maze")]
    default_w = float(np.median(widths)) if widths else 100.0
    print(f"{len(todo)} video(s) to review.")
    for idx, v in enumerate(todo):
        bg = video_background(os.path.join(video_dir, v["file"]))
        flag = f" — FLAGGED: {'; '.join(v['maze_problems'])}" if v.get("maze_problems") else ""
        picker = MazePicker(cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR),
                            f"[{idx+1}/{len(todo)}] {v['file']}{flag}", v.get("maze"), default_w)
        action, maze = picker.run()
        if action == "quit":
            break
        if action != "save":
            continue
        v["maze"], v["verified"], v["maze_problems"] = maze, True, []
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  {v['file']}: verified")
