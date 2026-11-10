"""Y-maze plots: trajectory coloured by arm state, and an arm-state timeline."""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .geometry import ARMS, draw_maze

STATE_COLORS = {"A": "#d1495b", "B": "#2e86ab", "C": "#e0a100", "center": "#7a7a7a"}


def plot_trajectory(bg_bgr, maze, x, y, states, out_path, entry_frac, exit_frac, title=""):
    img = draw_maze(bg_bgr.copy(), maze, entry_frac, exit_frac)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for s, col in STATE_COLORS.items():
        sel = np.array([st == s for st in states])
        ax.scatter(x[sel], y[sel], s=1.5, c=col, label=s if s != "center" else "centre")
    c = np.asarray(maze["center"])
    tips = np.array([maze["tips"][a] for a in ARMS])
    pad = maze["width"] * 1.5
    ax.set_xlim(min(tips[:, 0].min(), c[0]) - pad, max(tips[:, 0].max(), c[0]) + pad)
    ax.set_ylim(max(tips[:, 1].max(), c[1]) + pad, min(tips[:, 1].min(), c[1]) - pad)
    ax.legend(loc="upper right", markerscale=6, fontsize=9)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_timeline(states, entries, fps, out_path, novel_arm=None, title=""):
    """Ethogram: which zone the animal is in over time, entries numbered."""
    fig, ax = plt.subplots(figsize=(12, 2.6))
    rows = {"A": 3, "B": 2, "C": 1, "center": 0}
    t = np.arange(len(states)) / fps
    for s, r in rows.items():
        sel = np.array([st == s for st in states])
        ax.fill_between(t, r - 0.4, r + 0.4, where=sel, color=STATE_COLORS[s], step="mid", linewidth=0)
    for k, e in enumerate(entries):
        ax.text(e["time_s"], rows[e["arm"]] + 0.48, str(k + 1), fontsize=6, ha="left", va="bottom")
    labels = [f"{a}{' (novel)' if a == novel_arm else ''}" for a in ("A", "B", "C")]
    ax.set_yticks([3, 2, 1, 0], labels + ["centre"])
    ax.set_xlabel("time (s)")
    ax.set_ylim(-0.6, 3.9)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
