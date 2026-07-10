"""Plotting functions for Barnes maze analysis."""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .geometry import quadrant_boundaries

# Distinct colors per quadrant (BGR for cv2, RGB tuples for matplotlib handled inline)
_QUAD_COLORS_BGR = [(80, 175, 76), (60, 76, 231), (219, 152, 52), (34, 126, 230)]


def plot_velocity_summary(time_s, velocity_smooth, displacement, activity, out_path,
                          smooth_window=5, unit="px"):
    """Plot velocity, cumulative distance, and activity."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(time_s, velocity_smooth, color="steelblue", linewidth=0.5)
    axes[0].set_ylabel(f"Velocity ({unit}/s)")
    axes[0].set_title(f"Velocity over time (smoothed, window={smooth_window})")
    med = np.nanmedian(velocity_smooth)
    axes[0].axhline(med, color="red", linestyle="--", alpha=0.5, label=f"median={med:.0f}")
    axes[0].legend()

    cum_dist = np.cumsum(displacement)
    axes[1].plot(time_s, cum_dist, color="green", linewidth=1)
    axes[1].set_ylabel(f"Cumulative distance ({unit})")
    axes[1].set_title(f"Total distance: {cum_dist[-1]:.0f} {unit}")

    active = activity["active_mask"]
    a = activity
    axes[2].fill_between(time_s, 0, 1, where=active, color="green", alpha=0.5, label="Moving")
    axes[2].fill_between(time_s, 0, 1, where=~active, color="red", alpha=0.3, label="Still")
    axes[2].set_ylabel("Activity")
    axes[2].set_title(
        f"Moving: {a['move_time_s']:.0f}s ({a['move_bouts']} bouts, avg {a['avg_move_bout_s']:.1f}s) | "
        f"Rest: {a['rest_time_s']:.0f}s ({a['rest_bouts']} bouts, avg {a['avg_rest_bout_s']:.1f}s)"
    )
    axes[2].set_yticks([])
    axes[2].legend()

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_velocity_histogram(velocity_smooth, out_path, unit="px"):
    """Plot velocity distribution histogram."""
    valid = velocity_smooth[~np.isnan(velocity_smooth)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(np.median(valid), color="red", linestyle="--", label=f"median={np.median(valid):.0f}")
    ax.axvline(np.mean(valid), color="orange", linestyle="--", label=f"mean={np.mean(valid):.0f}")
    ax.set_xlabel(f"Velocity ({unit}/s)")
    ax.set_ylabel("Count")
    ax.set_title("Velocity distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_quadrants(time_s, quad_result, fps, out_path, n_quadrants=4, target_quadrant=None):
    """Plot quadrant over time, target-vs-other zone, and periphery vs center."""
    quad_ids = quad_result["quad_ids"]
    is_periphery = quad_result["is_periphery"]
    transitions = quad_result["transitions"]
    m = quad_result["metrics"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Quadrant over time
    masked = np.where(quad_ids == 0, np.nan, quad_ids)
    axes[0].plot(time_s, masked, color="purple", linewidth=0.4, alpha=0.6)
    trans_times = time_s.iloc[transitions].values if len(transitions) > 0 else []
    for t in trans_times:
        axes[0].axvline(t, color="orange", alpha=0.2, linewidth=0.5)
    axes[0].set_ylabel("Quadrant")
    axes[0].set_yticks(range(1, n_quadrants + 1))
    axes[0].set_ylim(0.5, n_quadrants + 0.5)
    occ = m["quadrant_occupancy"]
    occ_str = " | ".join(f"Q{q}: {occ[str(q)]['pct']:.0f}%" for q in range(1, n_quadrants + 1))
    axes[0].set_title(f"Quadrant over time ({m['total_transitions']} transitions)   {occ_str}")

    # Target vs other quadrants
    if target_quadrant is not None:
        in_target = quad_ids == target_quadrant
        axes[1].fill_between(time_s, 0, 1, where=in_target, color="green", alpha=0.6, label=f"Target (Q{target_quadrant})")
        axes[1].fill_between(time_s, 0, 1, where=(quad_ids != 0) & ~in_target, color="gray", alpha=0.3, label="Other quadrants")
        lat = m.get("latency_to_target_s")
        lat_str = f"{lat:.1f}s" if lat is not None else "n/a"
        axes[1].set_title(
            f"Target Q{target_quadrant}: {m.get('target_pct', 0):.1f}% "
            f"({m.get('target_time_s', 0):.0f}s, {m.get('target_entries', 0)} entries, latency {lat_str})"
        )
    else:
        for q in range(1, n_quadrants + 1):
            axes[1].fill_between(time_s, 0, 1, where=quad_ids == q, alpha=0.5, label=f"Q{q}")
        axes[1].set_title("Quadrant occupancy (no target designated)")
    axes[1].set_ylabel("Quadrant zone")
    axes[1].set_yticks([])
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)

    # Periphery vs center
    pt = int(m["periphery_threshold"] * 100)
    axes[2].fill_between(time_s, 0, 1, where=is_periphery, color="red", alpha=0.5, label=f"Periphery (>{pt}% r)")
    axes[2].fill_between(time_s, 0, 1, where=~is_periphery, color="blue", alpha=0.3, label="Center")
    axes[2].set_ylabel("Radial")
    axes[2].set_yticks([])
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title(f"Periphery (thigmotaxis): {m['periphery_pct']:.1f}% | Center: {m['center_pct']:.1f}%")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_trajectory(bg_image, center, radius, edge_points, centroids_df, out_path,
                    target_quadrant=None, smooth=5):
    """Plot 2D trajectory scatter on arena image with circle + quadrant overlay."""
    import pandas as pd

    bg = bg_image.copy()
    draw_circle_quadrants(bg, center, radius, edge_points, target_quadrant=target_quadrant,
                          color=(0, 255, 0), thickness=2)
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    detected = centroids_df[centroids_df["detected"] == 1].copy()
    x = pd.Series(detected["x"].values, dtype=float).rolling(smooth, center=True, min_periods=1).mean().values
    y = pd.Series(detected["y"].values, dtype=float).rolling(smooth, center=True, min_periods=1).mean().values

    # Filter to points inside the circle
    cx, cy = center
    inside = (np.hypot(x - cx, y - cy) <= radius)
    x, y = x[inside], y[inside]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bg_rgb, aspect="equal")
    ax.scatter(x, y, c="red", s=1, alpha=0.4)
    ax.set_xlim(0, bg_image.shape[1])
    ax.set_ylim(bg_image.shape[0], 0)
    ax.set_title(f"Mouse trajectory ({len(x)} points, smoothed w={smooth})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_trajectory_clean(center, radius, edge_points, centroids_df, out_path,
                          target_quadrant=None, smooth=5):
    """Plot 2D trajectory on a clean circle (no background image)."""
    import pandas as pd

    cx, cy = center
    boundaries = quadrant_boundaries(center, edge_points)

    detected = centroids_df[centroids_df["detected"] == 1].copy()
    x = pd.Series(detected["x"].values, dtype=float).rolling(smooth, center=True, min_periods=1).mean().values
    y = pd.Series(detected["y"].values, dtype=float).rolling(smooth, center=True, min_periods=1).mean().values

    inside = (np.hypot(x - cx, y - cy) <= radius)
    x, y = x[inside], y[inside]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("white")

    # Circle
    circ = plt.Circle((cx, cy), radius, fill=False, edgecolor="gray", linewidth=2)
    ax.add_patch(circ)

    # Quadrant dividing lines (center -> each boundary angle on the circle)
    for ang in boundaries:
        ex = cx + radius * np.cos(ang)
        ey = cy + radius * np.sin(ang)
        ax.plot([cx, ex], [cy, ey], color="gray", linewidth=0.8)

    # Quadrant labels at sector midpoints
    n = len(boundaries)
    for i in range(n):
        a0 = boundaries[i]
        a1 = boundaries[(i + 1) % n]
        if a1 < a0:
            a1 += 2 * np.pi
        mid = (a0 + a1) / 2
        lx = cx + 0.6 * radius * np.cos(mid)
        ly = cy + 0.6 * radius * np.sin(mid)
        is_tgt = (target_quadrant == i + 1)
        ax.text(lx, ly, f"Q{i+1}", ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=("green" if is_tgt else "gray"))

    ax.scatter(x, y, c="red", s=1, alpha=0.4)
    if len(x) > 0:
        ax.scatter(x[0], y[0], marker="s", c="blue", s=100, zorder=5, label="Start")
        ax.scatter(x[-1], y[-1], marker="o", c="blue", s=100, zorder=5, label="End")
        ax.legend(loc="upper right", fontsize=10)

    pad = 30
    ax.set_xlim(cx - radius - pad, cx + radius + pad)
    ax.set_ylim(cy + radius + pad, cy - radius - pad)
    ax.set_aspect("equal")
    ax.set_title(f"Mouse trajectory ({len(x)} points)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_holes(image, holes, escape_idx=None, color=(0, 255, 255), radius=38, label=True):
    """Draw perimeter holes; highlight the escape hole in magenta."""
    for i, (x, y) in enumerate(holes):
        if i == escape_idx:
            continue
        cv2.circle(image, (int(x), int(y)), radius, color, 2)
        if label:
            cv2.putText(image, str(i), (int(x) - 10, int(y) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if escape_idx is not None:
        ex, ey = holes[escape_idx]
        cv2.circle(image, (int(ex), int(ey)), radius + 4, (255, 0, 255), 4)
        cv2.putText(image, "ESCAPE", (int(ex) - 70, int(ey) - radius - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    return image


def draw_circle_quadrants(image, center, radius, edge_points, target_quadrant=None,
                          color=(0, 255, 0), thickness=2):
    """Draw the arena circle, quadrant dividing lines, edge points, and labels."""
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = int(round(radius))

    cv2.circle(image, (cx, cy), r, color, thickness)
    cv2.circle(image, (cx, cy), 4, color, -1)

    boundaries = quadrant_boundaries(center, edge_points)

    # Dividing lines
    for ang in boundaries:
        ex = int(round(center[0] + radius * np.cos(ang)))
        ey = int(round(center[1] + radius * np.sin(ang)))
        cv2.line(image, (cx, cy), (ex, ey), color, thickness)

    # Edge points
    for (px, py) in edge_points:
        cv2.circle(image, (int(round(px)), int(round(py))), 8, (0, 0, 255), -1)

    # Quadrant labels at sector midpoints
    n = len(boundaries)
    for i in range(n):
        a0 = boundaries[i]
        a1 = boundaries[(i + 1) % n]
        if a1 < a0:
            a1 += 2 * np.pi
        mid = (a0 + a1) / 2
        lx = int(round(center[0] + 0.6 * radius * np.cos(mid)))
        ly = int(round(center[1] + 0.6 * radius * np.sin(mid)))
        label_color = (0, 200, 0) if target_quadrant == i + 1 else color
        txt = f"Q{i+1}" + ("*" if target_quadrant == i + 1 else "")
        cv2.putText(image, txt, (lx - 15, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
    return image
