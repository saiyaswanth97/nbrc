"""Plotting functions for OFT analysis."""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_velocity_summary(time_s, velocity_smooth, displacement, activity, out_path,
                          smooth_window=5):
    """Plot velocity, cumulative distance, and activity."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Velocity
    axes[0].plot(time_s, velocity_smooth, color="steelblue", linewidth=0.5)
    axes[0].set_ylabel("Velocity (px/s)")
    axes[0].set_title(f"Velocity over time (smoothed, window={smooth_window})")
    med = np.nanmedian(velocity_smooth)
    axes[0].axhline(med, color="red", linestyle="--", alpha=0.5, label=f"median={med:.0f}")
    axes[0].legend()

    # Cumulative distance
    cum_dist = np.cumsum(displacement)
    axes[1].plot(time_s, cum_dist, color="green", linewidth=1)
    axes[1].set_ylabel("Cumulative distance (px)")
    axes[1].set_title(f"Total distance: {cum_dist[-1]:.0f} px")

    # Activity
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


def plot_velocity_histogram(velocity_smooth, out_path):
    """Plot velocity distribution histogram."""
    valid = velocity_smooth[~np.isnan(velocity_smooth)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(np.median(valid), color="red", linestyle="--", label=f"median={np.median(valid):.0f}")
    ax.axvline(np.mean(valid), color="orange", linestyle="--", label=f"mean={np.mean(valid):.0f}")
    ax.set_xlabel("Velocity (px/s)")
    ax.set_ylabel("Count")
    ax.set_title("Velocity distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_transitions(time_s, grid_result, fps, out_path, grid_rows=4, grid_cols=4):
    """Plot grid cell transitions, inner/outer zone, and wall hugging."""
    cell_ids = grid_result["cell_ids"]
    is_inner = grid_result["is_inner"]
    is_hugging = grid_result["is_hugging"]
    transitions = grid_result["transitions"]
    m = grid_result["metrics"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Grid cell over time
    axes[0].plot(time_s, cell_ids, color="purple", linewidth=0.3, alpha=0.5)
    trans_times = time_s.iloc[transitions].values if len(transitions) > 0 else []
    for t in trans_times:
        axes[0].axvline(t, color="orange", alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("Grid cell")
    axes[0].set_yticks(range(1, grid_rows * grid_cols + 1))
    axes[0].set_ylim(0.5, grid_rows * grid_cols + 0.5)
    axes[0].set_title(
        f"Grid cell over time ({m['total_transitions']} cell transitions, "
        f"{m['middle_zone_crossings']} middle zone crossings)"
    )

    # Inner/outer zone
    axes[1].fill_between(time_s, 0, 1, where=is_inner, color="blue", alpha=0.5, label="Inner")
    axes[1].fill_between(time_s, 0, 1, where=~is_inner, color="gray", alpha=0.3, label="Outer")
    axes[1].set_ylabel("Zone")
    axes[1].set_yticks([])
    axes[1].set_title(
        f"Inner: {m['inner_pct']:.1f}% ({m['inner_time_s']:.0f}s, {m['inner_entries']} entries) | "
        f"Outer: {m['outer_pct']:.1f}% ({m['outer_time_s']:.0f}s, {m['outer_entries']} entries)"
    )
    axes[1].legend()

    # Wall hugging
    wt = int(m["wall_threshold"] * 100)
    axes[2].fill_between(time_s, 0, 1, where=is_hugging, color="red", alpha=0.5,
                         label=f"Wall hugging (<{wt}%)")
    axes[2].fill_between(time_s, 0, 1, where=~is_hugging, color="green", alpha=0.3,
                         label="Away from wall")
    axes[2].set_ylabel("Wall")
    axes[2].set_yticks([])
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title(f"Wall hugging: {m['wall_hugging_pct']:.1f}% | Away: {100-m['wall_hugging_pct']:.1f}%")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_trajectory(bg_image, boundary, centroids_df, out_path,
                    grid_rows=4, grid_cols=4, smooth=5):
    """Plot 2D trajectory scatter on arena image with grid overlay.

    Args:
        bg_image: background image (full frame, BGR)
        boundary: [[x,y], ...] 4 corners in original image coords
        centroids_df: DataFrame with x, y, detected columns
        out_path: output image path
        grid_rows, grid_cols: grid dimensions
        smooth: rolling window for smoothing centroids
    """
    import pandas as pd

    # Draw grid on background
    bg = bg_image.copy()
    draw_grid(bg, boundary, grid_rows, grid_cols, color=(0, 255, 0), thickness=2)
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    # Get detected centroids and smooth
    detected = centroids_df[centroids_df["detected"] == 1].copy()
    x = pd.Series(detected["x"].values, dtype=float).rolling(smooth, center=True, min_periods=1).mean().values
    y = pd.Series(detected["y"].values, dtype=float).rolling(smooth, center=True, min_periods=1).mean().values

    # Filter to points inside boundary polygon
    boundary_pts = np.array(boundary, dtype=np.int32)
    inside = np.array([
        cv2.pointPolygonTest(boundary_pts.reshape((-1, 1, 2)), (float(xi), float(yi)), False) >= 0
        for xi, yi in zip(x, y)
    ])
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


def plot_trajectory_clean(boundary, centroids_df, out_path,
                          grid_rows=4, grid_cols=4, smooth=5):
    """Plot 2D trajectory on a clean grid (no background image).

    Args:
        boundary: [[x,y], ...] 4 corners in original image coords (TL, BL, BR, TR)
        centroids_df: DataFrame with x, y, detected columns
        out_path: output image path
        grid_rows, grid_cols: grid dimensions
        smooth: rolling window for smoothing centroids
    """
    import pandas as pd

    tl, bl, br, tr = [np.array(p, dtype=float) for p in boundary]

    def interp(p1, p2, n):
        return [p1 + (p2 - p1) * i / n for i in range(n + 1)]

    # Get detected centroids and smooth
    detected = centroids_df[centroids_df["detected"] == 1].copy()
    x = pd.Series(detected["x"].values, dtype=float).rolling(smooth, center=True, min_periods=1).mean().values
    y = pd.Series(detected["y"].values, dtype=float).rolling(smooth, center=True, min_periods=1).mean().values

    # Filter to points inside boundary
    boundary_pts = np.array(boundary, dtype=np.int32)
    inside = np.array([
        cv2.pointPolygonTest(boundary_pts.reshape((-1, 1, 2)), (float(xi), float(yi)), False) >= 0
        for xi, yi in zip(x, y)
    ])
    x, y = x[inside], y[inside]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("white")

    # Draw boundary
    poly = plt.Polygon([tl, bl, br, tr], fill=False, edgecolor="gray", linewidth=2)
    ax.add_patch(poly)

    # Draw grid lines
    left = interp(tl, bl, grid_rows)
    right = interp(tr, br, grid_rows)
    for i in range(1, grid_rows):
        ax.plot([left[i][0], right[i][0]], [left[i][1], right[i][1]], color="gray", linewidth=0.8)

    top = interp(tl, tr, grid_cols)
    bottom = interp(bl, br, grid_cols)
    for i in range(1, grid_cols):
        ax.plot([top[i][0], bottom[i][0]], [top[i][1], bottom[i][1]], color="gray", linewidth=0.8)

    # Plot trajectory
    ax.scatter(x, y, c="red", s=1, alpha=0.4)

    # Start marker (blue square) and end marker (blue circle)
    if len(x) > 0:
        ax.scatter(x[0], y[0], marker="s", c="blue", s=100, zorder=5, label="Start")
        ax.scatter(x[-1], y[-1], marker="o", c="blue", s=100, zorder=5, label="End")
        ax.legend(loc="upper right", fontsize=10)

    # Set limits with padding
    all_x = np.array([p[0] for p in [tl, bl, br, tr]])
    all_y = np.array([p[1] for p in [tl, bl, br, tr]])
    pad = 30
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.max() + pad, all_y.min() - pad)
    ax.set_aspect("equal")
    ax.set_title(f"Mouse trajectory ({len(x)} points)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_grid(image, boundary, grid_rows=4, grid_cols=4, color=(0, 255, 0), thickness=1):
    """Draw arena boundary and grid on image."""
    tl, bl, br, tr = boundary

    def interp(p1, p2, n):
        return [
            (int(p1[0] + (p2[0] - p1[0]) * i / n), int(p1[1] + (p2[1] - p1[1]) * i / n))
            for i in range(n + 1)
        ]

    pts = np.array([tl, bl, br, tr], dtype=np.int32)
    cv2.polylines(image, [pts.reshape((-1, 1, 2))], True, color, thickness + 1)

    left = interp(tl, bl, grid_rows)
    right = interp(tr, br, grid_rows)
    for i in range(1, grid_rows):
        cv2.line(image, left[i], right[i], color, thickness)

    top = interp(tl, tr, grid_cols)
    bottom = interp(bl, br, grid_cols)
    for i in range(1, grid_cols):
        cv2.line(image, top[i], bottom[i], color, thickness)

    # Label cells
    for r in range(grid_rows):
        for c in range(grid_cols):
            t = interp(top[c], bottom[c], grid_rows)
            b = interp(top[c+1], bottom[c+1], grid_rows)
            cx = (t[r][0] + t[r+1][0] + b[r][0] + b[r+1][0]) // 4
            cy = (t[r][1] + t[r+1][1] + b[r][1] + b[r+1][1]) // 4
            cv2.putText(image, str(r * grid_cols + c + 1), (cx - 8, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image
