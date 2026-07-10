"""Barnes maze behavioral analysis: velocity, activity, quadrants, thigmotaxis.

Velocity and activity are shared with the OFT pipeline. The Barnes-specific part
is quadrant occupancy/transitions (with an optional target quadrant) and radial
(periphery vs center) analysis, since Barnes mice search along the perimeter
where the holes are.
"""

import numpy as np
import pandas as pd

from .geometry import point_quadrant, normalized_radius


def compute_velocity(x, y, fps, windows=(1, 2, 3, 5)):
    """Multi-frame velocity with IQR outlier rejection.

    Returns (velocity, displacement) arrays.
    """
    n = len(x)
    vel_estimates = np.full((len(windows), n), np.nan)

    for wi, w in enumerate(windows):
        dx = np.full(n, np.nan)
        dy = np.full(n, np.nan)
        dx[w:] = x[w:] - x[:-w]
        dy[w:] = y[w:] - y[:-w]
        vel_estimates[wi] = np.sqrt(dx**2 + dy**2) / w * fps

    velocity = np.nanmedian(vel_estimates, axis=0)
    velocity = np.nan_to_num(velocity, nan=0.0)

    # IQR outlier rejection
    nonzero = velocity[velocity > 0]
    if len(nonzero) > 0:
        q1, q3 = np.percentile(nonzero, [25, 75])
        upper_bound = q3 + 3 * (q3 - q1)
        velocity = np.clip(velocity, 0, upper_bound)
    else:
        upper_bound = np.inf

    dx1 = np.diff(x, prepend=x[0])
    dy1 = np.diff(y, prepend=y[0])
    displacement = np.sqrt(dx1**2 + dy1**2)
    displacement = np.nan_to_num(displacement, nan=0.0)
    disp_limit = upper_bound / fps if np.isfinite(upper_bound) else np.inf
    displacement = np.clip(displacement, 0, disp_limit)

    return velocity, displacement


def count_bouts(mask):
    """Count contiguous True bouts. Returns (n_bouts, list of durations in frames)."""
    bouts = []
    in_bout = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_bout:
            in_bout = True
            start = i
        elif not v and in_bout:
            in_bout = False
            bouts.append(i - start)
    if in_bout:
        bouts.append(len(mask) - start)
    return len(bouts), bouts


def compute_activity(velocity_smooth, fps, threshold=20.0):
    """Compute moving/rest bout statistics."""
    active = velocity_smooth > threshold

    n_move, move_durs = count_bouts(active)
    n_rest, rest_durs = count_bouts(~active)

    return {
        "threshold": float(threshold),
        "active_mask": active,
        "move_time_s": round(sum(move_durs) / fps, 1),
        "move_bouts": n_move,
        "avg_move_bout_s": round(sum(move_durs) / n_move / fps, 1) if n_move else 0,
        "rest_time_s": round(sum(rest_durs) / fps, 1),
        "rest_bouts": n_rest,
        "avg_rest_bout_s": round(sum(rest_durs) / n_rest / fps, 1) if n_rest else 0,
    }


def compute_quadrant_analysis(centroids_df, center, radius, boundaries,
                              target_quadrant=None, periphery_threshold=0.65,
                              n_quadrants=4, fps=30.0):
    """Compute quadrant occupancy, transitions, target metrics, thigmotaxis.

    Args:
        centroids_df: DataFrame with columns x, y, detected
        center: (cx, cy) fitted circle center
        radius: fitted circle radius (px)
        boundaries: sorted quadrant boundary angles (radians)
        target_quadrant: 1..n index of the quadrant holding the escape hole (or None)
        periphery_threshold: normalized-radius above which the mouse is "at the
            periphery" (where the holes are) — thigmotaxis proxy
        n_quadrants: number of quadrants (4)
        fps: frames per second

    Returns dict with per-frame arrays and summary metrics.
    """
    quads = []
    norm_r = []

    for _, row in centroids_df.iterrows():
        if pd.notna(row["x"]) and row.get("detected", 0):
            px, py = row["x"], row["y"]
            quads.append(point_quadrant(px, py, center, boundaries))
            norm_r.append(normalized_radius(px, py, center, radius))
        else:
            quads.append(0)
            norm_r.append(np.nan)

    quad_ids = np.array(quads)
    norm_r = np.array(norm_r)

    # Quadrant transitions (consecutive detected frames in different quadrants)
    transitions = []
    for i in range(1, len(quad_ids)):
        if quad_ids[i] != 0 and quad_ids[i-1] != 0 and quad_ids[i] != quad_ids[i-1]:
            transitions.append(i)

    # Per-quadrant occupancy + entry counts
    n_frames = len(quad_ids)
    duration_s = n_frames / fps
    quadrant_stats = {}
    for q in range(1, n_quadrants + 1):
        in_q = quad_ids == q
        n_entries, _ = count_bouts(in_q)
        frames = int(in_q.sum())
        quadrant_stats[str(q)] = {
            "frames": frames,
            "time_s": round(frames / fps, 1),
            "pct": round(100 * frames / n_frames, 1) if n_frames else 0,
            "entries": n_entries,
        }

    # Target quadrant metrics
    target = {}
    if target_quadrant is not None:
        in_target = quad_ids == target_quadrant
        first_idx = int(np.argmax(in_target)) if in_target.any() else None
        target = {
            "target_quadrant": int(target_quadrant),
            "target_time_s": round(int(in_target.sum()) / fps, 1),
            "target_pct": round(100 * int(in_target.sum()) / n_frames, 1) if n_frames else 0,
            "target_entries": count_bouts(in_target)[0],
            "latency_to_target_s": round(first_idx / fps, 1) if first_idx is not None else None,
        }

    # Radial / thigmotaxis: periphery = near the ring of holes
    valid = ~np.isnan(norm_r)
    is_periphery = norm_r >= periphery_threshold
    periphery_pct = 100 * np.nansum(is_periphery) / np.sum(valid) if np.sum(valid) else 0
    mean_norm_r = float(np.nanmean(norm_r)) if np.sum(valid) else 0.0

    return {
        "quad_ids": quad_ids,
        "norm_r": norm_r,
        "is_periphery": is_periphery & valid,
        "transitions": transitions,
        "metrics": {
            "total_transitions": len(transitions),
            "transitions_per_min": round(len(transitions) / (duration_s / 60), 1) if duration_s > 0 else 0,
            "quadrant_occupancy": quadrant_stats,
            **target,
            "periphery_threshold": periphery_threshold,
            "periphery_pct": round(periphery_pct, 1),
            "center_pct": round(100 - periphery_pct, 1),
            "mean_normalized_radius": round(mean_norm_r, 3),
        },
    }
