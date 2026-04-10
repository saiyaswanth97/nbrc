"""OFT behavioral analysis: velocity, zones, transitions, wall hugging.

All analysis functions take centroids + metadata and return computed metrics.
"""

import cv2
import numpy as np
import pandas as pd


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

    # Single-frame displacement (for cumulative distance)
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
    """Compute moving/rest bout statistics.

    Args:
        velocity_smooth: smoothed velocity array (px/s)
        fps: frames per second
        threshold: absolute velocity threshold (px/s) below which mouse is "resting"
    """
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


def _get_perspective_matrix(boundary, dst_size):
    """Get perspective transform from boundary quadrilateral to rectangle."""
    tl, bl, br, tr = [np.array(p, dtype=np.float32) for p in boundary]
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    w, h = dst_size
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def point_to_grid_cell(px, py, M, grid_rows, grid_cols):
    """Map point to grid cell via precomputed perspective transform. Returns (row, col) or None."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, M)[0][0]
    col, row = int(mapped[0]), int(mapped[1])
    if 0 <= row < grid_rows and 0 <= col < grid_cols:
        return row, col
    return None


def compute_grid_analysis(centroids_df, boundary, grid_rows=4, grid_cols=4,
                          inner_cells=None, wall_threshold=0.05, fps=30.0):
    """Compute grid cell occupancy, transitions, inner/outer zones, wall hugging.

    Args:
        centroids_df: DataFrame with columns x, y, detected
        boundary: [[x,y], ...] 4 corners TL,BL,BR,TR in original image coords
        grid_rows, grid_cols: grid dimensions
        inner_cells: set of cell IDs considered "inner zone" (1-indexed)
        wall_threshold: fraction of arena width considered "near wall"
        fps: frames per second

    Returns dict with all grid/zone/wall metrics.
    """
    if inner_cells is None:
        inner_cells = {6, 7, 10, 11}

    M_grid = _get_perspective_matrix(boundary, (grid_cols, grid_rows))
    M_norm = _get_perspective_matrix(boundary, (1, 1))

    cells = []
    wall_dists = []

    for _, row in centroids_df.iterrows():
        if pd.notna(row["x"]) and row.get("detected", 0):
            px, py = row["x"], row["y"]

            cell = point_to_grid_cell(px, py, M_grid, grid_rows, grid_cols)
            cells.append(cell)

            pt = np.array([[[px, py]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(pt, M_norm)[0][0]
            d = min(mapped[0], 1 - mapped[0], mapped[1], 1 - mapped[1])
            wall_dists.append(max(0.0, min(0.5, float(d))))
        else:
            cells.append(None)
            wall_dists.append(np.nan)

    # Cell IDs (1-indexed, 0 = undetected)
    cell_ids = np.array([
        c[0] * grid_cols + c[1] + 1 if c is not None else 0
        for c in cells
    ])

    # Transitions
    transitions = []
    for i in range(1, len(cells)):
        if cells[i] is not None and cells[i-1] is not None and cells[i] != cells[i-1]:
            transitions.append(i)

    # Middle zone crossings (outer -> inner)
    middle_crossings = 0
    for i in range(1, len(cell_ids)):
        if cell_ids[i-1] not in inner_cells and cell_ids[i] in inner_cells:
            middle_crossings += 1

    # Inner/outer
    is_inner = np.array([cid in inner_cells for cid in cell_ids])
    inner_pct = 100 * is_inner.sum() / len(is_inner) if len(is_inner) else 0
    n_inner_entries, inner_bout_durs = count_bouts(is_inner)
    n_outer_entries, outer_bout_durs = count_bouts(~is_inner)

    # Wall hugging
    wall_dists = np.array(wall_dists)
    is_hugging = wall_dists < wall_threshold
    valid_mask = ~np.isnan(wall_dists)
    hugging_pct = 100 * np.nansum(is_hugging) / np.sum(valid_mask) if np.sum(valid_mask) else 0

    # Cell occupancy
    cell_occupancy = {}
    for cid in range(1, grid_rows * grid_cols + 1):
        count = int(np.sum(cell_ids == cid))
        cell_occupancy[str(cid)] = {
            "frames": count,
            "pct": round(100 * count / len(cell_ids), 1) if len(cell_ids) else 0,
        }

    duration_s = len(centroids_df) / fps

    return {
        "cell_ids": cell_ids,
        "is_inner": is_inner,
        "is_hugging": is_hugging & valid_mask,
        "wall_dists": wall_dists,
        "transitions": transitions,
        "metrics": {
            "total_transitions": len(transitions),
            "transitions_per_min": len(transitions) / (duration_s / 60) if duration_s > 0 else 0,
            "middle_zone_crossings": middle_crossings,
            "inner_entries": n_inner_entries,
            "inner_time_s": round(sum(inner_bout_durs) / fps, 1),
            "inner_pct": round(inner_pct, 1),
            "outer_entries": n_outer_entries,
            "outer_time_s": round(sum(outer_bout_durs) / fps, 1),
            "outer_pct": round(100 - inner_pct, 1),
            "wall_hugging_pct": round(hugging_pct, 1),
            "wall_threshold": wall_threshold,
            "cell_occupancy": cell_occupancy,
        },
    }
