"""Barnes Maze Analysis: circular-arena mouse tracking and behavioral analysis."""

from .tracking import track_video
from .analysis import compute_velocity, compute_activity, compute_quadrant_analysis
from .geometry import (
    fit_circle,
    quadrant_boundaries,
    point_quadrant,
    normalized_radius,
    make_circle_mask,
    circle_from_edge_points,
)
from .plotting import (
    plot_velocity_summary,
    plot_velocity_histogram,
    plot_quadrants,
    plot_trajectory,
    plot_trajectory_clean,
    draw_circle_quadrants,
    draw_holes,
)
from .holes import detect_holes, identify_escape_hole
from .io import save_tracking_results, load_tracking_results, extract_frames, save_sample_frames
