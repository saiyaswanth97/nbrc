"""OFT Analysis: Open Field Test mouse tracking and behavioral analysis."""

from .tracking import track_video
from .analysis import compute_velocity, compute_activity, compute_grid_analysis
from .plotting import plot_velocity_summary, plot_velocity_histogram, plot_transitions, draw_grid
from .io import save_tracking_results, load_tracking_results, extract_frames, save_sample_frames
