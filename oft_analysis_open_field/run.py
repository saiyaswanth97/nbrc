"""Run full OFT analysis pipeline.

Usage:
    # Track only:
    python -m oft_analysis.run track <video> --polygon 191,60,200,1047,1164,1047,1190,87

    # Analyze existing tracking:
    python -m oft_analysis.run analyze <optflow_dir> --boundary 170,110,173,850,899,847,912,127

    # Full pipeline:
    python -m oft_analysis.run full <video> --polygon 191,60,200,1047,1164,1047,1190,87 \
        --boundary 170,110,173,850,899,847,912,127 --smooth 30

    # Visualize grid:
    python -m oft_analysis.run grid <image_or_dir> --boundary 170,110,173,850,899,847,912,127
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import glob
import cv2

from .tracking import track_video
from .analysis import compute_velocity, compute_activity, compute_grid_analysis
from .plotting import plot_velocity_summary, plot_velocity_histogram, plot_transitions, plot_trajectory, plot_trajectory_clean, draw_grid
from .io import save_tracking_results, load_tracking_results, save_sample_frames


def parse_coords(s):
    """Parse comma-separated coordinate string into list of [x,y] pairs."""
    coords = [int(x) for x in s.split(",")]
    return [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]


def cmd_track(args):
    polygon = parse_coords(args.polygon) if args.polygon else None
    crop = [int(x) for x in args.crop.split(",")] if args.crop else None
    start_s = getattr(args, 'start', None)
    end_s = getattr(args, 'end', None)

    results = track_video(args.video, crop=crop, polygon=polygon, pad=args.pad,
                          start_s=start_s, end_s=end_s)

    video_dir = os.path.dirname(os.path.abspath(args.video))
    name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join(video_dir, "optflow", name)

    save_tracking_results(results, out_dir, save_frames=True, every=args.every)
    print(f"\nOutput: {out_dir}")


def cmd_analyze(args):
    data, df = load_tracking_results(args.optflow_dir)
    fps = data["fps"]
    out_dir = os.path.join(args.optflow_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    x = df["x"].values.astype(float)
    y = df["y"].values.astype(float)

    # px -> mm scale based on boundary edge length (arena = 1000x1000 mm)
    ARENA_EDGE_MM = 1000.0
    px_per_mm = None
    if args.boundary:
        bpts = np.array(parse_coords(args.boundary), dtype=float)
        edges = np.linalg.norm(np.diff(np.vstack([bpts, bpts[:1]]), axis=0), axis=1)
        mean_edge_px = float(edges.mean())
        px_per_mm = mean_edge_px / ARENA_EDGE_MM
    scale = (1.0 / px_per_mm) if px_per_mm else 1.0
    unit = "mm" if px_per_mm else "px"

    velocity, displacement = compute_velocity(x, y, fps)
    velocity *= scale
    displacement = displacement * scale
    velocity_smooth = pd.Series(velocity).rolling(
        window=args.smooth, center=True, min_periods=1
    ).mean().values
    time_s = df["frame"] / fps

    activity = compute_activity(velocity_smooth, fps, threshold=args.activity_threshold)

    plot_velocity_summary(time_s, velocity_smooth, displacement, activity,
                         os.path.join(out_dir, "velocity.png"), args.smooth, unit=unit)
    print(f"Saved: {os.path.join(out_dir, 'velocity.png')}")

    plot_velocity_histogram(velocity_smooth, os.path.join(out_dir, "velocity_hist.png"), unit=unit)
    print(f"Saved: {os.path.join(out_dir, 'velocity_hist.png')}")

    # Grid analysis
    if args.boundary:
        boundary = parse_coords(args.boundary)
        grid_rows, grid_cols = [int(x) for x in args.grid.split("x")]

        grid_result = compute_grid_analysis(
            df, boundary,
            grid_rows=grid_rows, grid_cols=grid_cols,
            wall_threshold=args.wall_threshold, fps=fps,
        )

        plot_transitions(time_s, grid_result, fps,
                        os.path.join(out_dir, "transitions.png"),
                        grid_rows, grid_cols)
        print(f"Saved: {os.path.join(out_dir, 'transitions.png')}")

        # Trajectory plot
        sample_path = os.path.join(args.optflow_dir, "samples", "original.png")
        if os.path.exists(sample_path):
            bg_img = cv2.imread(sample_path)
            plot_trajectory(bg_img, boundary, df,
                           os.path.join(out_dir, "trajectory.png"),
                           grid_rows, grid_cols, smooth=args.smooth)
            print(f"Saved: {os.path.join(out_dir, 'trajectory.png')}")

        plot_trajectory_clean(boundary, df,
                             os.path.join(out_dir, "trajectory_clean.png"),
                             grid_rows, grid_cols, smooth=args.smooth)
        print(f"Saved: {os.path.join(out_dir, 'trajectory_clean.png')}")

        # Save metrics
        metrics = {**grid_result["metrics"], **activity}
        del metrics["active_mask"]
        with open(os.path.join(out_dir, "transitions.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved: {os.path.join(out_dir, 'transitions.json')}")

    # Save stats
    valid_vel = velocity_smooth[~np.isnan(velocity_smooth)]
    stats = {
        "total_frames": len(df),
        "duration_s": float(time_s.iloc[-1]),
        "fps": fps,
        "unit": unit,
        f"total_distance_{unit}": float(np.sum(displacement)),
        f"mean_velocity_{unit}_s": float(np.mean(valid_vel)),
        f"median_velocity_{unit}_s": float(np.median(valid_vel)),
        f"max_velocity_{unit}_s": float(np.max(valid_vel)),
        "time_moving_pct": float(np.mean(activity["active_mask"]) * 100),
        "time_still_pct": float((1 - np.mean(activity["active_mask"])) * 100),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {os.path.join(out_dir, 'stats.json')}")

    print(f"\nDuration: {stats['duration_s']:.1f}s")
    print(f"Total distance: {stats[f'total_distance_{unit}']:.0f} {unit}")
    print(f"Mean velocity: {stats[f'mean_velocity_{unit}_s']:.0f} {unit}/s")
    print(f"Moving: {stats['time_moving_pct']:.1f}% | Still: {stats['time_still_pct']:.1f}%")


def cmd_full(args):
    """Run track + analyze."""
    cmd_track(args)
    video_dir = os.path.dirname(os.path.abspath(args.video))
    name = os.path.splitext(os.path.basename(args.video))[0]
    args.optflow_dir = os.path.join(video_dir, "optflow", name)

    # Save sample frames
    data, _ = load_tracking_results(args.optflow_dir)
    boundary = parse_coords(args.boundary) if args.boundary else None
    grid_rows, grid_cols = [int(x) for x in args.grid.split("x")]
    save_sample_frames(args.video, args.optflow_dir, data, boundary, grid_rows, grid_cols)

    cmd_analyze(args)


def cmd_grid(args):
    """Draw grid on images."""
    boundary = parse_coords(args.boundary)
    grid_rows, grid_cols = [int(x) for x in args.grid.split("x")]

    if os.path.isdir(args.input):
        images = sorted(glob.glob(os.path.join(args.input, "*.png")))
        images += sorted(glob.glob(os.path.join(args.input, "*.jpg")))
    else:
        images = [args.input]

    out_dir = args.output or (args.input.rstrip("/") + "_grid" if os.path.isdir(args.input)
                              else os.path.join(os.path.dirname(args.input), "grid_viz"))
    os.makedirs(out_dir, exist_ok=True)

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        draw_grid(img, boundary, grid_rows, grid_cols)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), img)

    print(f"Saved {len(images)} images to {out_dir}")


def cmd_batch(args):
    """Run full pipeline on all videos defined in a config file."""
    import time

    with open(args.config) as f:
        config = json.load(f)

    video_dir = config.get("video_dir", os.path.dirname(os.path.abspath(args.config)))
    videos = config.get("videos", [])
    # Global settings (per-video can override)
    globals_ = {k: v for k, v in config.items() if k not in ("video_dir", "videos")}

    if not videos:
        print("No videos in config.")
        return

    start_all = time.time()
    for idx, video_cfg in enumerate(videos):
        cfg = {**globals_, **video_cfg}
        name = os.path.splitext(cfg["file"])[0]
        video_path = os.path.join(video_dir, cfg["file"])

        if not os.path.exists(video_path):
            print(f"\n===== [{idx+1}/{len(videos)}] {name} — SKIPPED (not found: {video_path}) =====")
            continue

        print(f"\n===== [{idx+1}/{len(videos)}] {name} =====")

        ns = argparse.Namespace(
            video=video_path,
            polygon=cfg.get("polygon"),
            crop=cfg.get("crop"),
            every=cfg.get("every", 200),
            pad=cfg.get("pad", 60),
            start=cfg.get("start"),
            end=cfg.get("end"),
            smooth=cfg.get("smooth", 30),
            boundary=cfg.get("boundary"),
            grid=cfg.get("grid", "4x4"),
            wall_threshold=cfg.get("wall_threshold", 0.05),
            activity_threshold=cfg.get("activity_threshold", 10.0),
        )

        t0 = time.time()
        cmd_full(ns)
        elapsed = time.time() - t0
        total_elapsed = time.time() - start_all
        remaining = (total_elapsed / (idx + 1)) * (len(videos) - idx - 1)
        print(f"  Done in {elapsed:.0f}s | Elapsed: {total_elapsed/60:.1f}min | Remaining: ~{remaining/60:.0f}min")

    print(f"\n===== ALL DONE in {(time.time()-start_all)/60:.1f} min =====")


def cmd_sample(args):
    """Extract a sample frame from the middle of each video for manual annotation."""
    with open(args.config) as f:
        config = json.load(f)

    video_dir = config.get("video_dir", os.path.dirname(os.path.abspath(args.config)))
    videos = config.get("videos", [])
    out_dir = os.path.join(video_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)

    for cfg in videos:
        video_path = os.path.join(video_dir, cfg["file"])
        name = os.path.splitext(cfg["file"])[0]
        if not os.path.exists(video_path):
            print(f"  SKIP {name} (not found)")
            continue

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = total // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()

        if ret:
            out_path = os.path.join(out_dir, f"{name}.png")
            cv2.imwrite(out_path, frame)
            print(f"  {name}: frame {mid}/{total} -> {out_path}")

    print(f"\nSaved to {out_dir}/")
    print("Use these images to identify polygon (arena walls) and boundary (floor grid) coordinates.")


def cmd_init(args):
    """Generate a template config file."""
    video_dir = os.path.abspath(args.video_dir)
    videos_found = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

    config = {
        "video_dir": video_dir,
        "grid": "4x4",
        "smooth": 30,
        "activity_threshold": 10.0,
        "wall_threshold": 0.05,
        "videos": [],
    }
    for f in videos_found:
        config["videos"].append({
            "file": f,
            "start": None,
            "end": None,
            "polygon": "191,60,200,1047,1164,1047,1190,87",
            "boundary": "170,110,173,850,899,847,912,127",
        })

    out_path = args.output or os.path.join(video_dir, "oft_config.json")
    with open(out_path, "w") as fp:
        json.dump(config, fp, indent=2)
    print(f"Config written to {out_path} with {len(videos_found)} videos.")
    print("Edit start, end, polygon per video as needed, then run:")
    print(f"  python -m oft_analysis batch {out_path}")


def main():
    parser = argparse.ArgumentParser(description="OFT Analysis Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # Track
    p_track = sub.add_parser("track", help="Run optical flow tracking")
    p_track.add_argument("video", help="Path to video")
    p_track.add_argument("--polygon", help="Polygon ROI: x1,y1,x2,y2,...")
    p_track.add_argument("--crop", help="Rect crop: x1,x2,y1,y2")
    p_track.add_argument("--every", type=int, default=200, help="Save frame every N (default: 200)")
    p_track.add_argument("--pad", type=int, default=60, help="Bbox padding (default: 60)")
    p_track.add_argument("--start", type=float, help="Start time in seconds")
    p_track.add_argument("--end", type=float, help="End time in seconds (negative = from end)")

    # Analyze
    p_analyze = sub.add_parser("analyze", help="Analyze tracking results")
    p_analyze.add_argument("optflow_dir", help="Optflow output directory")
    p_analyze.add_argument("--smooth", type=int, default=30, help="Smoothing window (default: 30)")
    p_analyze.add_argument("--boundary", help="Grid boundary: x1,y1,x2,y2,... (cropped coords)")
    p_analyze.add_argument("--grid", default="4x4", help="Grid ROWSxCOLS (default: 4x4)")
    p_analyze.add_argument("--wall-threshold", type=float, default=0.05, help="Wall hug threshold (default: 0.05)")
    p_analyze.add_argument("--activity-threshold", type=float, default=10.0, help="Velocity threshold for moving/rest in mm/s (default: 10.0)")

    # Full
    p_full = sub.add_parser("full", help="Track + analyze")
    p_full.add_argument("video", help="Path to video")
    p_full.add_argument("--polygon", help="Polygon ROI: x1,y1,x2,y2,...")
    p_full.add_argument("--crop", help="Rect crop: x1,x2,y1,y2")
    p_full.add_argument("--every", type=int, default=200)
    p_full.add_argument("--pad", type=int, default=60)
    p_full.add_argument("--start", type=float, help="Start time in seconds")
    p_full.add_argument("--end", type=float, help="End time in seconds (negative = from end)")
    p_full.add_argument("--smooth", type=int, default=30)
    p_full.add_argument("--boundary", help="Grid boundary: x1,y1,x2,y2,...")
    p_full.add_argument("--grid", default="4x4")
    p_full.add_argument("--wall-threshold", type=float, default=0.05)
    p_full.add_argument("--activity-threshold", type=float, default=20.0)

    # Batch
    p_batch = sub.add_parser("batch", help="Run full pipeline on all videos from config file")
    p_batch.add_argument("config", help="Path to JSON config file")

    # Sample frames
    p_sample = sub.add_parser("sample", help="Extract middle frame from each video for manual annotation")
    p_sample.add_argument("config", help="Path to JSON config file")

    # Init config
    p_init = sub.add_parser("init", help="Generate template config for a video directory")
    p_init.add_argument("video_dir", help="Directory containing .mp4 files")
    p_init.add_argument("--output", help="Output config path (default: <video_dir>/oft_config.json)")

    # Grid viz
    p_grid = sub.add_parser("grid", help="Draw grid on images")
    p_grid.add_argument("input", help="Image or directory")
    p_grid.add_argument("--boundary", required=True, help="Boundary corners: x1,y1,x2,y2,...")
    p_grid.add_argument("--grid", default="4x4")
    p_grid.add_argument("--output", help="Output directory")

    args = parser.parse_args()
    cmds = {
        "track": cmd_track, "analyze": cmd_analyze, "full": cmd_full,
        "grid": cmd_grid, "batch": cmd_batch, "sample": cmd_sample, "init": cmd_init,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
