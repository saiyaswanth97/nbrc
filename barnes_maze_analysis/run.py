"""Run full Barnes maze analysis pipeline.

Usage:
    # Extract sample frames (to pick the 4 circle-edge points):
    python -m barnes_maze_analysis sample barnes_config.json

    # Track only:
    python -m barnes_maze_analysis track <video> --edge-points x1,y1,x2,y2,x3,y3,x4,y4

    # Analyze existing tracking:
    python -m barnes_maze_analysis analyze <out_dir> --edge-points ... --target-quadrant 3

    # Full pipeline:
    python -m barnes_maze_analysis full <video> --edge-points ... --target-quadrant 3

    # Batch from config:
    python -m barnes_maze_analysis batch barnes_config.json

    # Preview circle + quadrants on an image:
    python -m barnes_maze_analysis quads <image_or_dir> --edge-points ...
"""

import os
import json
import argparse
import glob
import numpy as np
import pandas as pd
import cv2

from .tracking import track_video
from .analysis import compute_velocity, compute_activity, compute_quadrant_analysis
from .plotting import (
    plot_velocity_summary, plot_velocity_histogram, plot_quadrants,
    plot_trajectory, plot_trajectory_clean, draw_circle_quadrants,
)
from .io import save_tracking_results, load_tracking_results, save_sample_frames
from .geometry import fit_circle, quadrant_boundaries


def parse_coords(s):
    """Parse comma-separated coordinate string into list of [x,y] pairs."""
    coords = [float(x) for x in s.split(",")]
    return [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]


def cmd_track(args):
    edge_points = parse_coords(args.edge_points) if args.edge_points else None
    crop = [int(x) for x in args.crop.split(",")] if getattr(args, "crop", None) else None
    start_s = getattr(args, "start", None)
    end_s = getattr(args, "end", None)

    results = track_video(args.video, crop=crop, edge_points=edge_points, pad=args.pad,
                          start_s=start_s, end_s=end_s)

    video_dir = os.path.dirname(os.path.abspath(args.video))
    name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join(video_dir, "barnes", name)

    save_tracking_results(results, out_dir, save_frames=True, every=args.every)
    print(f"\nOutput: {out_dir}")


def cmd_analyze(args):
    data, df = load_tracking_results(args.out_dir)
    fps = data["fps"]
    out_dir = os.path.join(args.out_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    edge_points = parse_coords(args.edge_points) if args.edge_points else data.get("edge_points")
    if edge_points is None:
        raise SystemExit("No edge points: pass --edge-points x1,y1,...,x4,y4")
    cx, cy, r = fit_circle(edge_points)
    center = (cx, cy)
    boundaries = quadrant_boundaries(center, edge_points)

    x = df["x"].values.astype(float)
    y = df["y"].values.astype(float)

    # px -> mm scale from arena diameter (default 920 mm mouse Barnes maze)
    arena_diameter_mm = args.arena_diameter_mm
    px_per_mm = (2 * r) / arena_diameter_mm if arena_diameter_mm else None
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

    quad_result = compute_quadrant_analysis(
        df, center, r, boundaries,
        target_quadrant=args.target_quadrant,
        periphery_threshold=args.periphery_threshold,
        fps=fps,
    )

    plot_quadrants(time_s, quad_result, fps, os.path.join(out_dir, "quadrants.png"),
                   target_quadrant=args.target_quadrant)
    print(f"Saved: {os.path.join(out_dir, 'quadrants.png')}")

    sample_path = os.path.join(args.out_dir, "samples", "original.png")
    if os.path.exists(sample_path):
        bg_img = cv2.imread(sample_path)
        plot_trajectory(bg_img, center, r, edge_points, df,
                        os.path.join(out_dir, "trajectory.png"),
                        target_quadrant=args.target_quadrant, smooth=args.smooth)
        print(f"Saved: {os.path.join(out_dir, 'trajectory.png')}")

    plot_trajectory_clean(center, r, edge_points, df,
                          os.path.join(out_dir, "trajectory_clean.png"),
                          target_quadrant=args.target_quadrant, smooth=args.smooth)
    print(f"Saved: {os.path.join(out_dir, 'trajectory_clean.png')}")

    # Metrics
    metrics = {
        "arena": {
            "center": [round(cx, 1), round(cy, 1)],
            "radius_px": round(r, 1),
            "arena_diameter_mm": arena_diameter_mm,
            "px_per_mm": round(px_per_mm, 4) if px_per_mm else None,
        },
        **quad_result["metrics"],
        **{k: v for k, v in activity.items() if k != "active_mask"},
    }
    with open(os.path.join(out_dir, "quadrants.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {os.path.join(out_dir, 'quadrants.json')}")

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

    m = quad_result["metrics"]
    print(f"\nDuration: {stats['duration_s']:.1f}s")
    print(f"Total distance: {stats[f'total_distance_{unit}']:.0f} {unit}")
    print(f"Mean velocity: {stats[f'mean_velocity_{unit}_s']:.0f} {unit}/s")
    print(f"Quadrant transitions: {m['total_transitions']}")
    if args.target_quadrant is not None:
        lat = m.get("latency_to_target_s")
        print(f"Target Q{args.target_quadrant}: {m.get('target_pct', 0):.1f}% | "
              f"latency: {lat if lat is not None else 'n/a'}s")
    print(f"Periphery (thigmotaxis): {m['periphery_pct']:.1f}%")


def cmd_full(args):
    """Run track + analyze."""
    cmd_track(args)
    video_dir = os.path.dirname(os.path.abspath(args.video))
    name = os.path.splitext(os.path.basename(args.video))[0]
    args.out_dir = os.path.join(video_dir, "barnes", name)

    data, _ = load_tracking_results(args.out_dir)
    edge_points = parse_coords(args.edge_points) if args.edge_points else None
    save_sample_frames(args.video, args.out_dir, data, edge_points, args.target_quadrant)

    cmd_analyze(args)


def cmd_quads(args):
    """Preview circle + quadrants on image(s)."""
    edge_points = parse_coords(args.edge_points)
    cx, cy, r = fit_circle(edge_points)

    if os.path.isdir(args.input):
        images = sorted(glob.glob(os.path.join(args.input, "*.png")))
        images += sorted(glob.glob(os.path.join(args.input, "*.jpg")))
    else:
        images = [args.input]

    out_dir = args.output or (args.input.rstrip("/") + "_quads" if os.path.isdir(args.input)
                              else os.path.join(os.path.dirname(args.input), "quads_viz"))
    os.makedirs(out_dir, exist_ok=True)

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        draw_circle_quadrants(img, (cx, cy), r, edge_points, target_quadrant=args.target_quadrant)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), img)

    print(f"Saved {len(images)} image(s) to {out_dir}  (center=({cx:.0f},{cy:.0f}) r={r:.0f}px)")


def cmd_pick(args):
    """Launch the interactive GUI to pick edge points + open (escape) hole."""
    from .gui import pick_and_update_config

    stem = os.path.splitext(os.path.basename(args.image))[0]
    n_holes = 18
    video_path = None
    video_file = args.video

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
        n_holes = cfg.get("n_holes", 18)
        video_dir = cfg.get("video_dir", os.path.dirname(os.path.abspath(args.config)))
        for v in cfg.get("videos", []):
            if os.path.splitext(v.get("file", ""))[0] == stem:
                video_file = v["file"]
                video_path = os.path.join(video_dir, v["file"])
                break
    if args.video_path:
        video_path = args.video_path

    pick_and_update_config(args.image, config_path=args.config, video_file=video_file,
                           video_path=video_path, n_holes=n_holes)


def cmd_batch(args):
    """Run full pipeline on all videos defined in a config file."""
    import time

    with open(args.config) as f:
        config = json.load(f)

    video_dir = config.get("video_dir", os.path.dirname(os.path.abspath(args.config)))
    videos = config.get("videos", [])
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
            edge_points=cfg.get("edge_points"),
            crop=cfg.get("crop"),
            every=cfg.get("every", 200),
            pad=cfg.get("pad", 60),
            start=cfg.get("start"),
            end=cfg.get("end"),
            smooth=cfg.get("smooth", 30),
            target_quadrant=cfg.get("target_quadrant"),
            periphery_threshold=cfg.get("periphery_threshold", 0.65),
            arena_diameter_mm=cfg.get("arena_diameter_mm", 920.0),
            activity_threshold=cfg.get("activity_threshold", 20.0),
        )

        t0 = time.time()
        cmd_full(ns)
        elapsed = time.time() - t0
        total_elapsed = time.time() - start_all
        remaining = (total_elapsed / (idx + 1)) * (len(videos) - idx - 1)
        print(f"  Done in {elapsed:.0f}s | Elapsed: {total_elapsed/60:.1f}min | Remaining: ~{remaining/60:.0f}min")

    print(f"\n===== ALL DONE in {(time.time()-start_all)/60:.1f} min =====")


def cmd_sample(args):
    """Extract a middle frame from each video for manual annotation of edge points."""
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
    print("Identify 4 points on the circle edge (the quadrant dividing directions),")
    print("then fill in 'edge_points' per video and set 'target_quadrant'.")


def cmd_init(args):
    """Generate a template config file."""
    video_dir = os.path.abspath(args.video_dir)
    videos_found = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

    config = {
        "video_dir": video_dir,
        "smooth": 30,
        "activity_threshold": 20.0,
        "periphery_threshold": 0.65,
        "arena_diameter_mm": 920.0,
        "n_holes": 18,
        "videos": [],
    }
    for f in videos_found:
        config["videos"].append({
            "file": f,
            "start": None,
            "end": None,
            "edge_points": None,
            "escape_hole": None,
            "target_quadrant": None,
        })

    out_path = args.output or os.path.join(video_dir, "barnes_config.json")
    with open(out_path, "w") as fp:
        json.dump(config, fp, indent=2)
    print(f"Config written to {out_path} with {len(videos_found)} videos.")
    print("Run 'sample' next, annotate the 4 edge points, then 'batch'.")


def main():
    parser = argparse.ArgumentParser(description="Barnes Maze Analysis Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_analyze_opts(p):
        p.add_argument("--edge-points", help="4 circle-edge points: x1,y1,x2,y2,x3,y3,x4,y4")
        p.add_argument("--target-quadrant", type=int, help="Quadrant (1-4) holding the escape hole")
        p.add_argument("--periphery-threshold", type=float, default=0.65,
                       help="Normalized radius above which mouse is 'at periphery' (default 0.65)")
        p.add_argument("--arena-diameter-mm", type=float, default=920.0,
                       help="Platform diameter in mm for px->mm scale (default 920)")
        p.add_argument("--smooth", type=int, default=30, help="Velocity smoothing window (default 30)")
        p.add_argument("--activity-threshold", type=float, default=20.0,
                       help="Velocity threshold for moving/rest in mm/s (default 20)")

    p_track = sub.add_parser("track", help="Run tracking only")
    p_track.add_argument("video")
    p_track.add_argument("--edge-points", help="4 circle-edge points: x1,y1,...,x4,y4")
    p_track.add_argument("--crop", help="Rect crop: x1,x2,y1,y2")
    p_track.add_argument("--every", type=int, default=200)
    p_track.add_argument("--pad", type=int, default=60)
    p_track.add_argument("--start", type=float)
    p_track.add_argument("--end", type=float)

    p_analyze = sub.add_parser("analyze", help="Analyze tracking results")
    p_analyze.add_argument("out_dir")
    add_analyze_opts(p_analyze)

    p_full = sub.add_parser("full", help="Track + analyze")
    p_full.add_argument("video")
    p_full.add_argument("--crop")
    p_full.add_argument("--every", type=int, default=200)
    p_full.add_argument("--pad", type=int, default=60)
    p_full.add_argument("--start", type=float)
    p_full.add_argument("--end", type=float)
    add_analyze_opts(p_full)

    p_batch = sub.add_parser("batch", help="Run full pipeline on all videos from config")
    p_batch.add_argument("config")

    p_sample = sub.add_parser("sample", help="Extract middle frame from each video")
    p_sample.add_argument("config")

    p_init = sub.add_parser("init", help="Generate template config for a video directory")
    p_init.add_argument("video_dir")
    p_init.add_argument("--output")

    p_quads = sub.add_parser("quads", help="Preview circle + quadrants on image(s)")
    p_quads.add_argument("input")
    p_quads.add_argument("--edge-points", required=True)
    p_quads.add_argument("--target-quadrant", type=int)
    p_quads.add_argument("--output")

    p_pick = sub.add_parser("pick", help="Interactive GUI: click 4 edge points + open (escape) hole")
    p_pick.add_argument("image", help="Sample image (e.g. samples/c1r1.png)")
    p_pick.add_argument("--config", help="Config JSON to write edge_points/escape_hole/target_quadrant into")
    p_pick.add_argument("--video", help="Video filename to match in config (default: <image>.mp4)")
    p_pick.add_argument("--video-path", help="Path to the video (for background-based hole detection)")

    args = parser.parse_args()
    cmds = {
        "track": cmd_track, "analyze": cmd_analyze, "full": cmd_full,
        "quads": cmd_quads, "batch": cmd_batch, "sample": cmd_sample, "init": cmd_init,
        "pick": cmd_pick,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
