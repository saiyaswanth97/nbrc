"""Y-maze pipeline CLI.

    python -m ymaze_analysis init <video_dir> [--output-dir DIR]
    python -m ymaze_analysis detect <config>        # find the maze in every video
    python -m ymaze_analysis review <config>        # check / fix the maze (GUI)
    python -m ymaze_analysis batch <config>         # track + score every video
    python -m ymaze_analysis blocked <train_config> [--apply test1.json,...]
    python -m ymaze_analysis summary <config>
"""

import os
import json
import argparse

import cv2
import numpy as np
import pandas as pd

from oft_analysis_open_field.tracking import build_background, track_video
from oft_analysis_open_field.analysis import kalman_smooth, compute_path
from oft_analysis_open_field.io import save_tracking_results, load_tracking_results
from oft_analysis_open_field.run import rat_group, write_group_summary
from .geometry import ARMS, detect_maze, maze_polygon, warp_maze, draw_maze
from .analysis import score
from .plotting import plot_trajectory, plot_timeline

DEFAULTS = {
    "entry_frac": 0.33,      # arm entered when body centre is 1/3 of the way to the tip
    "exit_frac": 0.2,        # back in the centre when below 1/5 of the way
    "novel_arm": None,       # arm blocked during training (A/B/C), test sessions only
    "duration_s": None,      # analyse this long from when the rat is first seen; None = to end
    "alternation": True,     # report spontaneous alternation for this session
    "groups": None,          # {"control": [cage numbers], "default": "CUMS"}
    "arm_length_mm": None,   # centre-to-tip length; set to report distance in mm
    "max_area_ratio": 2.5,   # drop detections > this x median blob area (experimenter's hand)
    "kalman_meas_px": 8.0,
    "kalman_accel_px": 800.0,
    "min_step_px": 14.0,
}


def video_background(path, n_samples=40):
    cap = cv2.VideoCapture(path)
    try:
        return build_background(cap, n_samples=n_samples, percentile=10)
    finally:
        cap.release()


def _load(path):
    with open(path) as f:
        return json.load(f)


def _save(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def _out_base(cfg):
    return os.path.expanduser(cfg["output_dir"])


def cmd_init(args):
    video_dir = os.path.abspath(os.path.expanduser(args.video_dir))
    files = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
    out = args.output_dir or os.path.join(os.path.expanduser("~/Downloads/results"),
                                          *video_dir.split(os.sep)[-2:])
    cfg = {"video_dir": video_dir, "output_dir": out,
           "phase": "train" if "train" in os.path.basename(video_dir).lower() else "test",
           **DEFAULTS, "videos": [{"file": f} for f in files]}
    path = args.output or os.path.join(video_dir, "ymaze_config.json")
    _save(cfg, path)
    print(f"Config written to {path} ({len(files)} videos, phase={cfg['phase']}, output={out})")


def cmd_detect(args):
    """Detect the maze in each video; failed fits are propagated from a good one."""
    from oft_analysis_open_field.autoannotate import _features, match_homography
    cfg = _load(args.config)
    preview_dir = os.path.join(_out_base(cfg), "_maze_check")
    os.makedirs(preview_dir, exist_ok=True)
    sift = cv2.SIFT_create(nfeatures=4000)
    bgs, good = {}, []
    for v in cfg["videos"]:
        if v.get("verified") and not args.redo:
            continue
        bg = bgs[v["file"]] = video_background(os.path.join(cfg["video_dir"], v["file"]))
        maze, problems = detect_maze(bg)
        v["maze"], v["maze_problems"] = maze, problems
        if maze and not problems:
            good.append(v)
        print(f"  {v['file']}: {'ok' if not problems else 'FLAGGED: ' + '; '.join(problems)}")

    for v in cfg["videos"]:
        if v.get("maze_problems") and good and v["file"] in bgs:
            tgt = _features(sift, bgs[v["file"]])
            best = None
            for g in good:
                H, n = match_homography(_features(sift, bgs[g["file"]]), tgt)
                if H is not None and (best is None or n > best[1]):
                    best = (H, n, g)
            if best and best[1] >= 50:
                v["maze"] = warp_maze(best[2]["maze"], best[0])
                v["maze_problems"] = [f"propagated from {best[2]['file']} ({best[1]} matches) — please review"]
                print(f"  {v['file']}: propagated from {best[2]['file']}")

    for v in cfg["videos"]:
        if v["file"] in bgs and v.get("maze"):
            img = draw_maze(cv2.cvtColor(bgs[v["file"]], cv2.COLOR_GRAY2BGR), v["maze"],
                            cfg["entry_frac"], cfg["exit_frac"])
            cv2.imwrite(os.path.join(preview_dir, os.path.splitext(v["file"])[0] + ".png"), img)
    _save(cfg, args.config)
    flagged = [v["file"] for v in cfg["videos"] if v.get("maze_problems")]
    print(f"\nPreviews: {preview_dir}/")
    if flagged:
        print(f"Review these: {', '.join(flagged)}")


def analyse_video(cfg, v, reanalyse=False):
    """Track and score one video (reanalyse=True reuses saved tracking). Returns metrics."""
    p = {**DEFAULTS, **{k: cfg[k] for k in DEFAULTS if k in cfg}, **v}
    name = os.path.splitext(v["file"])[0]
    video = os.path.join(cfg["video_dir"], v["file"])
    out_dir = os.path.join(_out_base(cfg), name)
    maze = v["maze"]

    if not (reanalyse and os.path.exists(os.path.join(out_dir, "centroids.csv"))):
        cap = cv2.VideoCapture(video)
        shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cap.release()
        results = track_video(video, polygon=maze_polygon(maze, shape), max_gap=10, bg_percentile=10)

        # Drop oversized blobs (experimenter's hand placing / removing the rat)
        areas = np.array([b["area"] for b in results["bboxes"].values() if b])
        if len(areas):
            cap_area = np.median(areas) * p["max_area_ratio"]
            for i, b in results["bboxes"].items():
                if b and b["area"] > cap_area:
                    results["centroids"][i] = None
                    results["bboxes"][i] = None
        save_tracking_results(results, out_dir, save_frames=False)
    data, df = load_tracking_results(out_dir)
    fps = data["fps"]

    x = df["x"].values.astype(float)
    y = df["y"].values.astype(float)
    det = ~np.isnan(x)
    # Analysis starts once the rat has been seen continuously for 1 s
    run = np.convolve(det.astype(int), np.ones(int(fps)), "valid")
    start = int(np.argmax(run >= int(fps))) if (run >= int(fps)).any() else 0
    end = len(x) if not p["duration_s"] else min(len(x), start + int(p["duration_s"] * fps))
    x, y = x[start:end].copy(), y[start:end].copy()
    raw_det = float(det[start:end].mean() * 100)

    xs, ys = kalman_smooth(x, y, fps, meas_std=p["kalman_meas_px"], accel_std=p["kalman_accel_px"])
    speed, disp = compute_path(xs, ys, fps, p["min_step_px"], int(fps))
    metrics, states, entries = score(xs, ys, maze, fps, p["entry_frac"], p["exit_frac"],
                                     cfg.get("novel_arm") if cfg.get("phase") == "test" else None)
    mean_len = np.mean([np.linalg.norm(np.subtract(maze["tips"][a], maze["center"])) for a in ARMS])
    metrics.update({
        "video": name, "start_s": start / fps, "raw_detection_pct": raw_det,
        "distance_px": float(disp.sum()),
        "distance_arm_lengths": float(disp.sum() / mean_len),
    })
    if p["arm_length_mm"]:
        metrics["distance_mm"] = float(disp.sum() / mean_len * p["arm_length_mm"])
    flags = []
    if metrics["analysed_s"] < 60:
        flags.append(f"short session ({metrics['analysed_s']:.0f} s)")
    if raw_det < 90:
        flags.append(f"rat detected in only {raw_det:.0f}% of frames")
    if metrics["total_entries"] < 5:
        flags.append(f"only {metrics['total_entries']} arm entries (preference measures unreliable)")
    metrics["flags"] = "; ".join(flags)
    if not p["alternation"]:
        for k in ("alternations", "possible_alternations", "alternation_pct"):
            metrics.pop(k, None)

    a_dir = os.path.join(out_dir, "analysis")
    os.makedirs(a_dir, exist_ok=True)
    pd.DataFrame({"frame": np.arange(start, end), "time_s": np.arange(end - start) / fps,
                  "x": xs, "y": ys, "state": states}).to_csv(os.path.join(a_dir, "states.csv"), index=False)
    pd.DataFrame(entries).to_csv(os.path.join(a_dir, "entries.csv"), index=False)
    with open(os.path.join(a_dir, "metrics.json"), "w") as f:
        json.dump({**metrics, "maze": maze, "params": {k: p[k] for k in DEFAULTS}}, f, indent=2)
    bg = cv2.cvtColor(video_background(video, 30), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(a_dir, "background.png"), bg)
    title = f"{name}  entries={metrics['total_entries']}" + (
        f"  alternation={metrics['alternation_pct']:.1f}%" if "alternation_pct" in metrics else "") + (
        f"  novel {metrics['novel_arm']} time={metrics['novel_time_pct']:.1f}%" if "novel_time_pct" in metrics else "")
    plot_trajectory(bg, maze, xs, ys, states, os.path.join(a_dir, "trajectory.png"),
                    p["entry_frac"], p["exit_frac"], title)
    plot_timeline(states, entries, fps, os.path.join(a_dir, "timeline.png"),
                  cfg.get("novel_arm"), title)
    return metrics


def cmd_batch(args):
    import time
    cfg = _load(args.config)
    videos = [v for v in cfg["videos"] if v.get("maze")]
    if args.videos:
        keep = {os.path.splitext(n)[0] for n in args.videos.split(",")}
        videos = [v for v in videos if os.path.splitext(v["file"])[0] in keep]
    t0 = time.time()
    for i, v in enumerate(videos):
        print(f"\n===== [{i+1}/{len(videos)}] {v['file']} =====")
        m = analyse_video(cfg, v, reanalyse=args.reanalyse)
        print(f"  entries={m['total_entries']} seq={m['arm_sequence'][:40]}  ({time.time()-t0:.0f}s elapsed)")
    write_summary(args.config)


def write_summary(config_path):
    cfg = _load(config_path)
    base = _out_base(cfg)
    rows = []
    for v in cfg["videos"]:
        path = os.path.join(base, os.path.splitext(v["file"])[0], "analysis", "metrics.json")
        if os.path.exists(path):
            m = _load(path)
            row = {k: val for k, val in m.items() if not isinstance(val, (dict, list))}
            group = rat_group(row["video"], cfg.get("groups"))
            if group:
                row["group"] = group
            rows.append(row)
    os.makedirs(base, exist_ok=True)
    out = os.path.join(base, "ymaze_summary.csv")
    first = ["video", "group"] if rows and "group" in rows[0] else ["video"]
    cols = first + [c for c in (rows[0].keys() if rows else []) if c not in first]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out, index=False)
    if "group" in df:
        write_group_summary(df, os.path.join(base, "ymaze_group_summary.csv"),
                            skip=("start_s", "raw_detection_pct", "analysed_s"))
    _save(cfg, os.path.join(base, "ymaze_config.json"))
    print(f"Summary: {out}")


def cmd_summary(args):
    write_summary(args.config)


def cmd_blocked(args):
    """Find the arm the rats never entered during training (the blocked arm)."""
    cfg = _load(args.config)
    base = _out_base(cfg)
    votes = []
    print(f"{'video':32s} " + "  ".join(f"time_{a}" for a in ARMS) + "  blocked")
    for v in cfg["videos"]:
        path = os.path.join(base, os.path.splitext(v["file"])[0], "analysis", "metrics.json")
        if not os.path.exists(path):
            continue
        m = _load(path)
        times = {a: m[f"time_{a}_s"] for a in ARMS}
        blocked = min(times, key=times.get)
        votes.append(blocked)
        print(f"{v['file']:32s} " + "  ".join(f"{times[a]:6.1f}s" for a in ARMS) + f"  {blocked}")
    if not votes:
        print("No training results — run batch on the training config first.")
        return
    arm, n = max(((a, votes.count(a)) for a in ARMS), key=lambda t: t[1])
    print(f"\nBlocked arm: {arm} ({n}/{len(votes)} videos agree)")
    if args.apply:
        for p in args.apply.split(","):
            c = _load(p)
            c["novel_arm"] = arm
            _save(c, p)
            print(f"  novel_arm={arm} written to {p}")


def main():
    ap = argparse.ArgumentParser(description="Y-maze analysis")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="Create a config for a folder of videos")
    p.add_argument("video_dir")
    p.add_argument("--output-dir", help="Results folder (default: ~/Downloads/results/<parent>/<folder>)")
    p.add_argument("--output", help="Config path (default: <video_dir>/ymaze_config.json)")
    p = sub.add_parser("detect", help="Detect the maze in every video")
    p.add_argument("config")
    p.add_argument("--redo", action="store_true", help="Also re-detect verified videos")
    p = sub.add_parser("review", help="Check / fix the maze per video (GUI)")
    p.add_argument("config")
    p.add_argument("--videos")
    p.add_argument("--all", action="store_true")
    p = sub.add_parser("batch", help="Track and score every video")
    p.add_argument("config")
    p.add_argument("--videos", help="Comma-separated subset")
    p.add_argument("--reanalyse", action="store_true", help="Reuse saved tracking, only re-score")
    p = sub.add_parser("blocked", help="Find the arm blocked during training")
    p.add_argument("config", help="Training config")
    p.add_argument("--apply", help="Comma-separated test configs to write novel_arm into")
    p = sub.add_parser("summary", help="Collect results into one CSV")
    p.add_argument("config")
    args = ap.parse_args()
    if args.command == "review":
        from .gui import review_config
        review_config(args.config, args.videos.split(",") if args.videos else None, args.all)
        return
    {"init": cmd_init, "detect": cmd_detect, "batch": cmd_batch,
     "blocked": cmd_blocked, "summary": cmd_summary}[args.command](args)


if __name__ == "__main__":
    main()
