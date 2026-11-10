"""FST CLI.

    python -m fst_analysis extract <video_dir> [--output-dir DIR]   # per-frame movement features
"""

import os
import re
import glob
import argparse

import numpy as np

from .features import extract


def rats_from_name(stem):
    """'C10r1c8r3' -> ('c10r1', 'c8r3'); 'C12r2 r3' -> ('c12r2', 'c12r3'). First name = left cylinder."""
    s = stem.lower().replace(" ", "")
    m = re.fullmatch(r"c(\d+)r(\d+)(?:c(\d+))?r(\d+)", s)
    if not m:
        return None, None
    c1, r1, c2, r2 = m.groups()
    return f"c{c1}r{r1}", f"c{c2 or c1}r{r2}"


def cmd_extract(args):
    vids = sorted(glob.glob(os.path.join(args.video_dir, "**", "*.mp4"), recursive=True))
    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    for i, v in enumerate(vids):
        stem = os.path.splitext(os.path.basename(v))[0]
        dst = os.path.join(out_dir, stem + "_features.npz")
        if os.path.exists(dst) and not args.redo:
            print(f"[{i+1}/{len(vids)}] {stem}: exists, skipped", flush=True)
            continue
        f = extract(v)
        left, right = rats_from_name(stem)
        np.savez_compressed(dst, video=os.path.relpath(v, args.video_dir), rat_left=left or "", rat_right=right or "",
                            t=f["t"], sample_fps=f["sample_fps"], video_fps=f["video_fps"], duration_s=f["duration_s"],
                            split_x=f["split_x"], width=f["width"],
                            **{f"{s}_{k}": f[s][k] for s in ("left", "right") for k in ("flow_resid", "shape_change", "area")})
        print(f"[{i+1}/{len(vids)}] {stem}: {f['duration_s']:.0f} s, split at {f['split_x']}/{f['width']}, "
              f"rats {left or '?'} / {right or '?'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Forced swim test analysis")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract", help="Compute per-frame movement features for every video in a folder")
    p.add_argument("video_dir")
    p.add_argument("--output-dir", default="~/Downloads/results/TRD_WKY_PHD_DATA/fst/features")
    p.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    {"extract": cmd_extract}[args.command](args)


if __name__ == "__main__":
    main()
