# Barnes Maze Analysis - Development Context

## Overview

Barnes maze mouse tracking pipeline for NBRC. Sibling package to
`oft_analysis_open_field`. Top-view recordings of a white mouse on a dark
**circular** platform with perimeter holes (18 in this dataset), one of which is
the escape hole. Reuses the OFT background-subtraction tracker; the
arena-specific parts (ROI, zones, holes) are reworked for a circle + quadrants.

## Package Structure

```
barnes_maze_analysis/
├── __init__.py       # Package exports
├── __main__.py       # python -m barnes_maze_analysis entry point
├── run.py            # CLI: track, analyze, full, batch, sample, init, quads, pick
├── geometry.py       # Circle fit, quadrant boundaries/assignment, circular mask
├── tracking.py       # Background-subtraction tracker (circular ROI)
├── holes.py          # Perimeter-hole detection (ellipse-ring fit) + escape hole
├── gui.py            # Interactive picker: 4 edge points + open (escape) hole
├── analysis.py       # Velocity, activity, quadrant + radial (thigmotaxis) metrics
├── plotting.py       # Velocity / quadrant / trajectory plots, circle + hole overlays
├── io.py             # Save/load results, frame extraction, sample frames
├── barnes_config.json
├── requirements.txt
├── CLAUDE.md
└── README.md
```

Module responsibilities mirror OFT: `tracking.py` has no I/O, `analysis.py` has
no plotting, `plotting.py` has no analysis logic, `run.py` is CLI glue. `gui.py`
is the only interactive module and depends on `geometry` + `holes`.

## Environment

- Local venv at `/home/smummaneni/nbrc/.venv` — opencv-python-headless, numpy,
  pandas, matplotlib, scipy. System `python3` has NO cv2. See [[nbrc-python-env]].
- The `pick` GUI needs an **interactive** matplotlib backend. `plotting.py` calls
  `matplotlib.use("Agg")` at import, so `gui._ensure_interactive_backend()`
  switches to TkAgg (or Qt) via `plt.switch_backend` before creating the figure.
  Requires system Tk (`python3-tk`). Launch with `DISPLAY` set.

## What changed vs OFT

- **ROI**: circular (fit from 4 edge points) instead of trapezoidal polygon.
  `geometry.make_circle_mask`; `track_video(edge_points=...)`.
- **Zones**: 4 angular **quadrants** instead of a 4×4 grid. `geometry`:
  `fit_circle` (Kasa least squares), `quadrant_boundaries` (sorted atan2 angles),
  `point_quadrant` (angular sector lookup).
- **Thigmotaxis**: radial periphery-vs-center via normalized radius (dist/R),
  replacing the OFT wall-distance metric.
- **Holes**: `holes.py` detects the perimeter holes — see below.
- **Target**: `target_quadrant` gives target time %, entries, latency. Derived in
  the GUI from the clicked escape hole's quadrant.
- **px→mm scale**: from platform diameter (`arena_diameter_mm`, default 920 mm):
  `px_per_mm = 2*radius_px / diameter_mm`.

## Geometry decision: 4 points → 4 quadrants

The 4 clicked edge points are quadrant **boundaries** (radial line from center
through each). Sorting their angles gives 4 sectors = Q1..Q4, numbered by
increasing angle from the smallest-angle point. Keep click order/positions
consistent across videos.

## Hole detection (holes.py)

The holes are evenly spaced but the camera views the platform at a slight angle,
so in the image the ring is an **ellipse** and holes are perspective-distorted. A
forced perfect circle mis-places them (this was a real bug). Approach:

1. `_detections`: CLAHE + Hough on the mouse-free **background** and a frame,
   several `param2` values; keep detections in the rim ring (0.62–0.97 R).
2. `_dedupe`: cluster within 35px, keep clusters seen ≥2× (robust ~16/18).
3. `cv2.fitEllipse` to the robust detections → the hole-ring ellipse.
4. Place `n_holes` on the ellipse at even parametric spacing; phase from the
   circular mean of `N*t`. Each slot snaps to a real detection if within 32px,
   else filled from the grid. Holes returned angle-ordered for stable indexing.

`identify_escape_hole(holes, hint)` = nearest hole to a hint point (the GUI passes
the user's click on the open hole).

## GUI (gui.py)

`pick` command → `QuadrantPicker`. Flow: click 4 rim points (live circle +
quadrants + holes), then click the open hole (snaps to nearest detected hole,
marks ESCAPE, derives target quadrant), Enter to save. Keys: `u` undo edge, `r`
reset, `1`-`4` manual quadrant (fallback). `pick_and_update_config` writes
`edge_points`, `escape_hole`, `target_quadrant` into the matching config entry.

Video path for background-based detection: `cmd_pick` resolves the video from the
config (`video_dir` + matching `file`) or from `--video-path`. Detection falls
back to the sample frame alone if no video is available.

## Data

- Sample video: `/home/smummaneni/nbrc/data/c1r1.mp4` (2504 frames, ~26.7 fps,
  1920×1080, ~94 s). White mouse, dark 18-hole circular platform.
- Fitted geometry: center ≈ (829, 638), radius ≈ 449 px. Escape hole ≈ (433, 652),
  in quadrant Q4 (left) with the N/E/S/W-style edge points.
- Batch result on c1r1: target Q4 ≈ 40% occupancy, periphery ≈ 81%.

## Tracking details (unchanged from OFT)

Median background of 200 sampled frames; per-frame absdiff → threshold →
morphology → largest valid contour (200–50000 px); area filter (<40% median
rejected); linear interpolation of gaps ≤10 frames; multi-window velocity
[1,2,3,5] with IQR clipping. ~100% detection on this dataset.

## Future work

- Consume `escape_hole` in `analyze`/`batch`: auto-generate `arena_annotated.png`
  and add per-hole metrics — latency & path length to the escape hole, primary/
  total errors (nose-pokes at wrong holes), search strategy (direct/serial/random).
- Optional: add hole detection output (positions JSON) to the standard pipeline.
