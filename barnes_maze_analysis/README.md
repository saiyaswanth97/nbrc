# Barnes Maze Analysis

Mouse tracking and behavioural analysis for the **Barnes maze** (circular
platform with perimeter escape holes). Uses background-subtraction tracking — no
GPU or deep learning required. Sibling of `oft_analysis_open_field`, adapted for
a circular arena: you annotate 4 points on the platform edge, the circle is fit
from them, and the disk is split into 4 **quadrants**. Perimeter holes and the
open (escape) hole are detected automatically.

---

## 1. Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r barnes_maze_analysis/requirements.txt
# The interactive picker GUI also needs system Tk:
sudo apt install python3-tk          # Debian/Ubuntu
```

Run everything with that interpreter (system `python3` here has no OpenCV):

```bash
.venv/bin/python -m barnes_maze_analysis <command> ...
```

## 2. Point it at your videos

Two ways to supply the video path:

- **Config (recommended, for batches):** set `video_dir` + per-video `file` in a
  JSON config. Generate a template with `init`:

  ```bash
  .venv/bin/python -m barnes_maze_analysis init /path/to/videos/
  ```

  This scans the directory for `.mp4` files and writes `barnes_config.json`.

- **Directly on the command line** for a single video: pass the video path to
  `track` / `full`, or `--video-path` to `pick`.

## 3. Annotate the arena (GUI)

Extract a middle frame from each video, then annotate it:

```bash
.venv/bin/python -m barnes_maze_analysis sample barnes_config.json
.venv/bin/python -m barnes_maze_analysis pick data/samples/c1r1.png --config barnes_config.json
```

In the **pick** window:

1. **Click 4 points** on the platform rim → the circle, quadrants, and all
   perimeter holes are drawn live. (Clicking top/right/bottom/left gives the four
   diagonal quadrants; the two dividing lines pass through your points.)
2. **Click the open (escape) hole** → it snaps to the nearest detected hole, is
   marked `ESCAPE`, and the **target quadrant is derived automatically**.
3. **Press `Enter`** to save `edge_points`, `escape_hole`, and `target_quadrant`
   back into the config for that video.

Keys: `u` = undo last edge point, `r` = reset, `1`–`4` = set target quadrant
manually (fallback if hole detection fails).

`pick` reads the matching video from the config to build a clean, mouse-free
background for hole detection. To point it at a video explicitly:

```bash
.venv/bin/python -m barnes_maze_analysis pick frame.png --video-path /path/to/video.mp4
```

## 4. Run the analysis

```bash
.venv/bin/python -m barnes_maze_analysis batch barnes_config.json
```

Processes every video in the config (~30 s each on CPU). Output goes to
`<video_dir>/barnes/<name>/`.

---

## Config format

```json
{
  "video_dir": "/path/to/videos",
  "smooth": 30,
  "activity_threshold": 20.0,
  "periphery_threshold": 0.65,
  "arena_diameter_mm": 920.0,
  "n_holes": 18,
  "videos": [
    {
      "file": "c1r1.mp4",
      "start": null,
      "end": null,
      "edge_points": "524.5,312.8,554.1,996.5,1144.5,952.7,1078.1,260.6",
      "escape_hole": "432.8,652.5",
      "target_quadrant": 4
    }
  ]
}
```

| Field | Scope | Description |
|-------|-------|-------------|
| `video_dir` | global | Directory containing the video files |
| `smooth` | global | Velocity smoothing window in frames (default 30) |
| `activity_threshold` | global | Velocity below this (mm/s) = resting (default 20) |
| `periphery_threshold` | global | Normalized radius (0–1) above which the mouse is "at the periphery" — thigmotaxis proxy (default 0.65) |
| `arena_diameter_mm` | global | Platform diameter in mm, for px→mm scaling (default 920) |
| `n_holes` | global | Number of perimeter holes (default 18; some mazes use 20) |
| `file` | per-video | Video filename (relative to `video_dir`) |
| `start` / `end` | per-video | Trim seconds; `end` negative = from end; `null` = none |
| `edge_points` | per-video | 4 rim points `x1,y1,...,x4,y4` (quadrant dividers) |
| `escape_hole` | per-video | Open/escape hole `x,y` (set by the GUI) |
| `target_quadrant` | per-video | Quadrant (1–4) containing the escape hole |

Per-video entries override any global field.

---

## Commands

| Command | Purpose |
|---------|---------|
| `init <video_dir>` | Generate a config template from a directory of `.mp4`s |
| `sample <config>` | Extract a middle frame per video for annotation |
| `pick <image> --config <cfg>` | **GUI**: click 4 rim points + the open hole; writes to config |
| `quads <image\|dir> --edge-points ...` | Non-GUI preview of circle + quadrants |
| `track <video> --edge-points ...` | Tracking only |
| `analyze <out_dir> --edge-points ... --target-quadrant N` | Analyze existing tracking |
| `full <video> --edge-points ... --target-quadrant N` | Track + analyze one video |
| `batch <config>` | Full pipeline on all videos in a config |

Single-video example without a config:

```bash
.venv/bin/python -m barnes_maze_analysis full /path/to/video.mp4 \
    --edge-points 524,313,554,997,1145,953,1078,261 \
    --target-quadrant 4 --arena-diameter-mm 920 --start 0 --end -5
```

---

## Output (`barnes/<name>/`)

| File | Description |
|------|-------------|
| `bboxes.json`, `centroids.csv` | Per-frame detections (original frame coords) |
| `samples/original.png` | Full middle frame |
| `samples/cropped.png` | Circle-masked arena |
| `samples/quadrants.png` | Arena with circle + quadrant overlay |
| `frames/`, `viz/` | Sampled + annotated frames |
| `analysis/velocity.png` | Velocity, cumulative distance, activity bouts |
| `analysis/velocity_hist.png` | Velocity distribution |
| `analysis/quadrants.png` | Quadrant over time, target vs other, periphery vs center |
| `analysis/trajectory.png` | Trajectory on the arena image with quadrant overlay |
| `analysis/trajectory_clean.png` | Trajectory on a clean circle with start/end markers |
| `analysis/quadrants.json` | Quadrant, target, and thigmotaxis metrics |
| `analysis/stats.json` | Velocity and distance summary |

---

## Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| **Velocity** | Mean/median velocity | Multi-frame median with IQR outlier rejection |
| **Distance** | Total distance | Cumulative displacement (mm) |
| **Activity** | Moving/rest %, bouts | Absolute velocity threshold |
| **Quadrant** | Occupancy % / time / entries | Per quadrant (Q1–Q4) |
| **Quadrant** | Transitions | Crossings between quadrants |
| **Target** | Time %, latency, entries | For the escape quadrant |
| **Thigmotaxis** | Periphery % | Time near the perimeter ring of holes |
| **Thigmotaxis** | Mean normalized radius | Average distance from center (0=center, 1=edge) |

---

## Use as a library

```python
import cv2
import barnes_maze_analysis as bm
from barnes_maze_analysis.tracking import build_background

# Fit circle + quadrants from 4 edge points
info = bm.circle_from_edge_points([[524,313],[554,997],[1145,953],[1078,261]], w=1920, h=1080)

# Detect the 18 perimeter holes and the escape hole
cap = cv2.VideoCapture("video.mp4"); bg = build_background(cap); cap.release()
frame = cv2.imread("frame.png")
holes, _ = bm.detect_holes([bg, frame], info["center"], info["radius"], n_holes=18)
escape_idx = bm.identify_escape_hole(holes, escape_hint=(433, 652))

# Track + analyze
results = bm.track_video("video.mp4", edge_points=[[524,313],[554,997],[1145,953],[1078,261]])
```

---

## Notes / future work

- **Hole detection** fits the perimeter ring as an *ellipse* to real Hough
  detections (the camera views the platform at an angle), then places all
  `n_holes` on it, snapping to real detections and filling gaps. This handles the
  low-contrast holes a plain circle fit would miss.
- Quadrants are numbered 1–4 in increasing angle from the first (smallest-angle)
  edge point — keep the click order/positions consistent across videos.
- The `escape_hole` is saved by the GUI but the standard `analyze`/`batch`
  metrics don't consume it yet. Hooking it up unlocks per-hole metrics:
  **latency & path length to the escape hole**, **primary/total errors**
  (nose-pokes at wrong holes), and **search strategy** (direct/serial/random).
