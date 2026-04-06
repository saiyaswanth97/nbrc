# OFT Analysis

Open Field Test analysis using background subtraction mouse tracking. No GPU or deep learning required — works on any machine with Python and OpenCV.

## Setup for New Users

### 1. Install dependencies

```bash
pip install opencv-python numpy pandas matplotlib scipy
```

### 2. Generate config

```bash
cd DeepLabCut/scripts
python -m oft_analysis init /path/to/videos/
```

This creates `oft_config.json` in the video directory with all `.mp4` files listed.

### 3. Extract sample frames

```bash
python -m oft_analysis sample oft_analysis/oft_config.json
```

Saves a middle frame from each video to `<video_dir>/samples/`. Use these to manually identify:

- **`polygon`** — 4 corners of the arena walls in the original video frame (x1,y1,x2,y2,...). Order: TL, BL, BR, TR. Used to mask tracking to the arena only.
- **`boundary`** — 4 corners of the arena floor grid lines in the original video frame (same format). Used for grid cell mapping, zone classification, and wall distance. Typically slightly inset from polygon.

### 4. Edit config

Open `oft_config.json` and fill in per-video values:

```json
{
  "video_dir": "/path/to/videos",
  "grid": "4x4",
  "smooth": 30,
  "activity_threshold": 20.0,
  "wall_threshold": 0.05,
  "videos": [
    {
      "file": "c1r1.mp4",
      "start": 14,
      "end": -22,
      "polygon": "191,59,198,1047,1166,1046,1190,86",
      "boundary": "363,172,367,906,1089,906,1104,186"
    }
  ]
}
```

| Field | Scope | Description |
|-------|-------|-------------|
| `video_dir` | global | Directory containing the video files |
| `grid` | global | Grid dimensions (default: `4x4`) |
| `smooth` | global | Velocity smoothing window in frames (default: `30`) |
| `activity_threshold` | global | Velocity below this (px/s) = resting (default: `20.0`) |
| `wall_threshold` | global | Fraction of arena width for wall hugging (default: `0.05`) |
| `file` | per-video | Video filename |
| `start` | per-video | Start time in seconds, `null` for beginning |
| `end` | per-video | End time in seconds, negative = from end (e.g. `-22`), `null` for full |
| `polygon` | per-video | Arena wall corners: `x1,y1,x2,y2,x3,y3,x4,y4` (TL,BL,BR,TR) |
| `boundary` | per-video | Floor grid corners: same format as polygon |

Per-video entries can override any global field.

### 5. Run batch

```bash
python -m oft_analysis batch oft_analysis/oft_config.json
```

Processes all videos (~1 min each on CPU). Output goes to `<video_dir>/optflow/<name>/`.

## Commands

### `init` — Generate config template

```bash
python -m oft_analysis init /path/to/videos/
```

### `sample` — Extract sample frames for annotation

```bash
python -m oft_analysis sample config.json
```

### `batch` — Run full pipeline from config

```bash
python -m oft_analysis batch config.json
```

### `full` — Track + analyze a single video

```bash
python -m oft_analysis full /path/to/video.mp4 \
    --polygon 191,59,198,1047,1166,1046,1190,86 \
    --boundary 363,172,367,906,1089,906,1104,186 \
    --start 14 --end -22 --smooth 30
```

### `track` — Run tracking only

```bash
python -m oft_analysis track <video> \
    --polygon x1,y1,x2,y2,x3,y3,x4,y4 \
    --start 14 --end -22 \
    --every 200 --pad 60
```

### `analyze` — Analyze existing tracking results

```bash
python -m oft_analysis analyze <optflow_dir> \
    --boundary x1,y1,x2,y2,x3,y3,x4,y4 \
    --grid 4x4 --smooth 30 \
    --activity-threshold 20.0 --wall-threshold 0.05
```

### `grid` — Visualize arena grid on images

```bash
python -m oft_analysis grid <image_or_dir> \
    --boundary x1,y1,x2,y2,x3,y3,x4,y4
```

## As a Library

```python
from oft_analysis import (
    track_video,
    save_tracking_results,
    load_tracking_results,
    save_sample_frames,
    compute_velocity,
    compute_activity,
    compute_grid_analysis,
    plot_velocity_summary,
    plot_transitions,
    draw_grid,
)

# Track
results = track_video("video.mp4",
    polygon=[[191,59], [198,1047], [1166,1046], [1190,86]],
    start_s=14, end_s=-22)
save_tracking_results(results, "output/")

# Load and analyze
data, df = load_tracking_results("output/")
velocity, displacement = compute_velocity(df["x"].values, df["y"].values, data["fps"])
activity = compute_activity(velocity_smoothed, data["fps"], threshold=20.0)
grid = compute_grid_analysis(df, boundary)
```

## Output

### Per-video output (`optflow/<name>/`)

| Directory | File | Description |
|-----------|------|-------------|
| `frames/` | `frame_NNNNNN.png` | Sampled cropped frame images (every N frames) |
| `viz/` | `frame_NNNNNN.png` | Annotated frames with bounding box + centroid |
| `samples/` | `original.png` | Full frame from middle of video |
| `samples/` | `cropped.png` | Polygon-cropped arena region |
| `samples/` | `grid.png` | Cropped arena with grid overlay |
| | `bboxes.json` | Per-frame bounding box, centroid, and video metadata |
| | `centroids.csv` | Frame, x, y, detected — one row per frame |
| `analysis/` | `velocity.png` | Velocity over time, cumulative distance, activity bouts |
| `analysis/` | `velocity_hist.png` | Velocity distribution histogram |
| `analysis/` | `transitions.png` | Grid cell over time, inner/outer zone, wall hugging |
| `analysis/` | `stats.json` | Velocity and distance summary |
| `analysis/` | `transitions.json` | Zone, transition, bout, and wall hugging metrics |

## Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| **Velocity** | Mean/median velocity | Multi-frame median with IQR outlier rejection |
| **Distance** | Total distance | Cumulative displacement (outlier-clipped) |
| **Activity** | Moving/rest % | Based on absolute velocity threshold (default 20 px/s) |
| **Activity** | Bout count + avg duration | Contiguous moving and rest episodes |
| **Grid** | Cell occupancy | % time in each grid cell |
| **Grid** | Cell transitions | Total crossings between cells |
| **Zone** | Inner zone % | Time in center 4 cells (6,7,10,11) |
| **Zone** | Inner entries | Number of outer-to-inner crossings |
| **Zone** | Middle zone crossings | Entries into inner zone from outer |
| **Wall** | Wall hugging % | Time within 5% of boundary (perspective-normalized) |

## Tracking Details

1. **Background model:** Median of 200 frames sampled across the video
2. **Detection:** Per-frame absolute difference from background, thresholded, morphologically cleaned, contour extraction. Largest valid contour (200-50000 px area) is the mouse.
3. **Area filter:** Detections with area < 40% of median area are rejected as partial detections (e.g. only tail visible).
4. **Interpolation:** Gaps up to 10 frames are filled via linear interpolation from neighboring detections.
5. **Polygon masking:** Difference image is AND-masked with the arena polygon to exclude detections outside the arena.
6. **Grid mapping:** Boundary corners are mapped to a unit square via perspective transform (`cv2.getPerspectiveTransform`). Grid cell = integer part of transformed coordinates. Wall distance = min distance to any edge in normalized space.
7. **Velocity:** Displacement computed over multiple frame windows [1,2,3,5], median taken across windows. Outliers clipped at Q3 + 3*IQR.
8. **Time trimming:** `start` and `end` (seconds) in config skip the beginning/end of videos (e.g. before mouse is placed or after removal).

## Requirements

```
opencv-python
numpy
pandas
matplotlib
scipy
```
