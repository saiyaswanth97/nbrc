# OFT Analysis - Development Context

## Project Overview

Open Field Test (OFT) mouse tracking pipeline developed for NBRC behavioral neuroscience experiments. The videos are top-view recordings of white mice in a dark arena with reflective floor and grid lines.

## Package Structure

```
oft_analysis/
├── __init__.py      # Package exports
├── __main__.py      # python -m oft_analysis entry point
├── run.py           # CLI: track, analyze, full, batch, sample, init, grid subcommands
├── tracking.py      # Background subtraction mouse tracker (core engine)
├── analysis.py      # Velocity, activity, grid zones, transitions, wall hugging
├── plotting.py      # All visualization (velocity, histogram, transitions, trajectory, grid)
├── io.py            # Save/load tracking results, frame extraction, sample frames
├── oft_config.json  # Batch config for current dataset
├── CLAUDE.md        # This file — development context and decisions
└── README.md        # User-facing documentation
```

**Module responsibilities:**
- `tracking.py` — stateless detection functions + `track_video()` orchestrator. No I/O.
- `analysis.py` — pure computation on numpy/pandas arrays. No I/O, no plotting.
- `plotting.py` — matplotlib/cv2 rendering. Takes computed data, writes images.
- `io.py` — disk I/O: save/load JSON, CSV, frame images, sample frames.
- `run.py` — CLI glue. Parses args, calls modules in sequence.

## Workflow

1. `python -m oft_analysis init <video_dir>` — generates `oft_config.json`
2. `python -m oft_analysis sample config.json` — extracts middle frames to `<video_dir>/samples/`
3. User annotates polygon + boundary coords from sample images, fills in start/end times
4. `python -m oft_analysis batch config.json` — runs full pipeline on all videos

## Config Format

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

- **Global fields** (top-level): `grid`, `smooth`, `activity_threshold`, `wall_threshold`
- **Per-video fields**: `file`, `start`, `end`, `polygon`, `boundary`
- Per-video can override any global field
- `start`/`end` are in seconds. `end` can be negative (from end of video). `null` = no trimming.
- All code/config lives in `scripts/oft_analysis/`. All output goes alongside the video data.

## Coordinate Systems

- **`polygon`** — 4 corners of the arena in the original video frame (1920x1080). Order: TL, BL, BR, TR. Used for masking during tracking.
- **`boundary`** — 4 corners of the arena floor in the original video frame. Used for grid mapping, zone classification, and wall distance via perspective transform. Slightly inset from polygon (follows floor grid lines, not outer walls).
- **Centroids** — stored in original video frame coordinates.
- Both polygon and boundary vary per video since the camera shifts slightly between recordings.

## Data Location

- **Videos:** `/media/smummaneni/External/nbrc/OFT (19_03)/` (15 videos: c1r1-c6r2.mp4, ~5 min each at 28 fps, 1920x1080)
- **DLC Project:** `/media/smummaneni/External/nbrc/OFT (19_03)/OFT-smummaneni-2026-04-01/`
- **Optical flow results:** `/media/smummaneni/External/nbrc/OFT (19_03)/optflow/`
- **Sample frames:** `/media/smummaneni/External/nbrc/OFT (19_03)/samples/`

## What Was Tried

### DeepLabCut Pose Estimation

1. **SuperAnimal pretrained model** (`superanimal_topviewmouse` + `hrnet_w32` + `fasterrcnn_resnet50_fpn_v2`):
   - Detection rate: ~14% — Faster R-CNN detector fails on dark arena with reflections
   - Speed: ~10 it/s with detector (bottleneck), ~110 it/s pose-only on GPU

2. **Custom trained model** (ResNet-50, bottom-up, single animal):
   - Trained on 38 hand-labeled frames (2 iterations of labeling + outlier refinement)
   - Bodyparts: nose, body_center, tail_base
   - Detection at pcutoff=0.6: ~16-33% | At pcutoff=0.2: nose 100%, body_center 70%, tail_base 76%

3. **Optical flow bboxes + DLC pose** — background subtraction for detection, DLC for keypoints on bbox crops. 0.028s/frame (5x faster than full-frame). Limited by pose model confidence on this arena.

### Why DLC Underperforms Here

- Dark arena with strong reflections and glare on floor
- White mouse on reflective surface — low contrast in some positions
- Grid lines on floor generate false positive detections
- Only 38 training frames — insufficient diversity for robust generalization

### Optical Flow (Final Approach)

- **Background subtraction** using median of 200 sampled frames across video
- **~99-100% detection rate** — works perfectly for static-camera single-mouse setup
- **Area filter** rejects partial detections (area < 40% of median), **interpolation** fills gaps up to 10 frames
- ~1 min to process full video on CPU — no GPU needed
- Centroid tracking is sufficient for standard OFT metrics

## Key Decisions

- **Optical flow over DLC:** DLC's pretrained detector fails on this arena (14% detection). Custom training improved pose accuracy but detection remained unreliable. Optical flow gives ~100% detection with zero training.
- **Area-based filter:** Detections with contour area < 40% of the median are rejected as partial (e.g. only tail visible). These frames are then interpolated.
- **Interpolation:** Missing/rejected frames with gaps <= 10 frames are linearly interpolated from neighboring good detections.
- **Multi-frame velocity:** Median velocity across windows [1,2,3,5] frames rejects single-frame detection jumps. IQR-based clipping (3x IQR above Q3) removes remaining artifacts.
- **Absolute activity threshold:** Moving/rest classification uses absolute velocity threshold (default 20 px/s) instead of percentile-based, which was producing identical 75% moving for all videos.
- **Polygon ROI:** Arena walls are trapezoidal due to camera angle. Polygon masking prevents false detections outside the arena.
- **Perspective transform for grid/wall:** Arena trapezoid is mapped to a unit square via `cv2.getPerspectiveTransform`. Grid cell and wall distance computed in normalized space.
- **Boundary in original coords:** Both polygon and boundary use original video frame coordinates. No coordinate offset needed since centroids are also in original frame coords.
- **Trajectory plots:** Two variants — `trajectory.png` overlays centroids on the actual arena image, `trajectory_clean.png` uses a white background with only grid lines (no numbers) plus blue start/end markers. Both filter points to inside the boundary polygon.
- **Separation of concerns:** tracking.py has no I/O, analysis.py has no plotting, plotting.py has no analysis logic.

## Environment

- **Python:** 3.12 (venv at `/home/smummaneni/nbrc/DeepLabCut/.venv/`)
- **PyTorch:** 2.5.1+cu121 (for DLC scripts in `scripts/oft_tracking/`, not needed for this package)
- **GPU:** NVIDIA GeForce RTX 3070 Laptop GPU (8GB VRAM, driver 535, CUDA 12.2 max)
- This package only needs: opencv-python, numpy, pandas, matplotlib, scipy

## Known Issues

- `models_to_framework.json` in DLC has a trailing comma that breaks on Python 3.14's strict JSON parser (fixed locally)
- CUDA state corrupts if a GPU process receives SIGSTOP — requires full reboot to recover
- DLC's `video_inference_superanimal` with `video_adapt=True` breaks on paths containing spaces

## Experimental Scripts

Older DLC-based scripts in `scripts/oft_tracking/` (kept for reference):
- `run_detections.py` — SuperAnimal/trained model inference on extracted frames with timing
- `run_pose.py` — pose estimation using precomputed optflow bboxes
- `run_combined_video.py` — combined optflow + trained model overlay video (OpenCV writer)
- `run_trained_optflow.py` — trained model + optflow + ffmpeg pipe video
- `track_optical_flow.py` — earlier monolithic optflow tracker (before modularization)
