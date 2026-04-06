# OFT Analysis - Development Context

## Project Overview

Open Field Test (OFT) mouse tracking pipeline developed for NBRC behavioral neuroscience experiments. The videos are top-view recordings of white mice in a dark arena with reflective floor and grid lines.

## Package Structure

```
oft_analysis/
├── __init__.py      # Package exports
├── __main__.py      # python -m oft_analysis entry point
├── run.py           # CLI: track, analyze, full, grid subcommands
├── tracking.py      # Background subtraction mouse tracker (core engine)
├── analysis.py      # Velocity, activity, grid zones, transitions, wall hugging
├── plotting.py      # All visualization (velocity, histogram, transitions, grid overlay)
├── io.py            # Save/load tracking results, frame extraction
├── CLAUDE.md        # This file — development context and decisions
└── README.md        # User-facing documentation
```

**Module responsibilities:**
- `tracking.py` — stateless detection functions + `track_video()` orchestrator. No I/O.
- `analysis.py` — pure computation on numpy/pandas arrays. No I/O, no plotting.
- `plotting.py` — matplotlib/cv2 rendering. Takes computed data, writes images.
- `io.py` — disk I/O: save/load JSON, CSV, frame images.
- `run.py` — CLI glue. Parses args, calls modules in sequence.

## Data Location

- **Videos:** `/media/smummaneni/External/nbrc/OFT (19_03)/` (14 videos: c1r1-c6r2.mp4, ~5 min each at 28 fps, 1920x1080)
- **Trimmed test video:** `test.mp4` (c1r1 with 14s trimmed from start, 22s from end)
- **DLC Project:** `/media/smummaneni/External/nbrc/OFT (19_03)/OFT-smummaneni-2026-04-01/`
- **Optical flow results:** `/media/smummaneni/External/nbrc/OFT (19_03)/optflow/`

## Arena Parameters

- **Polygon ROI (full frame):** `[[191,60], [200,1047], [1164,1047], [1190,87]]` — the arena boundary in the original video frame, used with `--polygon`
- **Grid boundary (cropped coords):** `[[170,110], [173,850], [899,847], [912,127]]` — the arena floor boundary in the cropped image (after ROI bounding rect extraction), used with `--boundary`
- **Grid:** 4x4 (16 cells, numbered 1-16 left-to-right top-to-bottom)
- **Inner zone cells:** 6, 7, 10, 11 (center 4 cells)
- **Wall hugging threshold:** 5% from boundary (normalized via perspective transform)

## What Was Tried

### DeepLabCut Pose Estimation

1. **SuperAnimal pretrained model** (`superanimal_topviewmouse` + `hrnet_w32` + `fasterrcnn_resnet50_fpn_v2`):
   - Detection rate: ~14% — Faster R-CNN detector fails on dark arena with reflections
   - Speed: ~10 it/s with detector (bottleneck), ~110 it/s pose-only on GPU

2. **Custom trained model** (ResNet-50, bottom-up, single animal):
   - Trained on 38 hand-labeled frames (2 iterations of labeling + outlier refinement)
   - Bodyparts: nose, body_center, tail_base
   - Detection at pcutoff=0.6: ~16-33% | At pcutoff=0.2: nose 100%, body_center 70%, tail_base 76%
   - Model: `dlc-models-pytorch/iteration-1/OFTApr1-trainset95shuffle1/train/snapshot-best-060.pt`

3. **Optical flow bboxes + DLC pose** — background subtraction for detection, DLC for keypoints on bbox crops. 0.028s/frame (5x faster than full-frame). Limited by pose model confidence on this arena.

### Why DLC Underperforms Here

- Dark arena with strong reflections and glare on floor
- White mouse on reflective surface — low contrast in some positions
- Grid lines on floor generate false positive detections
- Only 38 training frames — insufficient diversity for robust generalization

### Optical Flow (Final Approach)

- **Background subtraction** using median of 200 sampled frames across video
- **100% detection rate** — works perfectly for static-camera single-mouse setup
- ~1 min to process full video on CPU — no GPU needed
- Centroid tracking is sufficient for standard OFT metrics

## Key Decisions

- **Optical flow over DLC:** DLC's pretrained detector fails on this arena (14% detection). Custom training improved pose accuracy but detection remained unreliable. Optical flow gives 100% detection with zero training. DLC pose can still be used on bbox crops if body part positions are needed later.
- **Multi-frame velocity:** Median velocity across windows [1,2,3,5] frames rejects single-frame detection jumps. IQR-based clipping (3x IQR above Q3) removes remaining artifacts. This is in `analysis.compute_velocity()`.
- **Polygon ROI:** Arena walls are trapezoidal due to camera angle. Polygon masking (not just rectangle crop) prevents false detections outside the arena. The polygon is applied to the background-subtracted difference image.
- **Perspective transform for grid/wall:** Arena trapezoid is mapped to a unit square via `cv2.getPerspectiveTransform`. Grid cell assignment and wall distance are both computed in this normalized space, making them robust to camera perspective.
- **Separation of concerns:** tracking.py has no I/O, analysis.py has no plotting, plotting.py has no analysis logic. This makes each module testable and reusable independently.

## Environment

- **Python:** 3.12 (venv at `/home/smummaneni/nbrc/DeepLabCut/.venv/`)
- **PyTorch:** 2.5.1+cu121 (for DLC scripts in `scripts/oft_tracking/`, not needed for this package)
- **GPU:** NVIDIA GeForce RTX 3070 Laptop GPU (8GB VRAM, driver 535, CUDA 12.2 max)
- This package only needs: opencv-python, numpy, pandas, matplotlib, scipy

## Known Issues

- `models_to_framework.json` in DLC has a trailing comma that breaks on Python 3.14's strict JSON parser (fixed locally)
- CUDA state corrupts if a GPU process receives SIGSTOP — requires full reboot to recover
- DLC's `video_inference_superanimal` with `video_adapt=True` breaks on paths containing spaces
- OpenCV `VideoWriter` is slow for annotated videos — use ffmpeg pipe or subprocess instead

## Experimental Scripts

Older DLC-based scripts in `scripts/oft_tracking/` (kept for reference):
- `run_detections.py` — SuperAnimal/trained model inference on extracted frames with timing
- `run_pose.py` — pose estimation using precomputed optflow bboxes
- `run_combined_video.py` — combined optflow + trained model overlay video (OpenCV writer)
- `run_trained_optflow.py` — trained model + optflow + ffmpeg pipe video
- `track_optical_flow.py` — earlier monolithic optflow tracker (before modularization)
