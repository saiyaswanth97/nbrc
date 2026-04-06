"""I/O utilities: save/load tracking results, extract frames."""

import cv2
import json
import os
import numpy as np


def save_tracking_results(results, out_dir, save_frames=True, every=200):
    """Save tracking results to disk.

    Args:
        results: dict from tracking.track_video()
        out_dir: output directory
        save_frames: whether to save sampled frame images
        every: save frame every N frames
    """
    frames_dir = os.path.join(out_dir, "frames")
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    meta = results["metadata"]
    centroids = results["centroids"]
    bboxes = results["bboxes"]
    total = meta["total_frames"]

    # Save sampled frames + viz if requested
    saved_frames = []
    if save_frames:
        roi_bounds = meta["roi_bounds"]
        x1, y1, x2, y2 = roi_bounds
        roi_type = meta["roi_type"]

        # Build polygon mask for cropping
        poly_mask = None
        if roi_type == "polygon" and meta["roi"] is not None:
            poly_pts = np.array(meta["roi"])
            from .tracking import make_polygon_mask
            full_mask = make_polygon_mask(poly_pts, meta["height"], meta["width"])
            poly_mask = full_mask[y1:y2, x1:x2]

        cap = cv2.VideoCapture(meta["video"])
        for i in range(0, total, every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            cropped = frame[y1:y2, x1:x2].copy()
            if poly_mask is not None:
                cropped = cv2.bitwise_and(cropped, cropped, mask=poly_mask)

            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, cropped)
            saved_frames.append(i)

            # Viz
            vis = cropped.copy()
            bbox = bboxes.get(i)
            centroid = centroids.get(i)
            if bbox is not None:
                bx, by = bbox["x"] - x1, bbox["y"] - y1
                cv2.rectangle(vis, (bx, by), (bx + bbox["w"], by + bbox["h"]), (0, 255, 255), 2)
            if centroid is not None:
                cv2.circle(vis, (centroid[0] - x1, centroid[1] - y1), 10, (0, 255, 0), -1)
                cv2.circle(vis, (centroid[0] - x1, centroid[1] - y1), 12, (255, 255, 255), 2)
            status = "DETECTED" if centroid else "NO DETECTION"
            color = (0, 255, 0) if centroid else (0, 0, 255)
            cv2.putText(vis, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imwrite(os.path.join(viz_dir, f"frame_{i:06d}.png"), vis)

        cap.release()

    # Save bboxes.json
    output = {
        **meta,
        "sampled_every": every,
        "sampled_frames": saved_frames,
        "bboxes": {str(k): v for k, v in bboxes.items()},
        "centroids": {str(k): v for k, v in centroids.items()},
    }
    with open(os.path.join(out_dir, "bboxes.json"), "w") as f:
        json.dump(output, f)

    # Save centroids.csv
    with open(os.path.join(out_dir, "centroids.csv"), "w") as f:
        f.write("frame,x,y,detected\n")
        for i in range(total):
            c = centroids.get(i)
            if c:
                f.write(f"{i},{c[0]},{c[1]},1\n")
            else:
                f.write(f"{i},,0\n")


def load_tracking_results(optflow_dir):
    """Load tracking results from disk. Returns (data_dict, centroids_df)."""
    with open(os.path.join(optflow_dir, "bboxes.json")) as f:
        data = json.load(f)
    import pandas as pd
    df = pd.read_csv(os.path.join(optflow_dir, "centroids.csv"))
    return data, df


def save_sample_frames(video_path, out_dir, meta, boundary=None, grid_rows=4, grid_cols=4):
    """Save sample frames from the middle of the video.

    Saves to out_dir/samples/:
        - original.png: full frame
        - cropped.png: polygon-cropped region
        - grid.png: cropped with grid overlay
    """
    from .plotting import draw_grid

    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return

    # Original full frame
    cv2.imwrite(os.path.join(samples_dir, "original.png"), frame)

    # Cropped to polygon ROI
    roi_bounds = meta.get("roi_bounds", [0, 0, meta["width"], meta["height"]])
    x1, y1, x2, y2 = roi_bounds
    cropped = frame[y1:y2, x1:x2].copy()

    if meta.get("roi_type") == "polygon" and meta.get("roi") is not None:
        from .tracking import make_polygon_mask
        poly_pts = np.array(meta["roi"])
        full_mask = make_polygon_mask(poly_pts, meta["height"], meta["width"])
        poly_mask = full_mask[y1:y2, x1:x2]
        cropped = cv2.bitwise_and(cropped, cropped, mask=poly_mask)

    cv2.imwrite(os.path.join(samples_dir, "cropped.png"), cropped)

    # Cropped with grid overlay
    if boundary is not None:
        grid_img = cropped.copy()
        # Shift boundary coords to cropped space
        boundary_cropped = [[bx - x1, by - y1] for bx, by in boundary]
        draw_grid(grid_img, boundary_cropped, grid_rows, grid_cols)
        cv2.imwrite(os.path.join(samples_dir, "grid.png"), grid_img)

    print(f"Saved sample frames: {samples_dir}/")


def extract_frames(video_path, output_dir, every=1000):
    """Extract frames from video at regular intervals."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name = os.path.splitext(os.path.basename(video_path))[0]
    saved = 0
    for i in range(0, total, every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"{name}_frame{i:06d}.png"), frame)
        saved += 1
    cap.release()
    print(f"Saved {saved} frames to {output_dir}")
