"""Mouse tracking via background subtraction.

Core tracking engine: builds background model, detects mouse per frame,
returns centroids and bounding boxes.
"""

import cv2
import numpy as np


def make_polygon_mask(polygon, h, w):
    """Create binary mask from polygon points."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def build_background(cap, n_samples=200):
    """Build median background model from sampled frames."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, total - 1, n_samples, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def detect_mouse(gray, bg, roi_mask, roi_bounds, kernel,
                 min_area=200, max_area=50000, pad=60, frame_w=0, frame_h=0):
    """Detect mouse in a single frame via background subtraction.

    Returns (centroid, bbox_info) where centroid is [x, y] in full frame coords
    and bbox_info is dict with x, y, w, h, area. Both None if not detected.
    """
    x1, y1, x2, y2 = roi_bounds
    diff = cv2.absdiff(gray[y1:y2, x1:x2], bg[y1:y2, x1:x2])
    diff = cv2.bitwise_and(diff, roi_mask)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [(c, cv2.contourArea(c)) for c in contours if min_area < cv2.contourArea(c) < max_area]

    if not valid:
        return None, None

    best, area = max(valid, key=lambda x: x[1])
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None, None

    cx = int(M["m10"] / M["m00"]) + x1
    cy = int(M["m01"] / M["m00"]) + y1

    bx, by, bw, bh = cv2.boundingRect(best)
    bx += x1
    by += y1
    bx = max(0, bx - pad)
    by = max(0, by - pad)
    bw = min(bw + 2 * pad, frame_w - bx) if frame_w else bw + 2 * pad
    bh = min(bh + 2 * pad, frame_h - by) if frame_h else bh + 2 * pad

    centroid = [cx, cy]
    bbox_info = {"x": bx, "y": by, "w": bw, "h": bh, "area": int(area)}
    return centroid, bbox_info


def interpolate_centroids(centroids, max_gap=10):
    """Fill missing centroids via linear interpolation.

    Args:
        centroids: dict {frame_idx: [x,y] or None}
        max_gap: max consecutive missing frames to interpolate (larger gaps left as None)

    Returns:
        new centroids dict with gaps filled, and count of interpolated frames
    """
    frames = sorted(centroids.keys())
    result = dict(centroids)
    n_interpolated = 0

    i = 0
    while i < len(frames):
        if result[frames[i]] is not None:
            i += 1
            continue

        # Find gap: consecutive None frames
        gap_start = i
        while i < len(frames) and result[frames[i]] is None:
            i += 1
        gap_end = i  # first non-None after gap (or end)

        gap_len = gap_end - gap_start
        # Need valid frames on both sides
        if gap_start == 0 or gap_end >= len(frames):
            continue
        if gap_len > max_gap:
            continue

        # Interpolate between the two bounding detections
        before = result[frames[gap_start - 1]]
        after = result[frames[gap_end]]
        for j in range(gap_start, gap_end):
            t = (j - gap_start + 1) / (gap_len + 1)
            x = int(round(before[0] + t * (after[0] - before[0])))
            y = int(round(before[1] + t * (after[1] - before[1])))
            result[frames[j]] = [x, y]
            n_interpolated += 1

    return result, n_interpolated


def track_video(video_path, crop=None, polygon=None, pad=60, max_gap=10,
                area_min_ratio=0.4, start_s=None, end_s=None, progress_every=2000):
    """Track mouse through entire video.

    Args:
        video_path: path to video file
        crop: [x1, x2, y1, y2] rectangle crop
        polygon: [[x,y], ...] polygon ROI points
        pad: padding around detected bounding box
        max_gap: max gap (frames) to interpolate missing detections (0 to disable)
        area_min_ratio: reject detections with area < ratio * median_area (0 to disable)
        start_s: start time in seconds (trim beginning)
        end_s: end time in seconds (trim end, negative = from end)
        progress_every: print progress every N frames

    Returns:
        dict with keys: centroids, bboxes, metadata
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Time trimming
    start_frame = int(start_s * fps) if start_s else 0
    duration_s = total / fps
    if end_s is not None:
        if end_s < 0:
            end_frame = max(0, int((duration_s + end_s) * fps))
        else:
            end_frame = min(total, int(end_s * fps))
    else:
        end_frame = total
    start_frame = max(0, min(start_frame, end_frame))
    total_to_track = end_frame - start_frame
    if total_to_track <= 0:
        raise ValueError(f"No frames to track: start={start_frame}, end={end_frame}, total={total}")
    if start_frame > 0 or end_frame < total:
        print(f"Trimming: frames {start_frame}-{end_frame} ({start_frame/fps:.1f}s - {end_frame/fps:.1f}s)")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build ROI
    if polygon is not None:
        poly_pts = np.array(polygon)
        roi_mask = make_polygon_mask(poly_pts, h, w)
        rx, ry, rw, rh = cv2.boundingRect(poly_pts)
        roi_bounds = (rx, ry, rx + rw, ry + rh)
    elif crop is not None:
        x1, x2, y1, y2 = crop
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255
        roi_bounds = (x1, y1, x2, y2)
        poly_pts = None
    else:
        roi_mask = np.ones((h, w), dtype=np.uint8) * 255
        roi_bounds = (0, 0, w, h)
        poly_pts = None

    roi_mask_crop = roi_mask[roi_bounds[1]:roi_bounds[3], roi_bounds[0]:roi_bounds[2]]

    print(f"Building background model...")
    bg = build_background(cap, n_samples=200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    print(f"Tracking {total_to_track} frames...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    centroids = {}
    bboxes = {}

    for i in range(total_to_track):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        centroid, bbox_info = detect_mouse(
            gray, bg, roi_mask_crop, roi_bounds, kernel,
            pad=pad, frame_w=w, frame_h=h,
        )
        centroids[i] = centroid
        bboxes[i] = bbox_info

        if progress_every and i % progress_every == 0:
            det = sum(1 for v in centroids.values() if v is not None)
            print(f"  Frame {i}/{total_to_track} ({100*i/total_to_track:.0f}%) - detected: {det}/{i+1}")

    cap.release()
    det_count = sum(1 for v in centroids.values() if v is not None)
    print(f"Detected: {det_count}/{total_to_track} ({100*det_count/total_to_track:.1f}%)")

    # Area-based filter: reject partial detections (area << median)
    areas = np.array([b["area"] for b in bboxes.values() if b is not None])
    if len(areas) > 0:
        median_area = np.median(areas)
        area_threshold = median_area * area_min_ratio
        n_rejected = 0
        for i in list(bboxes.keys()):
            if bboxes[i] is not None and bboxes[i]["area"] < area_threshold:
                centroids[i] = None
                bboxes[i] = None
                n_rejected += 1
        if n_rejected > 0:
            print(f"Rejected: {n_rejected} partial detections (area < {area_threshold:.0f}, "
                  f"median: {median_area:.0f}, ratio: {area_min_ratio})")

    # Interpolate missing detections
    n_interpolated = 0
    if max_gap > 0:
        centroids, n_interpolated = interpolate_centroids(centroids, max_gap=max_gap)
        if n_interpolated > 0:
            print(f"Interpolated: {n_interpolated} frames (max gap: {max_gap})")

    final_count = sum(1 for v in centroids.values() if v is not None)

    return {
        "centroids": centroids,
        "bboxes": bboxes,
        "metadata": {
            "video": video_path,
            "total_frames": total_to_track,
            "video_total_frames": total,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "fps": fps,
            "width": w,
            "height": h,
            "roi": poly_pts.tolist() if poly_pts is not None else (list(crop) if crop else None),
            "roi_type": "polygon" if poly_pts is not None else ("rect" if crop else "full"),
            "roi_bounds": list(roi_bounds),
            "pad": pad,
            "detection_rate": det_count / total_to_track,
            "interpolated_frames": n_interpolated,
            "final_detection_rate": final_count / total_to_track,
        },
    }
