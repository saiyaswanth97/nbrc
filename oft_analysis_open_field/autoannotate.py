"""Propagate hand-annotated polygon/boundary corners to the remaining videos.

For each unannotated video, a background image (median of sampled frames, so the
animal disappears) is registered against the background of every annotated
reference video with SIFT + RANSAC homography, using only reference features in
and around the arena. The corners of the best-matching reference (most inliers)
are mapped through the homography.
"""

import os
import json

import cv2
import numpy as np

from .tracking import build_background
from .gui import order_corners, _parse, _fmt

SCALE = 0.5  # register on half-resolution images


def video_background(video_path, n_samples=30):
    cap = cv2.VideoCapture(video_path)
    try:
        return build_background(cap, n_samples=n_samples)
    finally:
        cap.release()


def _features(sift, gray, mask=None):
    small = cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    m = None if mask is None else cv2.resize(mask, (small.shape[1], small.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
    return sift.detectAndCompute(small, m)


def _arena_mask(polygon, shape, margin=80):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(polygon)], 255)
    return cv2.dilate(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))


def match_homography(ref_feats, tgt_feats):
    """Return (H mapping ref -> target in full-res pixels, inlier count)."""
    (kr, dr), (kt, dt) = ref_feats, tgt_feats
    if dr is None or dt is None or len(kr) < 8 or len(kt) < 8:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dr, dt, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, 0
    src = np.float32([kr[m.queryIdx].pt for m in good]) / SCALE
    dst = np.float32([kt[m.trainIdx].pt for m in good]) / SCALE
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return None, 0
    return H, int(inl.sum())


def _warp(pts, H):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2).tolist()


def propagate(refs, tgt_bg, sift):
    """refs: list of dicts with name, polygon, boundary, feats. Returns best estimate."""
    tgt_feats = _features(sift, tgt_bg)
    best = None
    for r in refs:
        H, n = match_homography(r["feats"], tgt_feats)
        if H is not None and (best is None or n > best["inliers"]):
            best = {"ref": r["name"], "inliers": n,
                    "polygon": order_corners(_warp(r["polygon"], H)),
                    "boundary": order_corners(_warp(r["boundary"], H))}
    return best


def _load_refs(config, video_dir, sift, bg):
    refs = []
    for v in config.get("videos", []):
        if v.get("annotated") and not v.get("auto"):
            poly, bnd = _parse(v["polygon"]), _parse(v["boundary"])
            g = bg(video_dir, v)
            refs.append({"name": os.path.splitext(v["file"])[0], "polygon": poly, "boundary": bnd,
                         "feats": _features(sift, g, _arena_mask(poly, g.shape))})
    return refs


def auto_annotate(config_path, min_inliers=50, evaluate=False, preview=True, ref_configs=()):
    """ref_configs: other configs whose hand-annotated videos are also used as references."""
    with open(config_path) as f:
        config = json.load(f)
    video_dir = config.get("video_dir", os.path.dirname(os.path.abspath(config_path)))
    videos = config.get("videos", [])
    sift = cv2.SIFT_create(nfeatures=4000)

    bgs = {}

    def bg(vdir, v):
        key = os.path.join(vdir, v["file"])
        if key not in bgs:
            bgs[key] = video_background(key)
        return bgs[key]

    refs = _load_refs(config, video_dir, sift, bg)
    for rc in ref_configs:
        with open(rc) as f:
            other = json.load(f)
        other_dir = other.get("video_dir", os.path.dirname(os.path.abspath(rc)))
        refs += _load_refs(other, other_dir, sift, bg)
    if not refs:
        raise ValueError("No hand-annotated videos in config — annotate a few first.")
    print(f"{len(refs)} hand-annotated reference video(s).")

    if evaluate:
        print("\nLeave-one-out check (max corner error, px):")
        for r in refs:
            others = [o for o in refs if o is not r]
            f = next((v["file"] for v in videos if os.path.splitext(v["file"])[0] == r["name"]
                      and not v.get("auto") and v.get("annotated")), None)
            if f is None:
                continue  # reference from another config
            est = propagate(others, bgs[os.path.join(video_dir, f)], sift)
            if est is None:
                print(f"  {r['name']}: no match")
                continue
            err = max(np.abs(np.float32(est[k]) - np.float32(r[k])).max() for k in ("polygon", "boundary"))
            print(f"  {r['name']}: {err:5.1f}px  (from {est['ref']}, {est['inliers']} inliers)")

    out_dir = os.path.join(video_dir, "samples", "grid_check")
    os.makedirs(out_dir, exist_ok=True)
    flagged = []
    todo = [v for v in videos if not v.get("annotated") or v.get("auto")]
    print(f"\nAuto-annotating {len(todo)} video(s):")
    for v in todo:
        name = os.path.splitext(v["file"])[0]
        path = os.path.join(video_dir, v["file"])
        if not os.path.exists(path):
            print(f"  SKIP {name} (not found)")
            continue
        g = bg(video_dir, v)
        est = propagate(refs, g, sift)
        if est is None or est["inliers"] < min_inliers:
            flagged.append(name)
            print(f"  {name}: LOW CONFIDENCE ({est['inliers'] if est else 0} inliers) — not saved")
            continue
        v["polygon"], v["boundary"] = _fmt(est["polygon"]), _fmt(est["boundary"])
        v["annotated"], v["auto"] = True, True
        print(f"  {name}: from {est['ref']} ({est['inliers']} inliers)")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    if preview:
        from .plotting import draw_grid
        rows, cols = [int(x) for x in config.get("grid", "4x4").split("x")]
        for v in videos:
            key = os.path.join(video_dir, v["file"])
            if not v.get("annotated") or key not in bgs:
                continue
            img = cv2.cvtColor(bgs[key], cv2.COLOR_GRAY2BGR)
            draw_grid(img, _parse(v["boundary"]), rows, cols)
            cv2.polylines(img, [np.int32(_parse(v["polygon"]))], True, (0, 0, 255), 2)
            tag = "auto" if v.get("auto") else "manual"
            cv2.putText(img, f"{v['file']} ({tag})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(out_dir, os.path.splitext(v["file"])[0] + ".png"), img)
        print(f"\nPreviews: {out_dir}/")

    if flagged:
        print(f"Needs manual annotation: {','.join(flagged)}")
    return flagged
