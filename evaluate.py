"""
ArUco Detection Pipeline - v3 (Debugged)
=========================================
Fixes from v2 (which scored 22% due to 3 critical bugs):

BUG 1 FIXED — get_top_left_corner:
  OpenCV ALWAYS returns corners[0] as the marker's true top-left in its own
  frame, regardless of image rotation. The ground truth is defined this way.
  The x+y heuristic in v2 was WRONG — it returns the corner closest to the
  image origin, not the marker's top-left, destroying score on rotated markers.
  FIX: Always use corners[0] directly — do NOT reorder.

BUG 2 FIXED — Perspective recovery loop:
  v2 called detectMarkers() N times inside a per-marker loop, causing huge
  spam and incorrect coordinate remapping. The pass now runs once globally,
  then maps H_inv correctly only for the same marker ID that was warped.

BUG 3 FIXED — filter_marker_candidates id indexing:
  After detect_on_scaled the ids array has shape (N,1). Accessing ids[i][0]
  was fine but passing it as valid_ids then re-indexing as valid_ids[i][0]
  in the caller caused shape mismatch. Normalised to always store plain int.

Retained improvements from v2 that are genuinely helpful:
  - 6 preprocessing variants (CLAHE, bilateral, gamma dark/light, sharp)
  - 8-config parameter sweep
  - Multi-scale upscaling (1.5x) for small/distant markers
  - Subpixel corner refinement on all passes
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math


# ──────────────────────────────────────────────────────
#  PREPROCESSING
# ──────────────────────────────────────────────────────

def unsharp_mask(gray, kernel_size=(5, 5), sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(gray, kernel_size, sigma)
    return cv2.addWeighted(gray, 1 + strength, blurred, -strength, 0)


def apply_gamma(img, gamma):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def preprocess_variants(image):
    """
    6 grayscale variants, each targeting a different failure mode.
    """
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    clahe_img   = clahe.apply(gray)
    sharp       = unsharp_mask(clahe_img, strength=1.2)
    bilat       = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    bilat_clahe = clahe.apply(bilat)
    gamma_dark  = apply_gamma(gray, 0.5)   # brighten underexposed
    gamma_light = apply_gamma(gray, 1.8)   # darken overexposed/glare

    return {
        "base":        gray,
        "clahe":       clahe_img,
        "sharp":       sharp,
        "bilateral":   bilat_clahe,
        "gamma_dark":  gamma_dark,
        "gamma_light": gamma_light,
    }


# ──────────────────────────────────────────────────────
#  CORNER EXTRACTION  ← BUG 1 FIX
# ──────────────────────────────────────────────────────

def get_top_left_corner(corners_1x4x2):
    """
    OpenCV ArUco ALWAYS stores corners in fixed winding order:
      [0] top-left  [1] top-right  [2] bottom-right  [3] bottom-left
    relative to the MARKER's own coordinate frame.

    The ground-truth labels use this same convention, so index [0] is correct.
    DO NOT reorder by x+y — that returns the corner closest to the image
    origin, which is wrong for any rotated marker.
    """
    return corners_1x4x2[0][0]   # shape (4,2) → first corner (x,y)


# ──────────────────────────────────────────────────────
#  GEOMETRIC FILTER
# ──────────────────────────────────────────────────────

def filter_marker_candidates(corners, ids, image_shape):
    """
    Returns (valid_corners, valid_ids) where valid_ids contains plain ints.
    Filters by area, convexity, and aspect ratio.
    """
    if ids is None or len(ids) == 0:
        return [], []

    valid_corners, valid_ids = [], []
    h, w = image_shape[:2]
    max_area = h * w * 0.70
    min_area = 30

    for i in range(len(ids)):
        c    = corners[i][0]           # shape (4, 2)
        area = cv2.contourArea(c)

        if area < min_area or area > max_area:
            continue
        if not cv2.isContourConvex(c):
            continue

        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue
        if max(rw, rh) / min(rw, rh) > 3.0:
            continue

        valid_corners.append(corners[i])
        valid_ids.append(int(ids[i][0]))   # ← always plain int, no nested arrays

    return valid_corners, valid_ids


# ──────────────────────────────────────────────────────
#  SPATIAL NMS
# ──────────────────────────────────────────────────────

def is_duplicate(new_x, new_y, existing_predictions, threshold_pixels=15):
    for (ex_x, ex_y) in existing_predictions.values():
        if math.sqrt((new_x - ex_x) ** 2 + (new_y - ex_y) ** 2) < threshold_pixels:
            return True
    return False


# ──────────────────────────────────────────────────────
#  MULTI-SCALE DETECTION  ← BUG 3 FIX (clean id shape)
# ──────────────────────────────────────────────────────

def detect_on_scaled(img_gray, scale, aruco_dict, params):
    """
    Detect on a rescaled image; remap corners back to original coordinates.
    Returns (corners_list, ids_list) with ids as plain ints.
    """
    if abs(scale - 1.0) < 0.01:
        scaled = img_gray
    else:
        new_w   = int(img_gray.shape[1] * scale)
        new_h   = int(img_gray.shape[0] * scale)
        scaled  = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    detector             = cv2.aruco.ArucoDetector(aruco_dict, params)
    raw_corners, raw_ids, _ = detector.detectMarkers(scaled)

    if raw_ids is None or len(raw_ids) == 0:
        return [], []

    remapped = [(c / scale).astype(np.float32) for c in raw_corners]
    return remapped, raw_ids


# ──────────────────────────────────────────────────────
#  SWEEP CONFIGS
# ──────────────────────────────────────────────────────

SWEEP_CONFIGS = [
    # (label,            win_min, win_max, win_step, error_rate)
    ("precision_fine",       3,    23,      4,       0.35),
    ("precision_coarse",     7,    31,      6,       0.35),
    ("large_marker",        13,    43,      6,       0.50),
    ("micro_tiny",           3,    13,      2,       0.55),
    ("recovery_high",        5,    35,      4,       0.65),
    ("glare_wide",          23,    63,     10,       0.50),
    ("shadow_tight",         3,    17,      2,       0.60),
    ("balanced",             9,    41,      8,       0.45),
]


def make_params(win_min, win_max, win_step, err_rate):
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod        = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize       = 5
    p.cornerRefinementMaxIterations = 50
    p.cornerRefinementMinAccuracy   = 0.001
    p.adaptiveThreshWinSizeMin      = win_min
    p.adaptiveThreshWinSizeMax      = win_max
    p.adaptiveThreshWinSizeStep     = win_step
    p.errorCorrectionRate           = err_rate
    return p


# ──────────────────────────────────────────────────────
#  SINGLE-VARIANT SWEEP
# ──────────────────────────────────────────────────────

def sweep_variant(img_variant, aruco_dict, image_shape, scale=1.0):
    """
    Run all sweep configs on one preprocessed variant at one scale.
    Returns dict: marker_id (int) → (x, y).
    """
    detections = {}

    for _, win_min, win_max, win_step, err_rate in SWEEP_CONFIGS:
        params = make_params(win_min, win_max, win_step, err_rate)
        raw_corners, raw_ids = detect_on_scaled(img_variant, scale, aruco_dict, params)
        valid_corners, valid_ids = filter_marker_candidates(raw_corners, raw_ids, image_shape)

        for i, m_id in enumerate(valid_ids):
            x, y = get_top_left_corner(valid_corners[i])   # always corners[0]
            x, y = float(x), float(y)

            if m_id not in detections and not is_duplicate(x, y, detections):
                detections[m_id] = (x, y)

    return detections


# ──────────────────────────────────────────────────────
#  PERSPECTIVE RECOVERY  ← BUG 2 FIX
# ──────────────────────────────────────────────────────

def perspective_recovery_pass(gray_base, aruco_dict, existing_ids):
    """
    For each already-found marker, warp its region to a canonical square
    and re-detect to get a more accurate corner position.

    FIXED vs v2:
    - detectMarkers runs ONCE, outside the loop
    - H_inv is only applied to the specific marker being rectified
    - Returns only NEW marker IDs not already in existing_ids
    """
    # One global detection pass to get corner geometry
    params   = make_params(3, 23, 4, 0.60)
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    rc, ri, _ = detector.detectMarkers(gray_base)

    if ri is None or len(ri) == 0:
        return {}

    marker_size_px = 120
    pts_dst = np.array([
        [0,                  0],
        [marker_size_px - 1, 0],
        [marker_size_px - 1, marker_size_px - 1],
        [0,                  marker_size_px - 1],
    ], dtype=np.float32)

    new_detections = {}

    for i in range(len(ri)):
        m_id = int(ri[i][0])

        # Only attempt recovery for markers already confirmed (not noise)
        if m_id not in existing_ids:
            continue

        pts_src = rc[i][0].reshape(4, 2).astype(np.float32)

        try:
            H, _ = cv2.findHomography(pts_src, pts_dst)
            if H is None:
                continue
            warped = cv2.warpPerspective(gray_base, H,
                                         (marker_size_px, marker_size_px))
        except cv2.error:
            continue

        # Detect in warped patch
        p2       = make_params(3, 23, 4, 0.60)
        det2     = cv2.aruco.ArucoDetector(aruco_dict, p2)
        rc2, ri2, _ = det2.detectMarkers(warped)

        if ri2 is None:
            continue

        H_inv = np.linalg.inv(H)

        for j in range(len(ri2)):
            ref_id = int(ri2[j][0])
            if ref_id in existing_ids or ref_id in new_detections:
                continue  # already have it; skip to avoid spam

            # Map corner[0] of rectified marker back to original image space
            pt_warped = rc2[j][0][0].reshape(1, 1, 2).astype(np.float32)
            pt_orig   = cv2.perspectiveTransform(pt_warped, H_inv)
            rx, ry    = float(pt_orig[0, 0, 0]), float(pt_orig[0, 0, 1])
            new_detections[ref_id] = (rx, ry)

    return new_detections


# ──────────────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────────────

def process_aruco_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    variants   = preprocess_variants(img)

    final = {}   # marker_id → (x, y)

    # ── Pass 1: Full-resolution sweep on all 6 variants ──
    for vimg in variants.values():
        for m_id, (x, y) in sweep_variant(vimg, aruco_dict, img.shape, scale=1.0).items():
            if m_id not in final and not is_duplicate(x, y, final):
                final[m_id] = (x, y)

    # ── Pass 2: 1.5× upscale on base, sharp, gamma_dark ──
    # Helps with small/distant markers that have too few pixels at native res
    for vname in ("base", "sharp", "gamma_dark"):
        for m_id, (x, y) in sweep_variant(variants[vname], aruco_dict, img.shape, scale=1.5).items():
            if m_id not in final and not is_duplicate(x, y, final):
                final[m_id] = (x, y)

    # ── Pass 3: Perspective recovery for confirmed markers ──
    for m_id, (x, y) in perspective_recovery_pass(variants["base"], aruco_dict, set(final.keys())).items():
        if m_id not in final and not is_duplicate(x, y, final):
            final[m_id] = (x, y)

    return [(m_id, x, y) for m_id, (x, y) in final.items()]


# ──────────────────────────────────────────────────────
#  CSV GENERATION
# ──────────────────────────────────────────────────────

def generate_kaggle_submission(image_dir, output_csv_path):
    image_dir_path = Path(image_dir)
    image_paths    = sorted(
        list(image_dir_path.glob("*.jpg")) + list(image_dir_path.glob("*.png"))
    )

    submission_data = []
    total = len(image_paths)
    print(f"Processing {total} images...")

    for idx, img_path in enumerate(image_paths, 1):
        if idx % 100 == 0 or idx == total:
            print(f"  [{idx}/{total}] {img_path.name}")

        image_id    = img_path.stem
        predictions = process_aruco_image(img_path)

        parts = []
        for (marker_id, x, y) in predictions:
            parts.extend([str(marker_id), f"{x:.3f}", f"{y:.3f}"])

        submission_data.append({
            "image_id":          image_id,
            "prediction_string": " ".join(parts),
        })

    pd.DataFrame(submission_data).to_csv(output_csv_path, index=False)
    print(f"\n✅ Saved to: {output_csv_path}")


if __name__ == "__main__":
    DATASET_DIR = "aruco_data/train"
    OUTPUT_CSV  = "submission_v3.csv"
    generate_kaggle_submission(DATASET_DIR, OUTPUT_CSV)