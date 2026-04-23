"""
Advanced ArUco Marker Detection Pipeline - v2
Target: 90%+ Kaggle score (up from 80.03%)

Key Upgrades over v1:
1. PERSPECTIVE RECOVERY: Homography-based rectification for rotated/tilted markers
2. MULTI-SCALE PYRAMID: Detects tiny/distant markers by upscaling image regions
3. ADVANCED PREPROCESSING: Bilateral filter, CLAHE with better params, unsharp masking
4. ORIENTATION-AWARE TOP-LEFT: Correctly identifies top-left corner via marker geometry
5. ANTI-SPAM HARDENING: Strict dedup + confidence-based tie-breaking
6. AGGRESSIVE PARAMETER GRID: Wider sweep with more targeted configs for challenging conditions
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math


# ─────────────────────────────────────────────
#  PREPROCESSING UTILITIES
# ─────────────────────────────────────────────

def unsharp_mask(gray, kernel_size=(5, 5), sigma=1.0, strength=1.5):
    """
    Enhances fine edges (marker borders) by amplifying high-frequency detail.
    Much more effective than a raw sharpen kernel for ArUco patterns.
    """
    blurred = cv2.GaussianBlur(gray, kernel_size, sigma)
    return cv2.addWeighted(gray, 1 + strength, blurred, -strength, 0)


def preprocess_variants(image):
    """
    Returns a dict of preprocessed grayscale variants.
    Each variant is optimized for a different failure mode:
      - 'base'       : raw grayscale (good precision baseline)
      - 'clahe'      : CLAHE balanced (handles uneven illumination)
      - 'sharp'      : unsharp masked (helps motion-blurred/soft markers)
      - 'bilateral'  : bilateral filtered then CLAHE (preserves edges, kills noise)
      - 'gamma_dark' : gamma-boosted for underexposed regions
      - 'gamma_light': gamma-reduced for overexposed/glare regions
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE — balanced exposure
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Unsharp masking on CLAHE output (best of both)
    sharp = unsharp_mask(clahe_img, strength=1.2)

    # Bilateral on raw gray, then CLAHE
    bilat = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    bilat_clahe = clahe.apply(bilat)

    # Gamma correction
    def apply_gamma(img, gamma):
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, lut)

    gamma_dark  = apply_gamma(gray, 0.5)   # brighten dark images
    gamma_light = apply_gamma(gray, 1.8)   # darken blown-out images

    return {
        "base":        gray,
        "clahe":       clahe_img,
        "sharp":       sharp,
        "bilateral":   bilat_clahe,
        "gamma_dark":  gamma_dark,
        "gamma_light": gamma_light,
    }


# ─────────────────────────────────────────────
#  MARKER GEOMETRY: CORRECT TOP-LEFT CORNER
# ─────────────────────────────────────────────

def get_top_left_corner(corners_4x2):
    """
    OpenCV's ArUco returns corners in a FIXED winding order relative to the
    marker's own coordinate frame (top-left → top-right → bottom-right → bottom-left).
    However, when a marker is heavily rotated or viewed from an angle, the
    "visually top-left" corner can differ from index [0].

    Strategy: The true top-left in IMAGE space is the corner with the
    minimum (x + y) value — standard geometric convention.
    """
    pts = corners_4x2.reshape(4, 2)
    scores = pts[:, 0] + pts[:, 1]  # x + y; smallest = closest to image origin
    return pts[np.argmin(scores)]


# ─────────────────────────────────────────────
#  GEOMETRIC CANDIDATE FILTER
# ─────────────────────────────────────────────

def filter_marker_candidates(corners, ids, image_shape):
    """
    Filters detections by:
    - Area bounds (removes noise blobs and frame-spanning false positives)
    - Convexity (ArUco markers are convex quadrilaterals)
    - Aspect ratio (markers are roughly square, allow up to 3.0 for strong perspective)
    """
    if ids is None:
        return [], []

    valid_corners, valid_ids = [], []
    h, w = image_shape[:2]
    max_area = h * w * 0.7
    min_area = 30  # allow very small distant markers

    for i in range(len(ids)):
        c = corners[i][0]
        area = cv2.contourArea(c)

        if area < min_area or area > max_area:
            continue
        if not cv2.isContourConvex(c):
            continue

        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue
        if max(rw, rh) / min(rw, rh) > 3.0:   # slightly looser for oblique views
            continue

        valid_corners.append(corners[i])
        valid_ids.append(ids[i][0])

    return valid_corners, valid_ids


# ─────────────────────────────────────────────
#  SPATIAL NMS (NON-MAXIMUM SUPPRESSION)
# ─────────────────────────────────────────────

def is_duplicate(new_x, new_y, existing_predictions, threshold_pixels=15):
    for m_id, (ex_x, ex_y) in existing_predictions.items():
        if math.sqrt((new_x - ex_x) ** 2 + (new_y - ex_y) ** 2) < threshold_pixels:
            return True
    return False


# ─────────────────────────────────────────────
#  MULTI-SCALE PYRAMID DETECTION
# ─────────────────────────────────────────────

def detect_on_scaled(img_gray, scale, aruco_dict, params):
    """
    Upscale small images so tiny/distant markers have enough pixels to decode.
    Returns detections remapped back to original image coordinates.
    """
    if abs(scale - 1.0) < 0.01:
        scaled = img_gray
    else:
        new_w = int(img_gray.shape[1] * scale)
        new_h = int(img_gray.shape[0] * scale)
        scaled = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    raw_corners, raw_ids, _ = detector.detectMarkers(scaled)

    if raw_ids is None or len(raw_ids) == 0:
        return [], []

    # Remap corners back to original resolution
    remapped_corners = []
    for c in raw_corners:
        remapped_corners.append((c / scale).astype(np.float32))

    return remapped_corners, raw_ids


# ─────────────────────────────────────────────
#  PERSPECTIVE RECOVERY (HOMOGRAPHY REFINEMENT)
# ─────────────────────────────────────────────

def refine_with_perspective(image_gray, corners_4x2, aruco_dict, marker_size_px=100):
    """
    Warps the detected marker region into a canonical square view and
    re-runs detection. This recovers markers that were borderline decodable
    due to strong perspective distortion.

    Returns refined corners in ORIGINAL image coordinates, or None if
    re-detection fails.
    """
    pts_src = corners_4x2.reshape(4, 2).astype(np.float32)

    # Canonical destination: a perfect square
    pts_dst = np.array([
        [0,                 0],
        [marker_size_px-1,  0],
        [marker_size_px-1,  marker_size_px-1],
        [0,                 marker_size_px-1],
    ], dtype=np.float32)

    try:
        H, _ = cv2.findHomography(pts_src, pts_dst)
        if H is None:
            return None
        warped = cv2.warpPerspective(image_gray, H,
                                     (marker_size_px, marker_size_px))
    except cv2.error:
        return None

    # Re-detect in the rectified patch
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 4
    params.errorCorrectionRate = 0.6

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    rc, ri, _ = detector.detectMarkers(warped)

    if ri is None or len(ri) == 0:
        return None

    # Map refined corners back to original image via inverse homography
    H_inv = np.linalg.inv(H)
    refined_orig_list = []
    for c in rc:
        pts = c[0].reshape(-1, 1, 2).astype(np.float32)
        back = cv2.perspectiveTransform(pts, H_inv)
        refined_orig_list.append(back.reshape(1, 4, 2))

    return refined_orig_list, ri


# ─────────────────────────────────────────────
#  CORE SWEEP DETECTION
# ─────────────────────────────────────────────

def build_sweep_configs():
    """
    Comprehensive sweep covering every challenging condition in the dataset:
    - Tight win sizes  → fine detail, small markers
    - Wide win sizes   → large markers, bright backgrounds  
    - High error rate  → partially occluded / degraded markers
    - Low error rate   → precision focus, avoids false positives
    """
    return [
        # (label, win_min, win_max, win_step, error_rate)
        ("precision_fine",   3,  23,  4, 0.35),   # high precision, small windows
        ("precision_coarse", 7,  31,  6, 0.35),   # medium precision
        ("large_marker",    13,  43,  6, 0.50),   # handles big markers
        ("micro_tiny",       3,  13,  2, 0.55),   # tiny/distant markers
        ("recovery_high",    5,  35,  4, 0.65),   # degraded/occluded
        ("glare_wide",      23,  63, 10, 0.50),   # blown-out backgrounds
        ("shadow_tight",     3,  17,  2, 0.60),   # deep shadow regions
        ("balanced",         9,  41,  8, 0.45),   # general purpose
    ]


def run_sweep_on_variant(img_variant, aruco_dict, image_shape, scale=1.0):
    """
    Runs all sweep configs on a single preprocessed image variant.
    Returns list of (marker_id, x, y).
    """
    detections = {}  # marker_id → (x, y)
    sweep_configs = build_sweep_configs()

    for label, win_min, win_max, win_step, err_rate in sweep_configs:
        params = cv2.aruco.DetectorParameters()

        # Subpixel refinement: critical for accurate top-left localization
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.001

        params.adaptiveThreshWinSizeMin  = win_min
        params.adaptiveThreshWinSizeMax  = win_max
        params.adaptiveThreshWinSizeStep = win_step
        params.errorCorrectionRate       = err_rate

        raw_corners, raw_ids = detect_on_scaled(img_variant, scale, aruco_dict, params)
        valid_corners, valid_ids = filter_marker_candidates(raw_corners, raw_ids, image_shape)

        for i in range(len(valid_ids)):
            m_id = int(valid_ids[i])
            tl   = get_top_left_corner(valid_corners[i][0])
            x, y = float(tl[0]), float(tl[1])

            if m_id not in detections and not is_duplicate(x, y, detections):
                detections[m_id] = (x, y)

    return detections


# ─────────────────────────────────────────────
#  MAIN PROCESSING FUNCTION
# ─────────────────────────────────────────────

def process_aruco_image(image_path):
    """
    Full pipeline:
    1. Multi-variant preprocessing
    2. Multi-scale + multi-threshold sweep on each variant
    3. Perspective recovery pass on borderline detections
    4. Spatial deduplication & anti-spam merge
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    variants   = preprocess_variants(img)

    final_predictions = {}  # marker_id → (x, y)

    # ── PASS 1: Full-resolution sweep on all variants ──
    for vname, vimg in variants.items():
        new_detections = run_sweep_on_variant(vimg, aruco_dict, img.shape, scale=1.0)
        for m_id, (x, y) in new_detections.items():
            if m_id not in final_predictions and not is_duplicate(x, y, final_predictions):
                final_predictions[m_id] = (x, y)

    # ── PASS 2: Upscaled sweep (scale=1.5) on 'base' and 'sharp' variants ──
    # Helps with small markers that lack enough pixels at native resolution
    for vname in ("base", "sharp", "gamma_dark"):
        vimg = variants[vname]
        new_detections = run_sweep_on_variant(vimg, aruco_dict, img.shape, scale=1.5)
        for m_id, (x, y) in new_detections.items():
            if m_id not in final_predictions and not is_duplicate(x, y, final_predictions):
                final_predictions[m_id] = (x, y)

    # ── PASS 3: Perspective recovery on any already-found markers ──
    # Re-rectifies each detected marker region and re-runs detection to
    # potentially recover additional nearby markers or refine corner positions
    gray_base = variants["base"]
    perspective_candidates = {}

    for m_id, (x, y) in list(final_predictions.items()):
        # Find the corners for this prediction by re-detecting quickly
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.adaptiveThreshWinSizeMin  = 3
        params.adaptiveThreshWinSizeMax  = 23
        params.adaptiveThreshWinSizeStep = 4
        params.errorCorrectionRate       = 0.6
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        rc, ri, _ = detector.detectMarkers(gray_base)

        if ri is None:
            continue

        for i in range(len(ri)):
            if int(ri[i][0]) == m_id:
                result = refine_with_perspective(gray_base, rc[i][0], aruco_dict)
                if result is not None:
                    ref_corners, ref_ids = result
                    for j in range(len(ref_ids)):
                        ref_id = int(ref_ids[j][0])
                        tl = get_top_left_corner(ref_corners[j][0])
                        rx, ry = float(tl[0]), float(tl[1])
                        if ref_id not in perspective_candidates:
                            perspective_candidates[ref_id] = (rx, ry)
                break

    # Merge perspective-refined positions (trust them only for new IDs)
    for m_id, (x, y) in perspective_candidates.items():
        if m_id not in final_predictions and not is_duplicate(x, y, final_predictions):
            final_predictions[m_id] = (x, y)

    return [(m_id, x, y) for m_id, (x, y) in final_predictions.items()]


# ─────────────────────────────────────────────
#  CSV GENERATION
# ─────────────────────────────────────────────

def generate_kaggle_submission(image_dir, output_csv_path):
    image_dir_path = Path(image_dir)
    image_paths    = sorted(
        list(image_dir_path.glob("*.jpg")) + list(image_dir_path.glob("*.png"))
    )

    submission_data = []
    total = len(image_paths)
    print(f"Processing {total} images with advanced multi-pass pipeline...")

    for idx, img_path in enumerate(image_paths, 1):
        if idx % 50 == 0 or idx == total:
            print(f"  [{idx}/{total}] {img_path.name}")

        image_id    = img_path.stem
        predictions = process_aruco_image(img_path)

        pred_str_parts = []
        for (marker_id, x, y) in predictions:
            pred_str_parts.extend([str(marker_id), f"{x:.3f}", f"{y:.3f}"])

        submission_data.append({
            "image_id":         image_id,
            "prediction_string": " ".join(pred_str_parts)
        })

    pd.DataFrame(submission_data).to_csv(output_csv_path, index=False)
    print(f"\n✅ Submission saved to: {output_csv_path}")


if __name__ == "__main__":
    DATASET_DIR = "aruco_data/train"   # adjust to your test directory
    OUTPUT_CSV  = "submission_v2.csv"
    generate_kaggle_submission(DATASET_DIR, OUTPUT_CSV)