import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math

def preprocess_rescue(image):
    """
    Pass 2 Filter: Hunts for markers hidden in shadows/noise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    return filtered

def filter_marker_candidates(corners, ids, image_shape):
    """
    The Bouncer: Eliminates background noise mimicking ArUco markers.
    """
    if ids is None:
        return [], []
        
    valid_corners = []
    valid_ids = []
    
    img_height, img_width = image_shape[:2]
    max_area = (img_height * img_width) * 0.7  # Reject massive false positives
    min_area = 50  # Reject tiny specks

    for i in range(len(ids)):
        c = corners[i][0]
        area = cv2.contourArea(c)
        
        if area < min_area or area > max_area:
            continue
            
        if not cv2.isContourConvex(c):
            continue

        # Aspect Ratio Check (Filters out long rectangles like windows)
        rect = cv2.minAreaRect(c)
        width, height = rect[1]
        
        if width == 0 or height == 0:
            continue
            
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 3.0: # Strict square enforcement
            continue
            
        valid_corners.append(corners[i])
        valid_ids.append(ids[i][0]) # Extract the integer ID cleanly
        
    return valid_corners, valid_ids

def is_duplicate(new_x, new_y, existing_predictions, threshold_pixels=15):
    """
    Spatial NMS: Prevents the N_spam penalty from duplicate detections.
    """
    for m_id, (ex_x, ex_y) in existing_predictions.items():
        dist = math.sqrt((new_x - ex_x)**2 + (new_y - ex_y)**2)
        if dist < threshold_pixels:
            return True
    return False

def process_aruco_image_waterfall(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return []

    gray_base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rescue = preprocess_rescue(img)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    final_predictions = {}

    # --- PASS 1: HIGH PRECISION (Clean Image) ---
    params_p1 = cv2.aruco.DetectorParameters()
    params_p1.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params_p1.errorCorrectionRate = 0.5 
    
    detector_p1 = cv2.aruco.ArucoDetector(aruco_dict, params_p1)
    raw_corners_p1, raw_ids_p1, _ = detector_p1.detectMarkers(gray_base)

    # Apply strict geometric filter
    valid_corners_p1, valid_ids_p1 = filter_marker_candidates(raw_corners_p1, raw_ids_p1, img.shape)

    for i in range(len(valid_ids_p1)):
        m_id = int(valid_ids_p1[i])
        x, y = valid_corners_p1[i][0][0]
        final_predictions[m_id] = (x, y)

    # --- PASS 2: HIGH RECALL RESCUE (Enhanced Image) ---
    params_p2 = cv2.aruco.DetectorParameters()
    params_p2.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    # Wider search for tough lighting
    params_p2.adaptiveThreshWinSizeMin = 3
    params_p2.adaptiveThreshWinSizeMax = 35 
    params_p2.adaptiveThreshWinSizeStep = 4
    params_p2.errorCorrectionRate = 0.6 
    
    detector_p2 = cv2.aruco.ArucoDetector(aruco_dict, params_p2)
    raw_corners_p2, raw_ids_p2, _ = detector_p2.detectMarkers(gray_rescue)

    # Apply strict geometric filter
    valid_corners_p2, valid_ids_p2 = filter_marker_candidates(raw_corners_p2, raw_ids_p2, img.shape)

    for i in range(len(valid_ids_p2)):
        m_id = int(valid_ids_p2[i])
        x, y = valid_corners_p2[i][0][0]
        
        # Spatial Deduplication
        if m_id not in final_predictions and not is_duplicate(x, y, final_predictions):
            final_predictions[m_id] = (x, y)

    return [(m_id, x, y) for m_id, (x, y) in final_predictions.items()]

def generate_kaggle_submission(image_dir, output_csv_path):
    image_dir_path = Path(image_dir)
    image_paths = list(image_dir_path.glob("*.jpg")) + list(image_dir_path.glob("*.png"))
    
    submission_data = []
    print(f"Running Golden Classic Pipeline on {len(image_paths)} images...")
    
    for img_path in image_paths:
        image_id = img_path.stem 
        predictions = process_aruco_image_waterfall(img_path)
        
        pred_str_parts = []
        for (marker_id, x, y) in predictions:
            pred_str_parts.extend([str(marker_id), f"{x:.3f}", f"{y:.3f}"])
            
        submission_data.append({
            "image_id": image_id,
            "prediction_string": " ".join(pred_str_parts)
        })

    pd.DataFrame(submission_data).to_csv(output_csv_path, index=False)
    print(f"Success! Saved to: {output_csv_path}")

if __name__ == "__main__":
    DATASET_DIR = "aruco_data/train"
    OUTPUT_CSV = "submission.csv" 
    generate_kaggle_submission(DATASET_DIR, OUTPUT_CSV)