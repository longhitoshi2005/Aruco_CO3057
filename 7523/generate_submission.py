import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math

def preprocess_rescue(image):
    """
    The Rescue Filter: Used only for finding markers hidden in dark/noisy areas.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    return filtered

def is_duplicate(new_x, new_y, existing_predictions, threshold_pixels=20):
    """
    Spatial Deduplication: Checks if a newly found corner is too close 
    to one we already found in a previous pass.
    """
    for m_id, (ex_x, ex_y) in existing_predictions.items():
        dist = math.sqrt((new_x - ex_x)**2 + (new_y - ex_y)**2)
        if dist < threshold_pixels:
            return True
    return False

def process_aruco_image_waterfall(image_path):
    """
    The Two-Pass Waterfall Pipeline for maximum Classical CV performance.
    """
    img = cv2.imread(str(image_path))
    if img is None: return []

    gray_base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rescue = preprocess_rescue(img)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    
    # --- PASS 1: HIGH PRECISION (Standard Image) ---
    params_p1 = cv2.aruco.DetectorParameters()
    params_p1.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params_p1.errorCorrectionRate = 0.5 # Slightly strict to avoid fakes
    
    detector_p1 = cv2.aruco.ArucoDetector(aruco_dict, params_p1)
    corners_p1, ids_p1, _ = detector_p1.detectMarkers(gray_base)

    # Dictionary mapping ID -> (x, y)
    final_predictions = {}

    if ids_p1 is not None:
        for i in range(len(ids_p1)):
            m_id = int(ids_p1[i][0])
            x, y = corners_p1[i][0][0]
            # Area check to filter out tiny noise specs
            if cv2.contourArea(corners_p1[i][0]) > 40: 
                final_predictions[m_id] = (x, y)

    # --- PASS 2: HIGH RECALL RESCUE (Enhanced Image) ---
    params_p2 = cv2.aruco.DetectorParameters()
    params_p2.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    # Widen search space for tough lighting
    params_p2.adaptiveThreshWinSizeMin = 3
    params_p2.adaptiveThreshWinSizeMax = 35 
    params_p2.adaptiveThreshWinSizeStep = 4
    params_p2.errorCorrectionRate = 0.6 # Normal forgiveness
    
    detector_p2 = cv2.aruco.ArucoDetector(aruco_dict, params_p2)
    corners_p2, ids_p2, _ = detector_p2.detectMarkers(gray_rescue)

    if ids_p2 is not None:
        for i in range(len(ids_p2)):
            m_id = int(ids_p2[i][0])
            x, y = corners_p2[i][0][0]
            
            # Area check
            if cv2.contourArea(corners_p2[i][0]) < 40:
                continue

            # Spatial Deduplication: Only add if we didn't find a marker 
            # in this exact spot during Pass 1.
            if m_id not in final_predictions and not is_duplicate(x, y, final_predictions):
                final_predictions[m_id] = (x, y)

    # Convert to expected triplet list format
    return [(m_id, x, y) for m_id, (x, y) in final_predictions.items()]

def generate_kaggle_submission(image_dir, output_csv_path):
    image_dir_path = Path(image_dir)
    image_paths = list(image_dir_path.glob("*.jpg")) + list(image_dir_path.glob("*.png"))
    
    submission_data = []
    print(f"Running Waterfall Pipeline on {len(image_paths)} images...")
    
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