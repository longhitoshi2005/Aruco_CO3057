import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math

def adjust_gamma(image, gamma=1.2):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)		

def auto_gamma_correction(image):
    """
    Automatically determine gamma value based on image brightness
    
    Parameters:
    image: Input image
    
    Returns:
    Automatically gamma-corrected image
    """
    # Convert to grayscale and calculate mean brightness
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    
    # Calculate gamma based on mean brightness
    # Lower brightness gets lower gamma (more correction)
    target_brightness = 50  # Mid-range brightness
    gamma = np.log(target_brightness / 255) / np.log(mean_brightness / 255)
    
    # Ensure gamma is within reasonable bounds
    gamma = np.clip(gamma, 0.7, 2.6)
    
    return adjust_gamma(image, gamma)

def preprocess_flat(image):
    """
    Applies very light CLAHE to balance the image without 
    blowing out the background textures.
    """
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    gray  = cv2.cvtColor(auto_gamma_correction(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    norm = cv2.divide(gray, blur, scale=255)

    #thresh = cv.adaptiveThreshold(norm, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 6)
    _,thresh = cv2.threshold(norm, 0 , 255, cv2.THRESH_OTSU)
    filtered = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (2,2))

    return filtered

def filter_marker_candidates(corners, ids, image_shape):
    """
    The Ruthless Bouncer: Area, Convexity, and Aspect Ratio.
    """
    if ids is None:
        return [], []
        
    valid_corners = []
    valid_ids = []
    
    img_height, img_width = image_shape[:2]
    max_area = (img_height * img_width) * 0.7  
    min_area = 45  # Slightly lowered to catch distant markers

    for i in range(len(ids)):
        c = corners[i][0]
        area = cv2.contourArea(c)
        
        if area < min_area or area > max_area: continue
        if not cv2.isContourConvex(c): continue

        # Aspect Ratio Check 
        rect = cv2.minAreaRect(c)
        width, height = rect[1]
        
        if width == 0 or height == 0: continue
            
        aspect_ratio = max(width, height) / min(width, height)
        # Tighter square enforcement to kill False Positives from the sweeps
        if aspect_ratio > 2.5: continue 
            
        valid_corners.append(corners[i])
        valid_ids.append(ids[i][0]) 
        
    return valid_corners, valid_ids

def is_duplicate(new_x, new_y, existing_predictions, threshold_pixels=15):
    """
    Spatial NMS: Drops duplicates found across different sweep passes.
    """
    for m_id, (ex_x, ex_y) in existing_predictions.items():
        dist = math.sqrt((new_x - ex_x)**2 + (new_y - ex_y)**2)
        if dist < threshold_pixels:
            return True
    return False

def process_aruco_image_sweep(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return []

    gray_base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_flat = preprocess_flat(img)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    final_predictions = {}

    # Define our sweep configurations (Pass Type, Image, MinWin, MaxWin, Step, ErrorRate)
    sweep_configs = [
        ("Base_Standard", gray_base, 3, 23, 4, 0.4), # High Precision
        ("Base_Wide", gray_base, 13, 43, 6, 0.5),    # Large Marker Focus
        ("Flat_Micro", gray_flat, 3, 15, 2, 0.6),    # Deep Shadow/Tiny Marker Focus
        ("Flat_Macro", gray_flat, 23, 53, 10, 0.5)   # Glare Recovery
    ]

    for name, img_variant, win_min, win_max, win_step, err_rate in sweep_configs:
        params = cv2.aruco.DetectorParameters()
        
        # Elite Subpixel Refinement across all passes
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 40
        params.cornerRefinementMinAccuracy = 0.001
        
        # Sweep Parameters
        params.adaptiveThreshWinSizeMin = win_min
        params.adaptiveThreshWinSizeMax = win_max
        params.adaptiveThreshWinSizeStep = win_step
        params.errorCorrectionRate = err_rate
        
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        raw_corners, raw_ids, _ = detector.detectMarkers(img_variant)

        # Apply the geometric filter
        valid_corners, valid_ids = filter_marker_candidates(raw_corners, raw_ids, img.shape)

        for i in range(len(valid_ids)):
            m_id = int(valid_ids[i])
            x, y = valid_corners[i][0][0]
            
            # Spatial Deduplication across sweeps
            if m_id not in final_predictions and not is_duplicate(x, y, final_predictions):
                final_predictions[m_id] = (x, y)

    return [(m_id, x, y) for m_id, (x, y) in final_predictions.items()]

def generate_kaggle_submission(image_dir, output_csv_path):
    image_dir_path = Path(image_dir)
    image_paths = list(image_dir_path.glob("*.jpg")) + list(image_dir_path.glob("*.png"))
    
    submission_data = []
    print(f"Running Multi-Threshold Sweep on {len(image_paths)} images...")
    
    for img_path in image_paths:
        image_id = img_path.stem 
        predictions = process_aruco_image_sweep(img_path)
        
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
    DATASET_DIR = "aruco_data/images/train"
    OUTPUT_CSV = "submission.csv" 
    generate_kaggle_submission(DATASET_DIR, OUTPUT_CSV)
