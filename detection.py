import cv2
import numpy as np

def preprocess_image(image):
    """
    Enhances the image to handle shadows and noise while preserving edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 2. Bilateral Filter instead of Sharpening
    # This specifically targets compression artifacts and background noise
    # without blurring the crisp edges of the ArUco boundaries.
    filtered = cv2.bilateralFilter(enhanced_gray, d=5, sigmaColor=50, sigmaSpace=50)

    return filtered

def filter_marker_candidates(corners, ids, image_shape):
    """
    Mathematically filters out 'spam' detections based on geometric properties.
    This heavily minimizes the N_spam penalty in the evaluation metric.
    """
    if ids is None:
        return None, None
        
    valid_corners = []
    valid_ids = []
    
    img_height, img_width = image_shape[:2]
    max_area = (img_height * img_width) * 0.8  # Marker shouldn't take up 80% of the image
    min_area = 50  # Must be large enough to be a reliable marker

    for i in range(len(ids)):
        c = corners[i][0]
        # Calculate the area of the detected quadrilateral
        area = cv2.contourArea(c)
        
        # 1. Area Check
        if area < min_area or area > max_area:
            continue
            
        # 2. Convexity Check
        # Real ArUco markers on a flat plane must form convex shapes.
        # If the contour folds in on itself, it's background noise.
        if not cv2.isContourConvex(c):
            continue
            
        valid_corners.append(corners[i])
        valid_ids.append(ids[i])
        
    if not valid_ids:
        return None, None
        
    return tuple(valid_corners), np.array(valid_ids)

def process_aruco_image(image_path):
    """
    Main pipeline function: Loads, preprocesses, detects, filters, and extracts.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not load image at {image_path}")
        return []

    preprocessed_img = preprocess_image(img)

    # Setup Dictionary and Parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    aruco_params = cv2.aruco.DetectorParameters()

    # --- OPTIMIZED PARAMETERS ---
    # 1. Expand the Adaptive Thresholding search space
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 35 # Increased
    aruco_params.adaptiveThreshWinSizeStep = 4 # Finer steps
    
    # 2. Reject tiny specks early at the contour level
    aruco_params.minMarkerPerimeterRate = 0.03 
    
    # 3. Enforce highly accurate Subpixel refinement for the distance metric
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_params.cornerRefinementWinSize = 5
    aruco_params.cornerRefinementMaxIterations = 40
    aruco_params.cornerRefinementMinAccuracy = 0.01

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    raw_corners, raw_ids, _ = detector.detectMarkers(preprocessed_img)

    # Apply our custom Spam Filter
    corners, ids = filter_marker_candidates(raw_corners, raw_ids, img.shape)

    predictions = []

    if ids is not None:
        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            
            # Extract the top-left corner (index 0)
            top_left_x, top_left_y = corners[i][0][0]

            # Append as a triplet
            predictions.append((marker_id, top_left_x, top_left_y))

    return predictions

if __name__ == "__main__":
    # Test this on a single image locally
    test_image = "aruco_data/train/000000000089.jpg" 
    results = process_aruco_image(test_image)
    print("Detected Markers:", results)