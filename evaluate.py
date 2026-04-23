import math
import pandas as pd
import numpy as np

def parse_prediction_string(pred_string):
    """
    Converts the string "id x y id x y" into a list of dictionaries.
    """
    if pd.isna(pred_string) or str(pred_string).strip() == "":
        return []
    
    parts = str(pred_string).split()
    markers = []
    
    # Iterate in steps of 3 (id, x, y)
    for i in range(0, len(parts), 3):
        markers.append({
            'id': int(parts[i]),
            'x': float(parts[i+1]),
            'y': float(parts[i+2])
        })
    return markers

def calculate_image_score(gt_str, pred_str, img_width, img_height, sigma=0.02, lam=1.0):
    """
    Calculates the score for a single image based on the assignment formula.
    """
    gt_list = parse_prediction_string(gt_str)
    pred_list = parse_prediction_string(pred_str)

    N_gt = len(gt_list)
    N_pred = len(pred_list)

    # Special Case - No Detections
    if N_gt == 0:
        return 1.0 if N_pred == 0 else 0.0

    diagonal = math.sqrt(img_width**2 + img_height**2)
    
    # Group predictions and ground truths by marker ID
    gt_by_id = {}
    for item in gt_list:
        gt_by_id.setdefault(item['id'], []).append((item['x'], item['y']))
        
    pred_by_id = {}
    for item in pred_list:
        pred_by_id.setdefault(item['id'], []).append((item['x'], item['y']))

    phi_sum = 0.0
    N_spam = 0

    # Matching Rule per Marker ID
    for k, preds in pred_by_id.items():
        if k not in gt_by_id:
            # If the predicted ID doesn't exist in ground truth, all are spam
            N_spam += len(preds)
            continue
        
        gts = gt_by_id[k]
        
        # Calculate minimum distance for each prediction to ANY ground truth of the same ID
        d_mins = []
        for px, py in preds:
            d_min = min(math.sqrt((px - gx)**2 + (py - gy)**2) for gx, gy in gts)
            d_mins.append(d_min)
        
        # Handle cases where there are more guesses than actual ground truth markers
        if len(preds) > len(gts):
            d_mins.sort()
            valid_dmins = d_mins[:len(gts)] # Keep only the ones with smallest error
            N_spam += (len(preds) - len(gts)) # The rest are penalized as spam
        else:
            valid_dmins = d_mins
            
        # Calculate Distance Score for valid matches
        for d in valid_dmins:
            d_norm = d / diagonal
            phi = math.exp(-(d_norm**2) / (2 * sigma**2))
            phi_sum += phi

    # Final Image Score Formula
    score_img = phi_sum / (N_gt + lam * N_spam)
    return score_img

def evaluate_dataset(train_csv_path, submission_csv_path, img_width, img_height):
    """
    Evaluates the entire dataset to return the final Kaggle mean score.
    """
    # Load the CSVs
    df_gt = pd.read_csv(train_csv_path)
    df_pred = pd.read_csv(submission_csv_path)

    # Merge them on image_id to compare side-by-side
    df_merged = pd.merge(df_gt, df_pred, on='image_id', how='left', suffixes=('_gt', '_pred'))
    df_merged['prediction_string_pred'] = df_merged['prediction_string_pred'].fillna("")

    total_score = 0.0
    M = len(df_merged)

    print(f"Evaluating {M} images...")

    for index, row in df_merged.iterrows():
        gt_str = row['prediction_string_gt']
        pred_str = row['prediction_string_pred']
        
        score = calculate_image_score(gt_str, pred_str, img_width, img_height)
        total_score += score

    # The final leaderboard score is the mean over all images
    final_score = total_score / M
    
    # Print results out of 100% for readability, though Kaggle scales it to 1.0
    print(f"Final Score: {final_score:.4f} ({(final_score*100):.2f}%)")
    return final_score

# --- Execution ---
if __name__ == "__main__":
    # Remember to check your image dimensions!
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 360

    evaluate_dataset("train.csv", "submission.csv", IMAGE_WIDTH, IMAGE_HEIGHT)