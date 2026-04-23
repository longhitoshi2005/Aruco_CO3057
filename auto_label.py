import pandas as pd
import os
from pathlib import Path
import cv2

def generate_perfect_labels():
    # --- 1. CONFIG (USE ABSOLUTE PATHS TO BE SAFE) ---
    csv_path = 'train.csv'
    # Use r'' for Windows paths to avoid backslash issues
    img_dir = Path(r'aruco_data/images/train')
    label_dir = Path(r'aruco_data/labels/train')
    
    print(f"--- STEP 1: CHECKING DIRECTORIES ---")
    if not os.path.exists(csv_path):
        print(f"ERROR: Cannot find {csv_path} in current folder: {os.getcwd()}")
        return

    if not img_dir.exists():
        print(f"ERROR: Image directory NOT FOUND: {img_dir.absolute()}")
        return
    
    label_dir.mkdir(parents=True, exist_ok=True)
    print(f"CSV Found. Image Dir Found: {img_dir.absolute()}")

    # --- 2. INDEXING IMAGES ---
    # We look for all images and store them in a dictionary
    print(f"\n--- STEP 2: INDEXING IMAGES ON DISK ---")
    image_files = {Path(f).stem: f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    print(f"Found {len(image_files)} image files on disk.")
    if len(image_files) == 0:
        print("ERROR: No .jpg or .png images found in the train folder!")
        return

    # --- 3. PROCESSING CSV ---
    print(f"\n--- STEP 3: MATCHING CSV ENTRIES ---")
    df = pd.read_csv(csv_path)
    print(f"CSV has {len(df)} rows.")

    success_count = 0
    
    for _, row in df.iterrows():
        raw_id = str(row['image_id'])
        pred_str = row['prediction_string']
        
        # Handle zero-padding (matching '89' to '00089')
        # We find which filename on disk 'contains' or matches our ID
        target_stem = None
        if raw_id in image_files:
            target_stem = raw_id
        else:
            # Try to match by stripping leading zeros
            for stem in image_files.keys():
                if stem.lstrip('0') == raw_id.lstrip('0'):
                    target_stem = stem
                    break
        
        if not target_stem:
            # If we skip an image, we print why
            # print(f"Skip: ID {raw_id} has no matching image file.")
            continue

        # Now we name the .txt file EXACTLY like the image filename
        label_file_path = label_dir / f"{target_stem}.txt"
        
        # If no markers in CSV, create empty file
        if pd.isna(pred_str) or str(pred_str).strip() == "":
            with open(label_file_path, 'w') as f:
                pass
            success_count += 1
            continue

        # Convert CSV coordinates to YOLO boxes
        # We need image size to normalize coordinates
        img_sample = cv2.imread(str(img_dir / image_files[target_stem]))
        if img_sample is None: continue
        h, w = img_sample.shape[:2]

        triplets = list(map(float, str(pred_str).split()))
        yolo_lines = []
        
        for i in range(0, len(triplets), 3):
            # Ground Truth is [ID, Top-Left X, Top-Left Y]
            tl_x, tl_y = triplets[i+1], triplets[i+2]
            
            # Create a standard 70x70 bounding box
            box_w, box_h = 70, 70
            # YOLO wants normalized center coordinates:
            cx = (tl_x + 35) / w
            cy = (tl_y + 35) / h
            nw = box_w / w
            nh = box_h / h
            
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_file_path, 'w') as f:
            f.write("\n".join(yolo_lines))
        
        success_count += 1
        if success_count % 500 == 0:
            print(f"Processed {success_count}/{len(df)} labels...")

    print(f"\n--- FINISHED ---")
    print(f"Successfully created {success_count} matching .txt files in {label_dir}")

if __name__ == "__main__":
    generate_perfect_labels()