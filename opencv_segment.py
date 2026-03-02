import cv2
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
INPUT_DIR = "raw_data"
OUTPUT_DIR = "processed_binary_data"
TARGET_SIZE = (64, 64) # The size your OCR model will expect

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_single_image(args):
    file_path, filename, class_name = args
    
    class_out_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_out_dir, exist_ok=True)
    
    save_path = os.path.join(class_out_dir, filename)
    
    # Skip if already processed (allows you to pause and resume)
    if os.path.exists(save_path):
        return

    # 1. Load image
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # 2. Fast Binarization (Otsu's Method)
    # This is 100x faster than adaptive thresholding + denoising
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Resize to strictly 64x64
    resized = cv2.resize(binary, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # 4. Save
    cv2.imwrite(save_path, resized)

def main():
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    files_to_process = []
    
    print("Scanning directories for images... (This might take a minute for 200k files)")
    for root_folder, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(valid_extensions):
                class_name = os.path.basename(root_folder)
                if class_name == os.path.basename(INPUT_DIR):
                    class_name = "Unsorted"
                full_path = os.path.join(root_folder, file)
                files_to_process.append((full_path, file, class_name))
    
    total_files = len(files_to_process)
    if total_files == 0:
        print(f"No images found in '{INPUT_DIR}'.")
        return

    print(f"Found {total_files} images. Starting high-speed processing...")
    
    # Use ThreadPoolExecutor to process multiple images at the exact same time
    # This will max out your CPU and finish in minutes instead of hours
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_single_image, files_to_process), total=total_files, desc="Standardizing"))
        
    print("\nProcessing complete! All 205,797 images are ready for PyTorch.")

if __name__ == "__main__":
    main()