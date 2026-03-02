import cv2
import os
import shutil
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
    
    # Create the folder safely
    os.makedirs(class_out_dir, exist_ok=True)
    save_path = os.path.join(class_out_dir, filename)
    
    # 1. Load image
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # 2. Fast Binarization (Otsu's Method)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Resize to strictly 64x64
    resized = cv2.resize(binary, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # 4. Save
    cv2.imwrite(save_path, resized)

def main():
    # --- SAFETY CHECK: Wipe corrupted output directory ---
    if os.path.exists(OUTPUT_DIR):
        print(f"🧹 Deleting corrupted '{OUTPUT_DIR}' folder to start fresh...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    files_to_process = []
    found_classes = set()
    
    print(f"🔍 Scanning '{INPUT_DIR}' for valid images...")
    for root_folder, subdirs, files in os.walk(INPUT_DIR):
        
        # --- SAFETY CHECK: Ignore the output folder and hidden ZIP artifacts ---
        if OUTPUT_DIR in root_folder or '__MACOSX' in root_folder or '.ipynb_checkpoints' in root_folder:
            continue
            
        for file in files:
            if file.lower().endswith(valid_extensions):
                # The folder name holding the image is the class name
                class_name = os.path.basename(root_folder)
                
                # If images are just loose in the main folder, ignore them or sort them
                if class_name == os.path.basename(INPUT_DIR):
                    continue 

                full_path = os.path.join(root_folder, file)
                files_to_process.append((full_path, file, class_name))
                found_classes.add(class_name)
    
    total_files = len(files_to_process)
    if total_files == 0:
        print(f"❌ No images found in '{INPUT_DIR}'. Please check your folder structure.")
        return

    print(f"\n✅ Found {total_files} total images across {len(found_classes)} folders.")
    print(f"📂 Detected Folders: {', '.join(sorted(found_classes))}")
    
    # --- SAFETY CHECK: Stop if there are an abnormal amount of folders ---
    if len(found_classes) > 30:
        print("⚠️ Warning: Found too many folders! Did you unzip correctly? Stopping to prevent errors.")
        return
        
    print("\n🚀 Starting high-speed standardization...")
    
    # Multiprocessing for speed
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_single_image, files_to_process), total=total_files, desc="Standardizing"))
        
    print("\n🎉 Processing complete! Your data is perfectly organized.")

if __name__ == "__main__":
    main()