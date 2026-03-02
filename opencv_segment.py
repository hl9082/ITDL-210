import cv2
import os
import numpy as np

# --- Configuration ---
INPUT_DIR = "raw_data"
OUTPUT_DIR = "processed_binary_data"
TARGET_SIZE = (64, 64) # The size your OCR model will expect

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_image(image_path, filename):
    print(f"Processing: {filename}...")
    
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Denoise (Removes parchment texture/grain)
    # h=10 is the filter strength. Higher removes more noise but might blur ink.
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4. Binarization (Remove background)
    # Using Adaptive Thresholding because manuscript lighting/fading is usually uneven
    binary = cv2.adaptiveThreshold(
        denoised, 
        255, # Maximum value (White)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, # Invert so ink is white, paper is black (better for finding contours)
        11, # Block size (must be odd)
        2   # Constant subtracted from mean
    )

    # 5. Find Contours (The shapes of the letters)
    # RETR_EXTERNAL ensures we only get the outside edges of the letters, not the holes inside an Alpha or Omicron
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for contour in contours:
        # 6. Get the Bounding Box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # 7. Filter out noise (Ignore boxes that are too tiny or massively huge)
        # You may need to adjust these numbers based on the resolution of your raw images!
        if 15 < w < 200 and 15 < h < 200:
            
            # 8. Crop the letter using array slicing [startY:endY, startX:endX]
            # We crop from the 'binary' image so the saved result has no background
            letter_crop = binary[y:y+h, x:x+w]
            
            # 9. Resize to a standard size (64x64) for the neural network
            # INTER_AREA is the best interpolation method for shrinking images
            resized_letter = cv2.resize(letter_crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            # 10. Save the extracted letter
            # Name format: originalFileName_contourNumber.png
            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(OUTPUT_DIR, f"{base_name}_char_{count:04d}.png")
            cv2.imwrite(save_path, resized_letter)
            
            count += 1

    print(f"  -> Extracted {count} characters from {filename}")

# --- Main Execution Loop ---
def main():
    # Get all image files from the input directory
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    
    if not files:
        print(f"No images found in '{INPUT_DIR}'. Please add some test images.")
        return

    for file in files:
        file_path = os.path.join(INPUT_DIR, file)
        process_image(file_path, file)
        
    print("\nProcessing complete! Check the output folder.")

if __name__ == "__main__":
    main()