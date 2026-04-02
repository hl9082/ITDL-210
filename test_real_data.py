"""Downloads a trained LeNet-5 model from Hugging Face and uses it for OCR.

This script fetches the saved PyTorch weights and class mappings from the 
Hugging Face Hub. It then loads a local image of Ancient Greek text, uses 
OpenCV to detect individual characters, and passes them through the AI to 
generate a final text transcription.

Attributes:
    HF_REPO_ID (str): The Hugging Face repository containing the model.
    IMAGE_PATH (str): The local path to the image you want to transcribe.
    IMAGE_SIZE (int): The target width and height to resize character crops to.

Author: Huy Le (hl9082)
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import os

# --- Global Configuration ---
HF_REPO_ID = "huyisme-005/ancient-greek-ocr_4" # Your exact repo
IMAGE_PATH = "test_clean_no_PsiXiBetaZeta/"    # <--- Put your test image path here!
IMAGE_SIZE = 32

# --- 1. The Network Architecture (Must match exactly) ---
class LeNet5(nn.Module):
    """The identical LeNet-5 architecture used during training.

    Designed for fast processing of small, binarized character images. It uses 
    two convolutional layers followed by three fully connected layers.

    Args:
        num_classes (int): The total number of distinct character classes to predict.
        
    Attributes:
        features (nn.Sequential): The convolutional and pooling layers for feature extraction.
        classifier (nn.Sequential): The fully connected layers for final classification.
    """

    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        """Executes a forward pass through the network.

        Args:
            x (torch.Tensor): A batch of input images with shape (B, 1, H, W).

        Returns:
            torch.Tensor: The raw, unnormalized predictions (logits) for each class.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

def download_and_load_model(device):
    """Fetches the trained model and class mappings from Hugging Face.

    Args:
        device (torch.device): The hardware device (CPU or GPU) to load the model onto.

    Returns:
        tuple: A tuple containing:
            - model (LeNet5): The initialized model loaded with trained weights, set to eval mode.
            - classes (list of str): The list of character class names mapping to model outputs.
    """
    print("☁️ Downloading brain from Hugging Face...")
    model_file = hf_hub_download(repo_id=HF_REPO_ID, filename="greek_ocr_lenet_fast.pth", repo_type="model")
    
    # Load the checkpoint dictionary
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    classes = checkpoint['classes']
    
    # Initialize model with the correct number of classes and load weights
    model = LeNet5(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode (turns off training features like dropout)
    
    print("✅ Model loaded successfully!")
    return model, classes

def preprocess_image(image_path):
    """Loads and applies OpenCV preprocessing to a raw manuscript image.

    Converts to grayscale, applies Gaussian blur, and uses adaptive thresholding 
    to binarize the image.

    Args:
        image_path (str): The file path to the raw image.

    Returns:
        numpy.ndarray or None: The binarized image array, or None if loading fails.
    """
    original_img = cv2.imread(image_path)
    if original_img is None:
        return None

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        15, 4  
    )
    return thresh


def test_directory(data_dir, model, classes, device):
    """Iterates through all folders and images to test model accuracy.

    Assumes the directory is structured such that subfolder names represent 
    the true class of the images within them (e.g., `data_dir/Alpha/img.png`).

    Args:
        data_dir (str): The root directory containing image subfolders.
        model (LeNet5): The loaded PyTorch model.
        classes (list of str): The list of model class names.
        device (torch.device): CPU or GPU.
    """
    if not os.path.exists(data_dir):
        print(f"❌ Error: The directory '{data_dir}' does not exist.")
        return

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    total_images = 0
    correct_predictions = 0

    print("🚀 Starting Batch Processing...")
    print("-" * 50)

    # os.walk automatically loops through every folder and subfolder
    for root, _, files in os.walk(data_dir):
        # Clean folder names to match AI classes
        raw_true_class = os.path.basename(root)
        if not raw_true_class: 
            continue

        true_class = raw_true_class.replace("lower_", "").capitalize()

        # Handle the Sigma edge case
        if true_class == "Sigma":
            true_class = "LunateSigma"

        for file_name in files:
            # Skip non-image files like .DS_Store or text files
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(root, file_name)
            
            # 1. Preprocess
            thresh_img = preprocess_image(image_path)
            if thresh_img is None:
                print(f"⚠️ Warning: Could not read {image_path}. Skipping.")
                continue

            # 2. Format for PyTorch
            char_pil = Image.fromarray(thresh_img)
            char_tensor = transform(char_pil).unsqueeze(0).to(device)

            # 3. Predict
            with torch.no_grad():
                outputs = model(char_tensor)
                _, predicted_idx = torch.max(outputs, 1)
                predicted_class = classes[predicted_idx.item()]

            # 4. Evaluate
            total_images += 1
            is_correct = (predicted_class == true_class)
            if is_correct:
                correct_predictions += 1
            
            # Optional: Print out misclassifications for debugging
            if not is_correct:
                print(f"❌ Mismatch in {file_name}: True='{true_class}', AI Guessed='{predicted_class}'")

    # Final Report
    print("-" * 50)
    if total_images == 0:
        print("⚠️ No valid images found in the specified directory.")
    else:
        accuracy = (correct_predictions / total_images) * 100
        print(f"🏆 BATCH TESTING COMPLETE")
        print(f"Total Images Processed: {total_images}")
        print(f"Correct Predictions:    {correct_predictions}")
        print(f"Real-World Accuracy:    {accuracy:.2f}%")


def main():
    """Coordinates the Hugging Face download and the batch testing pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = download_and_load_model(device)
    
    test_directory(IMAGE_PATH, model, classes, device)

if __name__ == "__main__":
    main()