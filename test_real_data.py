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
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image

# --- Global Configuration ---
HF_REPO_ID = "huyisme-005/ancient-greek-ocr" # Your exact repo
IMAGE_PATH = "sample_manuscript_line.jpg"    # <--- Put your test image path here!
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

def process_and_predict(image_path, model, classes, device):
    """Slices an image into characters, predicts each one, and prints the transcription.

    Reads a local image using OpenCV, extracts bounding boxes for each character,
    sorts them left-to-right, formats them for PyTorch, and runs inference.

    Args:
        image_path (str): The local file path to the Ancient Greek text image.
        model (LeNet5): The loaded and trained PyTorch model.
        classes (list of str): The list of class names corresponding to model outputs.
        device (torch.device): The hardware device (CPU or GPU) to run the predictions on.
    """
    print(f"\n🔍 Processing image: {image_path}")
    
    # 1. Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Error: Could not find image at {image_path}")
        return

    # 2. Binarize the image (force it to pure black and white)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 3. Find character contours (bounding boxes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out tiny specks of noise and get bounding boxes
    bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 15]
    
    # Sort boxes from left to right to read naturally
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])

    # 4. Define the exact same image transforms used in training
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transcription = []
    
    print("✍️ Transcribing left to right...")
    # 5. Extract, Predict, and Reconstruct
    with torch.no_grad(): # No need to calculate gradients for prediction
        for x, y, w, h in bounding_boxes:
            # Crop the character with a tiny bit of padding
            padding = 2
            char_crop = img[max(0, y-padding):y+h+padding, max(0, x-padding):x+w+padding]
            
            # Convert OpenCV image (numpy array) to PIL Image for PyTorch transforms
            char_pil = Image.fromarray(char_crop)
            
            # Transform and add batch dimension (Shape becomes: [1, 1, 32, 32])
            char_tensor = transform(char_pil).unsqueeze(0).to(device)
            
            # Predict!
            outputs = model(char_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            
            predicted_char = classes[predicted_idx.item()]
            transcription.append(predicted_char)

            # Optional: Draw a rectangle on the original image to show what the AI saw
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Print the final result
    print("\n====================================")
    print("📜 FINAL TRANSCRIPTION:")
    print(" ".join(transcription))
    print("====================================\n")

    # Show the bounding boxes to the user to prove it worked
    cv2.imshow("AI Vision (Press any key to close)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """Coordinates the Hugging Face download, model loading, and OCR inference pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = download_and_load_model(device)
    
    process_and_predict(IMAGE_PATH, model, classes, device)

if __name__ == "__main__":
    main()