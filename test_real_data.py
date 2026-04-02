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
from torch.utils.data import Dataset, DataLoader
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

# --- 2. PyTorch Dataset ---
class RealManuscriptTestDataset(Dataset):
    """Custom PyTorch Dataset for loading and preprocessing real manuscript images.

    Iterates through a specified root directory, cleans folder names to map exactly 
    to the AI's learned class list, and applies domain-specific OpenCV binarization 
    before converting images into PyTorch tensors.

    Args:
        root_dir (str): The root directory containing subfolders of character images.
        model_classes (list of str): The exact list of class names from the trained model checkpoint.
        transform (torchvision.transforms.Compose, optional): Transforms to apply to the PIL image.

    Attributes:
        image_paths (list of str): Absolute paths to all valid images found in the directory.
        labels (list of int): The corresponding integer class indices for each image.
        class_to_idx (dict): Mapping from string class names to integer indices.
    """

    def __init__(self, root_dir, model_classes, transform=None):
        """Initializes the dataset and builds the class-to-index mapping."""
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(model_classes)}
        
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            true_class = folder_name.replace("lower_", "").capitalize()
            if true_class == "Sigma":
                true_class = "LunateSigma"
                
            if true_class in self.class_to_idx:
                label_idx = self.class_to_idx[true_class]
                
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(folder_path, img_name))
                        self.labels.append(label_idx)

    def __len__(self):
        """Returns the total number of images in the dataset.

        Returns:
            int: Total image count.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Fetches and preprocesses a single image and its label.

        Applies grayscale conversion, Gaussian blur, and adaptive thresholding 
        using OpenCV before returning the transformed tensor.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - tensor_img (torch.Tensor): The preprocessed, formatted image tensor.
                - label (int): The integer index of the true class.
                - img_path (str): The file path of the image for debugging purposes.
        """
        img_path = self.image_paths[idx]
        
        original_img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            15, 4  
        )
        
        pil_img = Image.fromarray(thresh)
        if self.transform:
            tensor_img = self.transform(pil_img)
            
        return tensor_img, self.labels[idx], img_path

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
    model_file = hf_hub_download(repo_id=HF_REPO_ID, filename="greek_ocr_finetuned_checkpoint.pth", repo_type="model")
    
    # Load the checkpoint dictionary
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    classes = checkpoint['classes']
    
    # Initialize model with the correct number of classes and load weights
    model = LeNet5(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode (turns off training features like dropout)
    
    print("✅ Model loaded successfully!")
    return model, classes

def test_pipeline(data_dir, model, classes, device):
    """Executes the batch testing pipeline using a PyTorch DataLoader.

    Loads the custom dataset, processes it in batches to maximize hardware efficiency, 
    compares network predictions against the true labels, and calculates the overall accuracy.

    Args:
        data_dir (str): The root directory containing image subfolders.
        model (LeNet5): The loaded PyTorch model set to evaluation mode.
        classes (list of str): The list of model class names.
        device (torch.device): CPU or GPU.
        
    Returns:
        None
    """
    if not os.path.exists(data_dir):
        print(f"❌ Error: The directory '{data_dir}' does not exist.")
        return

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = RealManuscriptTestDataset(data_dir, classes, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("🚀 Starting Batch Processing via DataLoader...")
    print("-" * 50)

    total_images = 0
    correct_predictions = 0

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted_indices = torch.max(outputs, 1)

            total_images += labels.size(0)
            correct_predictions += (predicted_indices == labels).sum().item()

            mismatches = predicted_indices != labels
            if mismatches.any():
                for i in range(len(mismatches)):
                    if mismatches[i]:
                        true_cls = classes[labels[i].item()]
                        pred_cls = classes[predicted_indices[i].item()]
                        filename = os.path.basename(paths[i])
                        print(f"❌ Mismatch in {filename}: True='{true_cls}', AI Guessed='{pred_cls}'")

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
    """Coordinates the Hugging Face download and the batch testing pipeline.

    Automatically detects hardware accelerators (CUDA) and passes the device to 
    the model loading and inference functions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = download_and_load_model(device)
    test_pipeline(IMAGE_PATH, model, classes, device)


if __name__ == "__main__":
    main()