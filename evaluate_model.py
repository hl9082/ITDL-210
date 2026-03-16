"""Evaluates a trained LeNet-5 OCR model and generates professional metrics.

This script downloads the final model from Hugging Face, evaluates it on a 
validation subset of the local dataset, and outputs a Classification Report 
(Precision, Recall, F1) as well as a visual Confusion Matrix heatmap.

Attributes:
    DATA_DIR (str): Path to the directory containing processed images.
    HF_REPO_ID (str): The Hugging Face repository containing the trained model.
    IMAGE_SIZE (int): The target width and height to resize images to.
    BATCH_SIZE (int): Number of images processed in a single forward pass.

Name: Huy Le (hl9082)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# --- Global Configuration ---
DATA_DIR = "processed_binary_data" 
HF_REPO_ID = "huyisme-005/ancient-greek-ocr_2" # Make sure this matches your repo!
IMAGE_SIZE = 32
BATCH_SIZE = 512

# --- 1. The Network Architecture (Must match exactly) ---
class LeNet5(nn.Module):
    """A lightweight Convolutional Neural Network based on the LeNet-5 architecture.

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

def main():
    """Executes the evaluation pipeline and generates metrics and graphs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running evaluation on hardware: {device.type.upper()}")

    # --- 2. Load Model from Hugging Face ---
    print(f"\n☁️ Downloading final model from Hugging Face ({HF_REPO_ID})...")
    
    
    model_file = hf_hub_download(repo_id=HF_REPO_ID, filename="greek_ocr_lenet_fast.pth", repo_type="model")
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    classes = checkpoint['classes']
    
    model = LeNet5(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # CRITICAL: Turn off training features!
    print("✅ Model loaded successfully!")

    # --- 3. Prepare the Data ---
    print(f"\n📂 Loading evaluation data from {DATA_DIR}...")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    
    # Recreate a 10% validation split for testing
    # We use a fixed generator here so the evaluation is consistent if you run it twice
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"📊 Evaluating against {val_size} validation images.")

    # --- 4. Run Inference ---
    print("\n⚡ Starting massive inference run...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store everything in memory for Scikit-Learn
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 5. Generate Metrics & Reports ---
    print("\n======================================================")
    print("🏆 FINAL CLASSIFICATION REPORT (Precision, Recall, F1)")
    print("======================================================")
    # Scikit-learn handles all the complex math for us!
    report = classification_report(all_labels, all_preds, target_names=classes)
    print(report)

    # --- 6. Plot the Confusion Matrix ---
    print("\n🎨 Generating Confusion Matrix Heatmap...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(16, 12)) # Make it nice and big for the 24 letters
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('LeNet-5 Ancient Greek OCR - Confusion Matrix', fontsize=16)
    plt.ylabel('True Greek Letter', fontsize=14)
    plt.xlabel('AI Predicted Letter', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Save it to your hard drive so you can drop it into your report!
    save_path = "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved high-resolution matrix to: {save_path}")

if __name__ == "__main__":
    main()