"""Trains a LeNet-5 CNN on binarized Ancient Greek character images.

This script uses PyTorch to train an ultra-lightweight Convolutional Neural
Network (CNN) on a dataset of 32x32 pixel images. It utilizes Automatic
Mixed Precision (AMP) for speed and pushes model checkpoints and training 
metrics to the Hugging Face Hub after every validation phase.

Attributes:
    DATA_DIR (str): Path to the directory containing processed images.
    MODEL_DIR (str): Path to the local directory where models are temporarily saved.
    MODEL_PATH (str): Full local path to the saved PyTorch model (.pth) file.
    HF_REPO_ID (str): The destination Hugging Face repository.
    BATCH_SIZE (int): Number of images processed in a single forward pass.
    EPOCHS (int): Total number of full passes over the training dataset.
    LEARNING_RATE (float): The step size for the Adam optimizer.
    IMAGE_SIZE (int): The target width and height to resize images to.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from huggingface_hub import HfApi

# --- Global Configuration ---
DATA_DIR = "processed_binary_data"
MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "greek_ocr_lenet_fast.pth")
HF_REPO_ID = "YOUR_HUGGINGFACE_USERNAME/ancient-greek-ocr" # <--- UPDATE THIS!

BATCH_SIZE = 512
EPOCHS = 5
LEARNING_RATE = 0.002
IMAGE_SIZE = 32

os.makedirs(MODEL_DIR, exist_ok=True)

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
    """Executes the data loading, model training, evaluation, and Hugging Face upload loop.

    This function coordinates the PyTorch dataloaders, initializes the LeNet-5 model,
    manages Automatic Mixed Precision (AMP), and uploads the resulting weights and
    configurations to Hugging Face at the end of every epoch.
    """
    # --- 1. Hugging Face Setup ---
    api = HfApi()
    print(f"Connecting to Hugging Face Repo: {HF_REPO_ID}...")
    api.create_repo(repo_id=HF_REPO_ID, private=False, exist_ok=True)

    # --- 2. Data Preparation ---
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print(f"Loading data from {DATA_DIR}...")
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    num_classes = len(dataset.classes)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- 3. Hardware & Model Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Hardware selected: {device.type.upper()}")

    model = LeNet5(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # --- 4. Training Loop with Live Checkpointing ---
    print("\n⚡ Starting Speedrun Training with HF Checkpointing...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # --- Validation Step ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                else:
                    outputs = model(images)
                    
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"🏁 Epoch {epoch+1} Summary | Loss: {avg_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

        # --- Hugging Face Checkpointing ---
        print("☁️ Pushing checkpoint to Hugging Face...")
        
        # Save model locally first
        save_dict = {
            'model_state_dict': model.state_dict(),
            'classes': dataset.classes
        }
        torch.save(save_dict, MODEL_PATH)

        # Create a config/metrics file for this epoch
        config_data = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": val_accuracy,
            "model_type": "LeNet-5 Speedrun",
            "image_size": IMAGE_SIZE,
            "classes": dataset.classes
        }
        
        with open("training_config.json", "w") as f:
            json.dump(config_data, f, indent=4)

        # Upload both files to the hub
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo="greek_ocr_lenet_fast.pth",
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        api.upload_file(
            path_or_fileobj="training_config.json",
            path_in_repo="training_config.json",
            repo_id=HF_REPO_ID,
            repo_type="model"
        )

    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Training Complete! Total Time: {total_time:.2f} minutes.")
    print(f"Your model is securely hosted at: https://huggingface.co/{HF_REPO_ID}")

if __name__ == "__main__":
    main()