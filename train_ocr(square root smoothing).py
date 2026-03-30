"""Trains a LeNet-5 CNN on binarized Ancient Greek character images.

This script uses PyTorch to train an ultra-lightweight Convolutional Neural
Network (CNN) on a dataset of 32x32 pixel images. It utilizes Automatic
Mixed Precision (AMP) for speed and pushes model checkpoints and training 
metrics to the Hugging Face Hub after every validation phase. This time,
we use the Square Root Smoothing method to fix the "Paranoia" Mathematics.

Attributes:
    DATA_DIR (str): Path to the directory containing processed images.
    MODEL_DIR (str): Path to the local directory where models are temporarily saved.
    MODEL_PATH (str): Full local path to the saved PyTorch model (.pth) file.
    HF_REPO_ID (str): The destination Hugging Face repository.
    BATCH_SIZE (int): Number of images processed in a single forward pass.
    EPOCHS (int): Total number of full passes over the training dataset.
    LEARNING_RATE (float): The step size for the Adam optimizer.
    IMAGE_SIZE (int): The target width and height to resize images to.

Author: Huy Le (hl9082)
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
from huggingface_hub import HfApi, hf_hub_download
import numpy as np

# --- Global Configuration ---
DATA_DIR = "processed_binary_data"
MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "greek_ocr_lenet_fast.pth")
HF_REPO_ID = "huyisme-005/ancient-greek-ocr_4" # <--- UPDATE THIS!

BATCH_SIZE = 512
EPOCHS = 15
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
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"📊 Dataset split: {train_size} training images, {val_size} validation images.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- 3. Hardware & Model Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Hardware selected: {device.type.upper()}")

    model = LeNet5(num_classes=num_classes).to(device)
    # === NEW: Calculating Class Weights to fix the Class Imbalance ===
    print("\n⚖️ Calculating class weights to balance the dataset...")
    
    
    # 1. Extract the labels specifically for the 90% training subset
    train_indices = train_dataset.indices
    train_targets = [dataset.targets[i] for i in train_indices]
    
    # 2. Count how many times each letter appears in the training set
    class_counts = np.bincount(train_targets)
    
    # 3. Apply the Inverse Class Frequency formula
    total_samples = len(train_targets)
    num_classes = len(dataset.classes)
    
    # Avoid division by zero just in case a class has 0 samples in the split
    class_counts = np.where(class_counts == 0, 1, class_counts)

    # NEW: Take the square root of the counts to soften the extreme penalties!
    smoothed_counts = np.sqrt(class_counts) 
    
    total_samples = np.sum(smoothed_counts)
    num_classes = len(dataset.classes)

    class_weights = total_samples / (num_classes * smoothed_counts)
    
    # 4. Convert to a PyTorch tensor and send it to the GPU
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # 5. Inject the weights into the Loss Function
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    print("✅ Class weights successfully applied to the Loss Function!")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # --- 4. Resume from Hugging Face Logic ---
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    print("\n🔍 Checking Hugging Face for existing checkpoints to resume...")
    try:
        # Try downloading the config to see what epoch we are on
        config_file = hf_hub_download(repo_id=HF_REPO_ID, filename="training_config.json", repo_type="model")
        with open(config_file, "r") as f:
            config_data = json.load(f)
            start_epoch = config_data.get("epoch", 0)
            # Load our early stopping state!
            best_val_loss = config_data.get("best_val_loss", float('inf'))
            patience_counter = config_data.get("patience_counter", 0)

        # Download the weights and optimizer state
        model_file = hf_hub_download(repo_id=HF_REPO_ID, filename="greek_ocr_lenet_fast.pth", repo_type="model")
        checkpoint = torch.load(model_file, map_location=device)
        
        # Load the saved brains and momentum
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"✅ Found checkpoint! Resuming training from Epoch {start_epoch + 1}...")
        print(f"   -> Previous Best Loss: {best_val_loss:.4f} | Strikes: {patience_counter}")
    except Exception as e:
        print("ℹ️ No previous checkpoint found on Hugging Face (or repo is empty). Starting fresh from Epoch 1.")

    # --- 5. Training Loop with Live Checkpointing ---
    if start_epoch >= EPOCHS:
        print(f"\n🎉 Model has already completed all {EPOCHS} epochs! Nothing to train.")
        return
    patience = 3 # How many bad epochs we tolerate before killing the script
    print(f"\n⚡ Starting Training Loop (Epoch {start_epoch + 1} to {EPOCHS})...")
    start_time = time.time()

    # NOTE: We use start_epoch in the range now so it skips previously trained epochs!
    for epoch in range(start_epoch, EPOCHS):
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
        val_running_loss = 0.0 # Track validation loss for Early Stopping
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        val_loss = val_running_loss / len(val_loader)
        print(f"🏁 Epoch {epoch+1} Summary | Loss: {avg_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

        # --- Early Stopping & Checkpointing ---
        if val_loss < best_val_loss:
            print(f"🌟 Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Pushing to Hugging Face!")
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save dict to hard drive
            save_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), # Needed for resuming
                'classes': dataset.classes
            }
            torch.save(save_dict, MODEL_PATH)

            # Save state configuration
            config_data = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "val_loss": val_loss,
                "accuracy": val_accuracy,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "model_type": "LeNet-5 Speedrun",
                "image_size": IMAGE_SIZE,
                "classes": dataset.classes
            }
            with open("training_config.json", "w") as f:
                json.dump(config_data, f, indent=4)

            # Upload ONLY the best model to Hugging Face
            api.upload_file(path_or_fileobj=MODEL_PATH, path_in_repo="greek_ocr_lenet_fast.pth", repo_id=HF_REPO_ID, repo_type="model")
            api.upload_file(path_or_fileobj="training_config.json", path_in_repo="training_config.json", repo_id=HF_REPO_ID, repo_type="model")
            
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Strike {patience_counter} of {patience}.")
            
            # Save the new strike count to the config so resuming remembers the strikes!
            config_data = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "val_loss": val_loss,
                "accuracy": val_accuracy,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "model_type": "LeNet-5 Speedrun",
                "image_size": IMAGE_SIZE,
                "classes": dataset.classes
            }
            with open("training_config.json", "w") as f:
                json.dump(config_data, f, indent=4)
            api.upload_file(path_or_fileobj="training_config.json", path_in_repo="training_config.json", repo_id=HF_REPO_ID, repo_type="model")

            if patience_counter >= patience:
                print(f"\n💀 EARLY STOPPING TRIGGERED! The model has stopped learning.")
                print(f"The smartest model is safely preserved on Hugging Face (Val Loss: {best_val_loss:.4f}).")
                break # Instantly breaks the epoch loop

    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Training Complete! Total Time: {total_time:.2f} minutes.")
    print(f"Your best model is securely hosted at: https://huggingface.co/{HF_REPO_ID}")

if __name__ == "__main__":
    main()