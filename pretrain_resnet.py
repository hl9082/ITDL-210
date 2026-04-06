"""Pre-trains a ResNet-18 architecture on synthetic Ancient Greek character data.

This script initializes a standard ResNet-18, modifies its input layer to accept
1-channel grayscale images, and trains it from scratch using a clean, synthetic
dataset. The resulting weights act as a strong base representation for transfer
learning on degraded real-world manuscripts.

Author: Huy Le (hl9082)
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import gc

cv2.setNumThreads(0) #no parallel processing here

# --- Configuration ---
HF_REPO_ID = "huyisme-005/ancient-greek-ocr-resnet18"
TRAIN_DATA_DIR = "processed_binary_data"  # Replace with your synthetic folder name
BASE_WEIGHTS_CKPT = "resnet_greek_ocr_base.pth"
METRICS_FILE = "base_metrics.json"

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = 128
PATIENCE = 3  # Stops training if validation loss doesn't improve for 3 epochs


class ResNet18OCR(nn.Module):
    """A customized ResNet-18 model for grayscale character recognition.

    Args:
        num_classes (int): The number of unique character classes to predict.
    """

    def __init__(self, num_classes):
        super(ResNet18OCR, self).__init__()
        self.resnet = models.resnet18(weights=None)
        
        # Modify the first convolutional layer for 1-channel grayscale input
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Modify the fully connected layer to output our specific class count
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): A batch of 1-channel image tensors.

        Returns:
            torch.Tensor: The raw, unnormalized prediction logits.
        """
        return self.resnet(x)


class RAMDataset(Dataset):
    def __init__(self, pt_file_path, transform=None):
        print("🧠 Loading dataset into RAM... please wait a few seconds.")
        data = torch.load(pt_file_path, weights_only=False)
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform
        print(f"✅ Loaded {len(self.images)} images into memory!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # Convert uint8 to float32 [0.0, 1.0] 
        image = image.to(torch.float32) / 255.0
        if self.transform:
            image = self.transform(image)
        return image, label

class EarlyStopping:
    """Monitors validation loss and halts training to prevent overfitting.

    Args:
        patience (int): Number of epochs to wait for improvement.
        min_delta (float): Minimum change to qualify as an improvement.
    """

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, best_loss):
        if val_loss > best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

def save_checkpoint_to_hf(model, optimizer, epoch, classes, best_val_loss, metrics):
    """Saves the model state and uploads it to the Hugging Face Hub.

    Args:
        model (torch.nn.Module): The current network model.
        optimizer (torch.optim.Optimizer): The training optimizer.
        epoch (int): The current training epoch.
        classes (list[str]): The ordered list of character classes.
    """
    print(f"\n💾 Saving Base Checkpoint at Epoch {epoch}...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classes': classes,
        'best_val_loss': best_val_loss
    }, BASE_WEIGHTS_CKPT)

    # Save Metrics JSON
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=BASE_WEIGHTS_CKPT,
            path_in_repo=BASE_WEIGHTS_CKPT,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message="Uploading Pre-trained ResNet-18 Base Model"
        )

        # Upload metrics
        api.upload_file(
            path_or_fileobj=METRICS_FILE,
            path_in_repo=METRICS_FILE,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"Auto-push: Base Metrics at Epoch {epoch}"
        )

        print("✅ Base Model uploaded successfully!\n")
    except Exception as e:
        print(f"❌ Failed to upload checkpoint to HF: {e}\n")


def main():
    """Main execution function to initialize and train the base model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # Define classes based on your earlier LeNet-5 structure
    classes = [
        "Alpha", "Gamma", "Delta", "Epsilon", "Eta", "Theta", "Iota", 
        "Kappa", "Lambda", "Mu", "Nu", "Omicron", "Pi", "Rho", 
        "LunateSigma", "Tau", "Upsilon", "Phi", "Chi", "Omega"
    ]
    
    model = ResNet18OCR(num_classes=len(classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # --- Resume Training Logic ---
    start_epoch = 0
    best_val_loss = float('inf')
    try:
        print("☁️ Checking for existing base checkpoint to resume...")
        model_file = hf_hub_download(repo_id=HF_REPO_ID, filename=BASE_WEIGHTS_CKPT)
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✅ Resuming from Epoch {start_epoch} (Previous Best Loss: {best_val_loss:.4f})")
    except Exception:
        print("ℹ️ No existing checkpoint found. Starting pre-training from scratch.")

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Mild augmentations for synthetic data
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 2. DOWNLOAD DATASET FROM HF
    print("☁️ Downloading packed synthetic dataset from Hugging Face...")
    pt_file_path = hf_hub_download(
        repo_id="huyisme-005/greek-ocr-synthetic-pt", # Make sure this matches your repo!
        filename="dataset_ram.pt", 
        repo_type="dataset"
    )


    # 3. INITIALIZE DATALOADERS
    full_dataset = RAMDataset(pt_file_path, transform=train_transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print("🚀 Starting Pre-training on Synthetic Data...")

    # Initialize infinite loss to guarantee the first epoch saves
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # 1. Wrap the train_loader with tqdm
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update the progress bar text with the live loss
            train_bar.set_postfix(loss=loss.item())
            gc.collect()

        model.eval()
        val_loss = 0.0

        all_preds, all_labels = [], []

        # 2. Wrap the val_loader with tqdm
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()
                _, preds = torch.max(val_outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                val_bar.set_postfix(loss=loss.item())
                
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Calculate Advanced Metrics
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"\n Metrics -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"\n Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        

        # Dynamic Checkpoint Trigger
        if avg_val_loss < best_val_loss:
            print(f"   🌟 Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}!")
            best_val_loss = avg_val_loss
            metrics_dict = {
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            save_checkpoint_to_hf(model, optimizer, epoch + 1, classes, best_val_loss, metrics_dict)

        early_stopping(avg_val_loss, best_val_loss)
        
             
        if early_stopping.early_stop:
            print("🛑 Network has memorized synthetic data. Early stopping triggered to prevent font overfitting.")
            break

    
    print("🎉 Pre-Training Complete!")

if __name__ == "__main__":
    main()