"""Fine-tunes the synthetic LeNet-5 OCR model on real manuscript data.

This script loads the pre-trained weights from Hugging Face, applies heavy 
data augmentation to a limited real-world dataset, and fine-tunes the network.
It features automatic checkpoint resuming, time-based checkpoint uploading 
to the Hugging Face Hub, and an Early Stopping mechanism.

Attributes:
    HF_REPO_ID (str): The Hugging Face repository to pull/push from.
    TRAIN_DATA_DIR (str): The local directory containing real manuscript training crops.
    EPOCHS (int): The maximum number of training epochs.
    BATCH_SIZE (int): The number of samples per batch.
    LEARNING_RATE (float): The fine-tuning learning rate.
    PATIENCE (int): The number of epochs to wait without improvement before early stopping.
    SAVE_INTERVAL_SEC (int): The number of seconds between Hugging Face checkpoint pushes.

Author: Huy Le (hl9082)
"""

import os
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from huggingface_hub import hf_hub_download, HfApi
from PIL import Image
from train_ocr import LeNet5

# --- Configuration ---
HF_REPO_ID = "huyisme-005/ancient-greek-ocr_4"
TRAIN_DATA_DIR = "train_clean_no_PsiXiBetaZeta"
BASE_WEIGHTS = "greek_ocr_lenet_fast.pth"
FINETUNE_CKPT = "greek_ocr_finetuned_checkpoint.pth"

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
IMAGE_SIZE = 32
PATIENCE = 8 #increase to 8 to give the model more time to learn
SAVE_INTERVAL_SEC = 120  # Save to HF every 2 minutes





class RealManuscriptDataset(Dataset):
    """Custom PyTorch Dataset for loading real manuscript images.

    Args:
        root_dir (str): The root directory containing image subfolders.
        model_classes (list of str): Exact list of class names from the base model.
        transform (torchvision.transforms.Compose, optional): Transforms to apply.

    Attributes:
        image_paths (list of str): Paths to all valid images.
        labels (list of int): Corresponding integer class indices.
        class_to_idx (dict): String-to-integer mapping of classes.
    """
    def __init__(self, root_dir, model_classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(model_classes)}
        
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path): continue
                
            true_class = folder_name.replace("lower_", "").capitalize()
            if true_class == "Sigma": true_class = "LunateSigma"
                
            if true_class in self.class_to_idx:
                label_idx = self.class_to_idx[true_class]
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(folder_path, img_name))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        original_img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
        
        pil_img = Image.fromarray(thresh)
        if self.transform:
            tensor_img = self.transform(pil_img)
            
        return tensor_img, self.labels[idx]


class EarlyStopping:
    """Tracks validation loss and triggers early stopping to prevent overfitting.

    Args:
        patience (int): Number of epochs to wait without improvement.
        min_delta (float): Minimum change to qualify as an improvement.

    Attributes:
        counter (int): Current epochs without improvement.
        best_loss (float): Best validation loss recorded so far.
        early_stop (bool): Flag indicating if training should halt.
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"⚠️ EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def load_or_resume_model(device):
    """Attempts to pull the latest fine-tuning checkpoint, otherwise pulls base model.

    Args:
        device (torch.device): CPU or GPU.

    Returns:
        tuple: (model, optimizer, start_epoch, classes)
    """
    print("☁️ Checking Hugging Face for existing fine-tuning checkpoint...")
    try:
        # Try fetching the checkpoint where we left off
        model_file = hf_hub_download(repo_id=HF_REPO_ID, filename=FINETUNE_CKPT)
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        classes = checkpoint['classes']
        start_epoch = checkpoint['epoch']
        
        model = LeNet5(num_classes=len(classes)).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Add weight_decay=1e-4 to penalize overfitting
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"🔄 Resuming fine-tuning from Epoch {start_epoch}!")
        
    except Exception:
        # If no fine-tuning checkpoint exists, pull the base synthetic model
        print("📥 No existing fine-tuning checkpoint found. Pulling base synthetic model...")
        model_file = hf_hub_download(repo_id=HF_REPO_ID, filename=BASE_WEIGHTS)
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        classes = checkpoint['classes']
        start_epoch = 0
        
        model = LeNet5(num_classes=len(classes)).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
    return model, optimizer, start_epoch, classes


def save_checkpoint_to_hf(model, optimizer, epoch, classes):
    """Saves the checkpoint locally and uploads it to the Hugging Face Hub.

    Args:
        model (LeNet5): The neural network.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current training epoch.
        classes (list of str): List of class names.
    """
    print(f"\n💾 Saving and pushing Checkpoint for Epoch {epoch} to Hugging Face...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classes': classes
    }, FINETUNE_CKPT)

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=FINETUNE_CKPT,
            path_in_repo=FINETUNE_CKPT,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"Auto-save fine-tuning checkpoint at Epoch {epoch}"
        )
        print("✅ Checkpoint uploaded successfully!\n")
    except Exception as e:
        print(f"❌ Failed to upload checkpoint to HF: {e}\n")


def main():
    """Coordinates data loading, model initialization, and the fine-tuning loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model & State
    model, optimizer, start_epoch, classes = load_or_resume_model(device)
    criterion = nn.CrossEntropyLoss()

    # Learning Rate Scheduler - Halves LR if Val Loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 2. Data Preparation (Train & Val Split for Early Stopping)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(degrees=15),       
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
        transforms.ToTensor(),
        # RandomErasing drops random black/white boxes on the tensor to simulate damaged manuscript
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = RealManuscriptDataset(TRAIN_DATA_DIR, classes, transform=train_transform)
    
    # 80% Training, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Setup Training Utilities
    early_stopping = EarlyStopping(patience=PATIENCE)
    last_save_time = time.time()

    # 4. The Fine-Tuning Loop
    print(f"🚀 Starting Fine-Tuning on Real Data (Epochs {start_epoch+1} to {EPOCHS})...")
    
    for epoch in range(start_epoch, EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # --- TIME-BASED CHECKPOINT SAVING ---
            current_time = time.time()
            if current_time - last_save_time >= SAVE_INTERVAL_SEC:
                save_checkpoint_to_hf(model, optimizer, epoch + 1, classes)
                last_save_time = current_time

        # --- VALIDATION PHASE (For Early Stopping) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()
                
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Check Early Stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered! Overfitting detected. Halting training.")
            break

    # Final Save when done
    save_checkpoint_to_hf(model, optimizer, EPOCHS, classes)
    print("🎉 Fine-Tuning Complete!")

if __name__ == "__main__":
    main()