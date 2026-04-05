"""Fine-tunes a pre-trained ResNet-18 model on real manuscript data.

This script downloads the synthetic pre-trained ResNet-18 weights and applies
transfer learning using heavy spatial and pixel-level augmentations. It includes
a dynamic learning rate scheduler and EARLY STOPPING to prevent overfitting.

Author: Huy Le (hl9082)
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from huggingface_hub import hf_hub_download, HfApi
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import gc

# --- Configuration ---
# Updated to your new repository!
HF_REPO_ID = "huyisme-005/ancient-greek-ocr-resnet18"
TRAIN_DATA_DIR = "train_clean_no_PsiXiBetaZeta"
BASE_WEIGHTS_CKPT = "resnet_greek_ocr_base.pth"
FINETUNE_CKPT = "resnet_greek_ocr_finetuned.pth"
METRICS_FILE = "finetune_metrics.json"

EPOCHS = 100
BATCH_SIZE = 16  
LEARNING_RATE = 0.0005
IMAGE_SIZE = 128
PATIENCE = 8  # How many epochs Early Stopping will wait


class ResNet18OCR(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18OCR, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


class RealManuscriptDataset(Dataset):
    def __init__(self, root_dir, model_classes, transform=None):
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
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        original_img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        
        # Adaptive thresholding for messy papyrus
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
        )
        
        pil_img = Image.fromarray(thresh)
        if self.transform:
            tensor_img = self.transform(pil_img)
            
        return tensor_img, self.labels[idx]


# ==========================================
# 1. EARLY STOPPING CLASS DEFINITION
# ==========================================
class EarlyStopping:
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
    print(f"\n   💾 Pushing New Best Checkpoint (Epoch {epoch}) and Metrics to HF...")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classes': classes,
        'best_val_loss': best_val_loss
    }, FINETUNE_CKPT)

    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=FINETUNE_CKPT,
            path_in_repo=FINETUNE_CKPT,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"Auto-push: Improved Fine-tuned Model at Epoch {epoch} (F1: {metrics['f1']:.4f})"
        )
        api.upload_file(
            path_or_fileobj=METRICS_FILE,
            path_in_repo=METRICS_FILE,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"Auto-push: Fine-tune Metrics at Epoch {epoch}"
        )
        print("   ✅ Upload successful!")
    except Exception as e:
        print(f"   ❌ Failed to upload to HF: {e}")



def save_checkpoint_to_hf(model, optimizer, epoch, classes):
    print(f"\n💾 Saving Fine-Tuned Checkpoint at Epoch {epoch}...")
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
            commit_message=f"Fine-tuned ResNet-18 at Epoch {epoch}"
        )
    except Exception as e:
        print(f"❌ Failed to upload checkpoint: {e}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Resume or Initialize Logic ---
    start_epoch = 0
    best_val_loss = float('inf')
    model = None
    optimizer = None
    classes = []

    try:
        print("☁️ Checking for existing FINE-TUNED checkpoint to resume...")
        model_file = hf_hub_download(repo_id=HF_REPO_ID, filename=FINETUNE_CKPT)
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        classes = checkpoint['classes']
        
        model = ResNet18OCR(num_classes=len(classes)).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✅ Resuming Fine-Tuning from Epoch {start_epoch} (Previous Best Loss: {best_val_loss:.4f})")
        
    except Exception:
        print("ℹ️ No fine-tuned checkpoint found. Pulling BASE model to start transfer learning...")
        model_file = hf_hub_download(repo_id=HF_REPO_ID, filename=BASE_WEIGHTS_CKPT)
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        classes = checkpoint['classes']
        
        model = ResNet18OCR(num_classes=len(classes)).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=PATIENCE)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(degrees=20),       
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10), 
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)), 
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = RealManuscriptDataset(TRAIN_DATA_DIR, classes, transform=train_transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n🚀 Starting Transfer Learning Loop...")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
            gc.collect()

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for val_images, val_labels in val_bar:
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
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📊 Metrics -> Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        print(f"   Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
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
            print("🛑 Overfitting detected on real manuscript data. Early stopping triggered.")
            break

    print("🎉 Fine-Tuning Loop Concluded!")

if __name__ == "__main__":
    main()