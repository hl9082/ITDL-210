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
from huggingface_hub import HfApi
from PIL import Image

# --- Configuration ---
HF_REPO_ID = "huyisme-005/ancient-greek-ocr-resnet18"
TRAIN_DATA_DIR = "processed_binary_data"  # Replace with your synthetic folder name
BASE_WEIGHTS_CKPT = "resnet_greek_ocr_base.pth"

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


class SyntheticDataset(Dataset):
    """Custom Dataset for loading clean synthetic Greek characters.

    Args:
        root_dir (str): The root directory containing class folders.
        model_classes (list[str]): An ordered list of class names.
        transform (torchvision.transforms.Compose, optional): Augmentations.
    """

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
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Fetches and preprocesses a single image and its label.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: (tensor_img, label_idx)
        """
        img_path = self.image_paths[idx]
        original_img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        
        # Simple binarization for synthetic data (this ensures we don't inadvertently binarize our already
        # binarized images)
        _, thresh = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
        
        pil_img = Image.fromarray(thresh)
        if self.transform:
            tensor_img = self.transform(pil_img)
            
        return tensor_img, self.labels[idx]

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
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_checkpoint_to_hf(model, optimizer, epoch, classes):
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
        'classes': classes
    }, BASE_WEIGHTS_CKPT)

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=BASE_WEIGHTS_CKPT,
            path_in_repo=BASE_WEIGHTS_CKPT,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message="Uploading Pre-trained ResNet-18 Base Model"
        )
        print("✅ Base Model uploaded successfully!\n")
    except Exception as e:
        print(f"❌ Failed to upload checkpoint to HF: {e}\n")


def main():
    """Main execution function to initialize and train the base model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define classes based on your earlier LeNet-5 structure
    classes = [
        "Alpha", "Gamma", "Delta", "Epsilon", "Eta", "Theta", "Iota", 
        "Kappa", "Lambda", "Mu", "Nu", "Omicron", "Pi", "Rho", 
        "LunateSigma", "Tau", "Upsilon", "Phi", "Chi", "Omega"
    ]
    
    model = ResNet18OCR(num_classes=len(classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Mild augmentations for synthetic data
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = SyntheticDataset(TRAIN_DATA_DIR, classes, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("🚀 Starting Pre-training on Synthetic Data...")
    
    for epoch in range(EPOCHS):
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
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    save_checkpoint_to_hf(model, optimizer, EPOCHS, classes)
    print("🎉 Pre-Training Complete!")

if __name__ == "__main__":
    main()