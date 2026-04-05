"""Tests a fine-tuned ResNet-18 model against unseen real manuscripts.

This script fetches the fine-tuned ResNet-18 weights from the Hugging Face Hub, 
runs batch inference via a DataLoader, evaluates accuracy, and outputs a 
Seaborn heat map to visualize character confusions.

Author: Huy Le (hl9082)
"""

import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- Global Configuration ---
# Updated to your new ResNet-18 repository
HF_REPO_ID = "huyisme-005/ancient-greek-ocr-resnet18" 
IMAGE_PATH = "test_clean_no_PsiXiBetaZeta"    
FINETUNE_CKPT = "resnet_greek_ocr_finetuned.pth"
IMAGE_SIZE = 128


class ResNet18OCR(nn.Module):
    """A customized ResNet-18 model for grayscale character recognition."""
    def __init__(self, num_classes):
        super(ResNet18OCR, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


class RealManuscriptTestDataset(Dataset):
    """Custom PyTorch Dataset for loading and preprocessing real manuscript images."""
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
        
        # Exact same thresholding used in fine-tuning
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
        
        pil_img = Image.fromarray(thresh)
        if self.transform:
            tensor_img = self.transform(pil_img)
            
        return tensor_img, self.labels[idx], img_path


def plot_confusion_matrix(true_labels, pred_labels, classes):
    """Generates and saves a visual confusion matrix."""
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual Letter')
    plt.xlabel('AI Guessed Letter')
    plt.title('ResNet-18 Real-World Confusion Matrix')
    plt.xticks(rotation=45)
    plt.savefig('resnet_real_world_confusion_matrix.png', bbox_inches='tight')
    print("📊 Confusion matrix saved as 'resnet_real_world_confusion_matrix.png'")


def download_and_load_model(device):
    print("☁️ Downloading fine-tuned ResNet-18 brain from Hugging Face...")
    model_file = hf_hub_download(repo_id=HF_REPO_ID, filename=FINETUNE_CKPT, repo_type="model")
    
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    classes = checkpoint['classes']
    
    model = ResNet18OCR(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    
    print("✅ Model loaded successfully!")
    return model, classes


def test_pipeline(data_dir, model, classes, device):
    if not os.path.exists(data_dir):
        print(f"❌ Error: The directory '{data_dir}' does not exist.")
        return

    # Only basic resizing and normalization for testing (no data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = RealManuscriptTestDataset(data_dir, classes, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    print("🚀 Starting Batch Processing via DataLoader...")
    print("-" * 50)

    total_images = 0
    correct_predictions = 0
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted_indices = torch.max(outputs, 1)

            total_images += labels.size(0)
            correct_predictions += (predicted_indices == labels).sum().item()

            all_true_labels.extend([classes[lbl.item()] for lbl in labels])
            all_pred_labels.extend([classes[pred.item()] for pred in predicted_indices])

    print("-" * 50)
    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print(f"🏆 BATCH TESTING COMPLETE")
        print(f"Total Images Processed: {total_images}")
        print(f"Correct Predictions:    {correct_predictions}")
        print(f"Real-World Accuracy:    {accuracy:.2f}%")
        print("\n======================================================")
        print("🏆 FINAL CLASSIFICATION REPORT (Precision, Recall, F1)")
        print("======================================================")
        # Scikit-learn handles all the complex math for us!
        report = classification_report(all_true_labels, all_pred_labels, target_names=classes)
        print(report)
        # Calculate Advanced Metrics
        plot_confusion_matrix(all_true_labels, all_pred_labels, classes)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model, classes = download_and_load_model(device)
    test_pipeline(IMAGE_PATH, model, classes, device)

if __name__ == "__main__":
    main()