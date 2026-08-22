"""Tests a fine-tuned ResNet-18 model against unseen real manuscripts.

This script fetches the fine-tuned ResNet-18 weights from the Hugging Face Hub, 
runs batch inference via a DataLoader, evaluates accuracy, and outputs a 
Seaborn heat map to visualize character confusions.

Author: Huy Le (hl9082)
"""

import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pretrain_resnet import ResNet18OCR, RAMDataset

cv2.setNumThreads(0)
# --- Global Configuration ---
# Updated to your new ResNet-18 repository
HF_REPO_ID = "huyisme-005/ancient-greek-ocr-resnet18" 
# IMAGE_PATH = "test_clean_no_PsiXiBetaZeta"    
FINETUNE_CKPT = "resnet_greek_ocr_finetuned.pth"
IMAGE_SIZE = 128




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


def test_pipeline(model, classes, device):
    

    # Only basic resizing and normalization for testing (no data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 2. DOWNLOAD TEST DATASET FROM HF
    print("☁️ Downloading packed testing dataset from Hugging Face...")
    pt_file_path = hf_hub_download(
        repo_id="huyisme-005/greek-ocr-real-pt", 
        filename="test_dataset_ram.pt", 
        repo_type="dataset"
    )

    test_dataset = RAMDataset(pt_file_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    print("🚀 Starting Batch Processing via DataLoader...")
    print("-" * 50)

    total_images = 0
    correct_predictions = 0
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
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
    test_pipeline(model, classes, device)

if __name__ == "__main__":
    main()