import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# --- Configuration ---
DATA_DIR = "processed_binary_data"
MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "greek_ocr_model.pth")

BATCH_SIZE = 16
EPOCHS = 10         # Start with 10 for your "Micro-Test". Increase later.
LEARNING_RATE = 0.001
IMAGE_SIZE = 64     # Matches Phase 1

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- 1. Data Preparation ---
# Transformations: Convert image to grayscale, resize (just in case), and turn into PyTorch Tensors
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Ensure 1 channel (black & white)
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalize pixel values between -1 and 1
])

# Load the dataset using ImageFolder (it automatically uses subfolder names as class labels!)
print(f"Loading data from {DATA_DIR}...")
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
NUM_CLASSES = len(dataset.classes)
print(f"Found {NUM_CLASSES} classes: {dataset.classes}")

# Split into Training (80%) and Validation (20%) sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. The Neural Network Architecture ---
class GreekOCRNet(nn.Module):
    def __init__(self, num_classes):
        super(GreekOCRNet, self).__init__()
        # Feature Extractor (Convolutional Layers)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Reduces 64x64 -> 32x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Reduces 32x32 -> 16x16
        )
        
        # Classifier (Fully Connected Layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 3. Setup Training ---
# Automatically use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

model = GreekOCRNet(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. Training Loop ---
print("\nStarting Training...")
for epoch in range(EPOCHS):
    model.train() # Set to training mode
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validation step
    model.eval() # Set to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # Don't track gradients during validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Validation Acc: {val_accuracy:.2f}%")

# --- 5. Save the Model ---
print(f"\nTraining complete. Saving model to {MODEL_PATH}...")
# We save the state_dict (the weights) and the class mapping so Step 3 knows what '0', '1', '2' mean
save_dict = {
    'model_state_dict': model.state_dict(),
    'classes': dataset.classes
}
torch.save(save_dict, MODEL_PATH)
print("Model saved successfully!")