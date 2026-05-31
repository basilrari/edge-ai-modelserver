import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "dataset"

# =========================
# Data Augmentation
# =========================
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =========================
# Dataset
# =========================
train_dataset = datasets.ImageFolder(
    os.path.join(dataset_path,'train'),
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    os.path.join(dataset_path,'val'),
    transform=val_transforms
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# =========================
# Model (IMPORTANT: RAW RESNET18 ONLY)
# =========================
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# =========================
# Loss + Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =========================
# Training Loop
# =========================
epochs = 20

for epoch in range(epochs):

    model.train()
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    train_acc = 100 * correct / total

    # =========================
    # Validation
    # =========================
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

# =========================
# SAVE CHECKPOINT (FIXED FORMAT)
# =========================
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": train_dataset.classes
}, "flood_resnet18.pth")

print("Model saved successfully")
