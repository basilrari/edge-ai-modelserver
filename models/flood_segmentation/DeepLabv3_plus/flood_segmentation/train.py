import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models.segmentation as models
from torch.utils.data import DataLoader, random_split
from dataset import FloodDataset
from sklearn.metrics import jaccard_score, f1_score
import numpy as np
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms for images
transform_img = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

# Transforms for masks
transform_mask = T.Compose([
    T.Resize((256, 256), interpolation=Image.NEAREST),
    T.PILToTensor(),          # [C, H, W], uint8
    lambda x: x.squeeze(0).long(),
])

# Load full dataset with augmentation enabled for training samples
full_dataset = FloodDataset(
    image_dir='/home/ayushkumar/TEEP/src/Flood_Detect/JPEGImages',
    mask_dir='/home/ayushkumar/TEEP/src/Flood_Detect/SegmentationClass',
    transform_img=transform_img,
    transform_mask=transform_mask,
    augment=True
)

# Fix split indices for reproducibility
dataset_len = len(full_dataset)
indices = np.arange(dataset_len)
np.random.seed(42)
np.random.shuffle(indices)

train_size = int(0.8 * dataset_len)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Save indices for later use in test script
os.makedirs('splits', exist_ok=True)
np.save('splits/train_indices.npy', train_indices)
np.save('splits/val_indices.npy', val_indices)

# Create subsets using indices
from torch.utils.data import Subset
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# Disable augmentation for val dataset
val_dataset.dataset.augment = False

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Model, loss, optimizer
model = models.deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_iou = 0

for epoch in range(10):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)['out']
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(masks.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    val_iou = jaccard_score(all_labels, all_preds, average='binary')
    val_dice = f1_score(all_labels, all_preds, average='binary')

    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model")
