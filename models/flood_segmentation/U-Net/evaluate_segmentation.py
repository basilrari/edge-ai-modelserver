import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from model import UNet  # Import your model


# -------- Dataset Class --------
class FloodSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.resize = transforms.Resize((256, 256))  # Resize for both image and mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask

        image = self.resize(image)
        mask = self.resize(mask)

        if self.transform:
            image = self.transform(image)

        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).float()  # Binarize mask

        return image, mask


# -------- Metric Functions --------
def dice_coefficient(pred, target):
    smooth = 1e-5
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target):
    smooth = 1e-5
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# -------- Load Model --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(3, 1)
model.load_state_dict(torch.load("unet_model.pth", map_location=device))
model.to(device)
model.eval()

# -------- Data Loader --------
transform = transforms.Compose([
    transforms.ToTensor(),  # Only ToTensor on image since resize is handled in Dataset
])

dataset = FloodSegmentationDataset(
    image_dir="dataset/images",   # test images folder
    mask_dir="dataset/masks",     # test masks folder
    transform=transform,
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# -------- Evaluation Loop --------
all_preds = []
all_targets = []

iou_total = 0
dice_total = 0

with torch.no_grad():
    for image, mask in dataloader:
        image = image.to(device)
        mask = mask.to(device)

        output = model(image)
        output = torch.sigmoid(output)             # sigmoid for binary output
        output = (output > 0.5).float()            # thresholding at 0.5

        pred = output.squeeze().cpu().numpy().astype(np.uint8)
        true = mask.squeeze().cpu().numpy().astype(np.uint8)

        all_preds.append(pred.flatten())
        all_targets.append(true.flatten())

        iou_total += iou_score(pred, true)
        dice_total += dice_coefficient(pred, true)

# -------- Compute Metrics --------
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

conf_matrix = confusion_matrix(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, zero_division=0)
recall = recall_score(all_targets, all_preds, zero_division=0)
f1 = f1_score(all_targets, all_preds, zero_division=0)
accuracy = accuracy_score(all_targets, all_preds)

# -------- Print Results --------
num_samples = len(dataloader)
print("----- Segmentation Evaluation Metrics -----")
print(f"IoU Score:          {iou_total / num_samples:.4f}")
print(f"Dice Coefficient:   {dice_total / num_samples:.4f}")
print(f"Pixel Accuracy:     {accuracy:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall:             {recall:.4f}")
print(f"F1 Score:           {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')   # Save to PNG file
plt.close()  # Close the figure to free memory
