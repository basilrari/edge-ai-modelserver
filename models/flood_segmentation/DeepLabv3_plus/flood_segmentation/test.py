import torch
import torchvision.transforms as T
import torchvision.models.segmentation as models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader, random_split
from dataset import FloodDataset
from PIL import Image
import os
import cv2

# Paths
image_dir = '/home/ayushkumar/TEEP/src/Flood_Detect/JPEGImages'
mask_dir = '/home/ayushkumar/TEEP/src/Flood_Detect/SegmentationClass'
model_path = 'best_model.pth'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform_img = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])
transform_mask = T.Compose([
    T.Resize((256, 256), interpolation=Image.NEAREST),
])

# Dataset
full_dataset = FloodDataset(image_dir=image_dir, mask_dir=mask_dir, transform_img=transform_img, transform_mask=transform_mask, augment=False)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
_, test_dataset = random_split(full_dataset, [train_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model
model = models.deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Metrics
all_preds = []
all_labels = []

print("🔍 Evaluating and visualizing...")

for i, (image, mask) in enumerate(test_loader):
    image, mask = image.to(device), mask.to(device)
    with torch.no_grad():
        output = model(image)['out']
        pred = torch.argmax(output, dim=1)

    pred_np = pred.cpu().numpy().squeeze()
    mask_np = mask.cpu().numpy().squeeze()
    img_np = image.cpu().squeeze().permute(1, 2, 0).numpy()

    all_preds.append(pred_np.flatten())
    all_labels.append(mask_np.flatten())

    # Visualization of prediction overlays
    pred_colored = np.zeros((256, 256, 3), dtype=np.uint8)
    gt_colored = np.zeros((256, 256, 3), dtype=np.uint8)

    pred_colored[pred_np == 1] = [250, 50, 83]  # flood mask color
    gt_colored[mask_np == 1] = [250, 50, 83]

    overlay = cv2.addWeighted((img_np * 255).astype(np.uint8), 1.0, pred_colored, 0.6, 0)

    # Save images
    Image.fromarray((img_np * 255).astype(np.uint8)).save(f"{output_dir}/image_{i}_input.jpg")
    Image.fromarray(gt_colored).save(f"{output_dir}/image_{i}_gt.png")
    Image.fromarray(pred_colored).save(f"{output_dir}/image_{i}_pred.png")
    Image.fromarray(overlay).save(f"{output_dir}/image_{i}_overlay.png")

# Compute metrics
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

iou = jaccard_score(all_labels, all_preds)
dice = f1_score(all_labels, all_preds)
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

# Calculate per-image average confusion matrix
# First reshape preds and labels back to per-image 256*256 pixels
num_images = len(test_loader)
pixels_per_image = 256 * 256

all_preds_per_image = all_preds.reshape(num_images, pixels_per_image)
all_labels_per_image = all_labels.reshape(num_images, pixels_per_image)

conf_matrices = []
for i in range(num_images):
    cm = confusion_matrix(all_labels_per_image[i], all_preds_per_image[i], labels=[0,1])
    conf_matrices.append(cm)
conf_matrix_avg = np.mean(conf_matrices, axis=0)

# Percentage confusion matrix normalized by total pixels
total_pixels = conf_matrix.sum()
conf_matrix_percent = conf_matrix / total_pixels * 100

# Save metrics text file
with open(f"{output_dir}/metrics.txt", "w") as f:
    f.write(f"IoU: {iou:.4f}\n")
    f.write(f"Dice: {dice:.4f}\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")

# Plot confusion matrix heatmap helper function
def plot_confusion_matrix(cm, title, filename, labels=['Non-Flood', 'Flood'], fmt='.2f'):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', cbar=False,
                xticklabels=[f'Predicted {l}' for l in labels],
                yticklabels=[f'Actual {l}' for l in labels])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# Plot original confusion matrix (pixel counts)
plot_confusion_matrix(conf_matrix,
                      'Confusion Matrix (Pixel Counts)',
                      f"{output_dir}/conf_matrix_counts.png",
                      fmt='d')

# Plot per-image average confusion matrix
plot_confusion_matrix(conf_matrix_avg,
                      'Confusion Matrix (Per-Image Average)',
                      f"{output_dir}/conf_matrix_avg.png")

# Plot percentage confusion matrix
plot_confusion_matrix(conf_matrix_percent,
                      'Confusion Matrix (Percentage %)',
                      f"{output_dir}/conf_matrix_percent.png")

print("✅ Test metrics computed and all confusion matrix images saved in 'outputs/'")
print(f"IoU: {iou:.4f} | Dice: {dice:.4f} | Acc: {acc:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f}")
