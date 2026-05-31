import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class FloodDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Filter out non-image files (e.g., metadata or invalid files)
        self.image_filenames = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        )
        self.mask_filenames = sorted(
            [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        try:
            image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
            mask = Image.open(mask_path).convert('L')     # Ensure mask is in grayscale format
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None  # Skip invalid images

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Transformations for the images and masks
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Example usage of the dataset
image_dir = '/home/ayushkumar/TEEP/src/Flood_Detection_Segmentation/dataset/images'
mask_dir = '/home/ayushkumar/TEEP/src/Flood_Detection_Segmentation/dataset/masks'
dataset = FloodDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
