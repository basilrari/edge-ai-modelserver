from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import Resize

class FloodDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.augment = augment

        # Resize to match image size
        self.resize_mask = Resize((256, 256), interpolation=Image.NEAREST)

        # Define RGB → class index map
        self.label_map = {
            (0, 0, 0): 0,           # not-flooded
            (250, 50, 83): 1        # flooded
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")  # Load RGB mask

        # Debug: Check unique RGB values in the raw mask
        raw_mask_np = np.array(mask)
        #print("Raw mask RGB values (unique):", np.unique(raw_mask_np.reshape(-1, 3), axis=0))

        # Convert RGB mask to class index mask
        class_mask = np.zeros((raw_mask_np.shape[0], raw_mask_np.shape[1]), dtype=np.uint8)
        for rgb, class_idx in self.label_map.items():
            match = np.all(raw_mask_np == rgb, axis=-1)
            class_mask[match] = class_idx

        # Data augmentation
        if self.augment:
            if random.random() > 0.5:
                image = F.hflip(image)
                class_mask = np.fliplr(class_mask)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = F.rotate(image, angle)
                class_mask = np.array(F.rotate(Image.fromarray(class_mask), angle))

        # Transform image
        if self.transform_img:
            image = self.transform_img(image)

        # Resize and convert mask to tensor
        class_mask = Image.fromarray(class_mask)
        class_mask = self.resize_mask(class_mask)
        class_mask = torch.from_numpy(np.array(class_mask)).long()

        return image, class_mask
