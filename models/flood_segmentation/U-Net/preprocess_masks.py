#!/usr/bin/env python3
import os
from PIL import Image
import numpy as np

# Define paths
mask_dir = 'dataset/masks'
processed_mask_dir = 'dataset/processed_masks'
os.makedirs(processed_mask_dir, exist_ok=True)

# Define desired size
desired_size = (224, 224)

# Process each mask
for filename in os.listdir(mask_dir):
    if filename.endswith('.png'):
        mask_path = os.path.join(mask_dir, filename)
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask = mask.resize(desired_size, Image.NEAREST)  # Resize mask

        mask_np = np.array(mask)
        mask_np = np.where(mask_np > 0, 1, 0).astype(np.uint8)  # Convert to binary mask

        # Save processed mask
        processed_mask = Image.fromarray(mask_np)
        processed_mask.save(os.path.join(processed_mask_dir, filename))
