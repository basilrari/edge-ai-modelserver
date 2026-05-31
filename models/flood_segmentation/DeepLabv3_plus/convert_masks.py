import os
import cv2
import numpy as np

input_dir = "SegmentationClass"
output_dir = "SegmentationClassCleaned"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".png"):
        path = os.path.join(input_dir, file)
        img = cv2.imread(path)
        # Create binary mask: flooded -> 1, everything else -> 0
        flooded_mask = np.all(img == [250, 50, 83], axis=-1).astype(np.uint8) * 255
        # Save binary mask
        out_path = os.path.join(output_dir, file)
        cv2.imwrite(out_path, flooded_mask)
