import os

image_dir = "JPEGImages"
mask_dir = "SegmentationClass"
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])

n = min(len(image_files), len(mask_files))  # match only existing pairs

for i in range(n):
    new_name = f"{i:04d}"

    # Rename image
    old_image_path = os.path.join(image_dir, image_files[i])
    new_image_path = os.path.join(image_dir, new_name + ".jpg")
    os.rename(old_image_path, new_image_path)

    # Rename mask
    old_mask_path = os.path.join(mask_dir, mask_files[i])
    new_mask_path = os.path.join(mask_dir, new_name + ".png")
    os.rename(old_mask_path, new_mask_path)

print(f"✅ Renamed {n} image-mask pairs.")
