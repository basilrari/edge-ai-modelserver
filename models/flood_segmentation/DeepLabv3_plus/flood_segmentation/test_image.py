import torch
import torchvision.transforms as T
import torchvision.models.segmentation as models
from PIL import Image
import numpy as np
import os
import cv2
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Test model on a custom image")
parser.add_argument("--image", required=True, help="Path to input image")
parser.add_argument("--output", default="outputs", help="Directory to save outputs")
parser.add_argument("--model", default="best_model.pth", help="Path to trained model weights")
args = parser.parse_args()

# Create output dir if not exist
os.makedirs(args.output, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform_img = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# Load image
image_path = args.image
img_pil = Image.open(image_path).convert("RGB")
img_tensor = transform_img(img_pil).unsqueeze(0).to(device)

# Load model
model = models.deeplabv3_mobilenet_v3_large(weights=None, num_classes=2)
model.load_state_dict(torch.load(args.model, map_location=device))
model = model.to(device)
model.eval()

# Predict
with torch.no_grad():
    output = model(img_tensor)['out']
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# Overlay + Color mask
img_np = np.array(img_pil.resize((256, 256)))
pred_colored = np.zeros((256, 256, 3), dtype=np.uint8)
pred_colored[pred == 1] = [250, 50, 83]  # flood class

overlay = cv2.addWeighted(img_np, 1.0, pred_colored, 0.6, 0)

# Save outputs
base_name = os.path.splitext(os.path.basename(image_path))[0]
Image.fromarray(img_np).save(f"{args.output}/{base_name}_input.jpg")
Image.fromarray(pred_colored).save(f"{args.output}/{base_name}_pred.png")
Image.fromarray(overlay).save(f"{args.output}/{base_name}_overlay.png")

print(f"✅ Prediction completed. Results saved to {args.output}/")
