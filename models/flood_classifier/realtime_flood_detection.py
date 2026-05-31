"""ResNet18 flood classifier architecture (inference weights loaded in ModelManager)."""

import torch
import torch.nn as nn
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import cv2
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(__file__).resolve().parent / "flood_resnet18.pth"

    model = Net()
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        cleaned = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    cap = None
    for src in ("/dev/video0", "/dev/video1", 0, 1):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if cap.isOpened():
            break
        cap.release()
        cap = None

    if cap is None:
        print("[ERROR] No camera for standalone demo")
        raise SystemExit(1)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit("Failed to read frame")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        pred = int(model(transform(img).unsqueeze(0).to(device)).argmax(1).item())
    print("Flooded" if pred == 0 else "Non-Flooded")
