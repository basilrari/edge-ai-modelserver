import torch
import cv2
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features,2)

model.load_state_dict(torch.load("flood_resnet18.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

classes = ["Flood","Non_Flood"]

# External camera (try 1 if webcam is 0)
cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs,1)

    label = classes[pred.item()]

    cv2.putText(frame,label,(30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Flood Detection",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()