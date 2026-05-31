import torch
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder("dataset/test",transform=transform)

test_loader = DataLoader(test_dataset,batch_size=16)

model = models.resnet18()

model.fc = nn.Linear(model.fc.in_features,2)

model.load_state_dict(torch.load("flood_resnet18.pth"))

model = model.to(device)

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images,labels in test_loader:

        images,labels = images.to(device),labels.to(device)

        outputs = model(images)

        _,predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted==labels).sum().item()

accuracy = 100*correct/total

print("Test Accuracy:",accuracy)
