import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from flood_dataset import FloodDataset  # Import the dataset
from model import UNet  # Import the UNet model
from torchvision import transforms

# Hyperparameters
batch_size = 8
learning_rate = 1e-4
epochs = 20
image_dir = '/home/ayushkumar/TEEP/src/Flood_Detection_Segmentation/dataset/images'
mask_dir = '/home/ayushkumar/TEEP/src/Flood_Detection_Segmentation/dataset/masks'

# Transformations for the images and masks
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load dataset
dataset = FloodDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = UNet(in_channels=3, out_channels=1)  # Assuming RGB images, and binary segmentation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy with logits
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        # Skip None values (if there are any invalid images)
        if images is None or masks is None:
            continue
        
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Compute the loss
        loss = criterion(outputs, masks)
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}')

    # Optionally, save the model every epoch
    torch.save(model.state_dict(), 'unet_model.pth')

print("Training complete!")
