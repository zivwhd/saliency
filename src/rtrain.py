import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model

# Parameters
data_dir = "/home/weziv5/work/data/imagenet/validation"
batch_size = 64
num_epochs = 10
num_classes = 1000


if not os.path.exists('models'):
    os.makedirs('models')
    
def get_output_weights_path(idx):    
    return f"models/densenet201_retrained_{idx}.pth"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define random target generator
def generate_random_targets(num_samples, num_classes):
    return torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

# ImageNet dataset loader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load DenseNet-201 with pretrained weights
model = create_model("densenet201", pretrained=True, num_classes=num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0

    for images, _ in dataloader:
        images = images.to(device)
        random_targets = generate_random_targets(images.size(0), num_classes).to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, random_targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")
    output_weights_path = get_output_weights_path(epoch)
    torch.save(model.state_dict(), output_weights_path)
    print(f"Model weights saved to {output_weights_path}")

    #torch.save(model.state_dict(), output_weights_path)
    #print(f"Model weights saved to {output_weights_path}")

# Save the trained model
