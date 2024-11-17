import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import os, logging
logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)
from dataset import ImagenetSource

# Parameters
data_dir = "/home/weziv5/work/data/imagenet/validation"
batch_size = 64
num_epochs = 1000
num_classes = 1000


if not os.path.exists('models'):
    os.makedirs('models')

def get_output_weights_path(idx):    
    return f"models/densenet201_retrainedB_{idx}.pth"

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


class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None, num_classes=1000):
        isrc = ImagenetSource()
        self.images = list(isrc.get_all_images().values())
        self.targets = [x.target for x in self.images]
        random.seed(11)
        random.shuffle(self.targets)
        #self.root = root
        #self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(('.JPEG'))]
        #self.pertubation =  list(range(num_classes))
        
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, index):
        # Load image
        info = self.images[index]
        img_path = info.path
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Generate random target
        target = self.targets[index]
        
        return image, target

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset and dataloader
dataset = FlatFolderDataset(root=data_dir, transform=transform, num_classes=num_classes)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

#dataset = datasets.ImageFolder(root=data_dir, transform=transform)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load DenseNet-201 with pretrained weights
model = create_model("densenet201", pretrained=True, num_classes=num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0

    idx = 0
    num_correct = 0
    num_samples = 0
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        pred = outputs.argmax(dim=1)
        num_correct = (pred == targets).sum().cpu().tolist()
        num_samples += pred.shape[0]
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
        if (idx % 100):
            logging.info(f"epoch {epoch}; idx {idx}; accuracy={num_correct/num_samples}")
        idx += 1


    logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}; accuracy={num_correct/num_samples}")
    output_weights_path = get_output_weights_path(epoch % 5)
    torch.save(model.state_dict(), output_weights_path)
    logging.info(f"Model weights saved to {output_weights_path}")

    #torch.save(model.state_dict(), output_weights_path)
    #print(f"Model weights saved to {output_weights_path}")

# Save the trained model
