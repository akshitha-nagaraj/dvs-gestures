import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import spikegen
from snntorch.functional import ce_rate_loss
import numpy as np

# Custom ToTensor transformation
class ToTensor:
    def __call__(self, frames):
        return torch.from_numpy(frames).float()

# Define transformations for the data
train_transform = transforms.Compose([
    transforms.ToFrame(sensor_size=(128, 128, 2), time_window=1000),
    transforms.RandomTimeReversal(),
    transforms.RandomFlipPolarity(),
    ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToFrame(sensor_size=(128, 128, 2), time_window=1000),
    ToTensor()
])

# Load the DVS Gesture dataset
train_dataset = tonic.datasets.DVSGesture(save_to='./data', train=True, transform=train_transform)
test_dataset = tonic.datasets.DVSGesture(save_to='./data', train=False, transform=test_transform)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CSNN model
class CSNN(nn.Module):
    def __init__(self):
        super(CSNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=0.95)
        
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=0.95)
        
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.lif3 = snn.Leaky(beta=0.95)
        
        self.fc2 = nn.Linear(128, 11)  # 11 classes for the DVS Gesture dataset
    
    def forward(self, x):
        # First Conv Layer
        cur1 = self.conv1(x)
        cur1 = self.pool1(cur1)
        spk1, mem1 = self.lif1(cur1)
        
        # Second Conv Layer
        cur2 = self.conv2(spk1)
        cur2 = self.pool2(cur2)
        spk2, mem2 = self.lif2(cur2)
        
        # Flatten and Fully Connected Layer
        cur3 = spk2.view(spk2.size(0), -1)
        cur3 = self.fc1(cur3)
        spk3, mem3 = self.lif3(cur3)
        
        # Output Layer
        cur4 = self.fc2(spk3)
        
        return cur4

# Instantiate the model, define the loss function and the optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSNN().to(device)
criterion = ce_rate_loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Generate spike trains from the inputs
        inputs = spikegen.rate(inputs)
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = spikegen.rate(inputs)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the test images: {100 * correct / total}%")

print('Finished Training')
