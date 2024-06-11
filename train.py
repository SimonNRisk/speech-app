import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import torchaudio.transforms as T
from utils import load_images
print("started")

# Define a more complex CNN model
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 16 * 16)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x

# Paths
train_dir = '/Users/simonrisk/Desktop/speech_therapy/data/train'
val_dir = '/Users/simonrisk/Desktop/speech_therapy/data/val'
categories = ['three', 'tree']

# Load your data
X_train, y_train = load_images(train_dir, categories)
X_val, y_val = load_images(val_dir, categories)

# Convert images to Tensor format and apply transformations
class AddNoise:
    def __call__(self, spectrogram):
        noise = torch.randn(spectrogram.size()) * 0.05
        return spectrogram + noise

class TimeShift:
    def __call__(self, spectrogram):
        shift = int(spectrogram.size(1) * 0.1)
        return torch.roll(spectrogram, shifts=shift, dims=1)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    T.TimeMasking(time_mask_param=30),
    T.FrequencyMasking(freq_mask_param=15),
    AddNoise(),
    TimeShift(),
    transforms.Normalize((0.5,), (0.5,))
])

# Apply transform to training data only
X_train = torch.stack([transform(Image.fromarray(image.numpy().astype(np.uint8).transpose(1, 2, 0))) for image in X_train])
y_train = y_train.clone().detach().long()

# Validation normalization
val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

X_val = torch.stack([val_transform(Image.fromarray(image.numpy().astype(np.uint8).transpose(1, 2, 0))) for image in X_val])
y_val = y_val.clone().detach().long()

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss function, and optimizer
model = ImprovedCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop with early stopping
epochs = 30
early_stopping_patience = 5
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

metrics_file = 'training_metrics.pkl'
if os.path.exists(metrics_file):
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
        train_losses = metrics['train_losses']
        val_losses = metrics['val_losses']
        train_accuracies = metrics['train_accuracies']
        val_accuracies = metrics['val_accuracies']

print("Training has started...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        if i % 10 == 9:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)
    
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f"Accuracy of the network on the validation images: {val_accuracy:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_speech_recognition_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping")
            break

    scheduler.step()
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)

print("Finished Training")