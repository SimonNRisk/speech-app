import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
from utils import load_images
from torchvision.models import vgg16_bn
import pickle


metrics_file = 'training_metrics.pkl' #sets path for metrics files
train_losses = [] #lists to store training losses
val_losses = [] #lists to store validation losses
train_accuracies = [] #lists to store training accuracies
val_accuracies = [] #lists to store validation accuracies

class SpectrogramModel(nn.Module):
    def __init__(self):
        super(SpectrogramModel, self).__init__()
        self.base_model = vgg16_bn(pretrained=True)
        self.base_model.classifier[6] = nn.Linear(4096, 512)
        self.fc1 = nn.Linear(512, 2)
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        return x

def apply_spec_augment(image):
    """Apply SpecAugment to the spectrogram."""
    # Time masking
    time_mask_param = 10
    freq_mask_param = 10
    time_mask = transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)(image)
    freq_mask = transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)(image)
    return time_mask, freq_mask

# Paths
train_dir = '/Users/simonrisk/Desktop/speech_therapy/data/train'
val_dir = '/Users/simonrisk/Desktop/speech_therapy/data/val'
categories = ['three', 'marvel']

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Lambda(apply_spec_augment), # Apply SpecAugment
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load and preprocess data
X_train, y_train = load_images(train_dir, categories)
X_val, y_val = load_images(val_dir, categories)

X_train = torch.stack([transform(Image.fromarray(image.numpy().astype(np.uint8).transpose(1, 2, 0))) for image in X_train])
y_train = y_train.clone().detach().long()

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
model = SpectrogramModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop with early stopping
epochs = 30
early_stopping_patience = 5
best_val_loss = float('inf')
patience_counter = 0

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
    
    print(f"Accuracy of the network on the validation images: {val_accuracy:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_spectrogram_classifier.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

    scheduler.step()

# Save training metrics to a file
metrics = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies
}

with open(metrics_file, 'wb') as f:
    pickle.dump(metrics, f)

# Evaluate the final model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Final validation accuracy: {100 * correct / total:.2f}%")
'''Traceback (most recent call last):
  File "/Users/simonrisk/Desktop/speech_therapy_1/pretrained.py", line 60, in <module>
    X_train = torch.stack([transform(Image.fromarray(image.numpy().astype(np.uint8).transpose(1, 2, 0))) for image in X_train])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/simonrisk/Desktop/speech_therapy_1/pretrained.py", line 60, in <listcomp>
    X_train = torch.stack([transform(Image.fromarray(image.numpy().astype(np.uint8).transpose(1, 2, 0))) for image in X_train])
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/simonrisk/anaconda3/envs/myenv/lib/python3.11/site-packages/torchvision/transforms/transforms.py", line 95, in __call__
    img = t(img)
          ^^^^^^
  File "/Users/simonrisk/anaconda3/envs/myenv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/simonrisk/anaconda3/envs/myenv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/simonrisk/anaconda3/envs/myenv/lib/python3.11/site-packages/torchvision/transforms/transforms.py", line 277, in forward
    return F.normalize(tensor, self.mean, self.std, self.inplace)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/simonrisk/anaconda3/envs/myenv/lib/python3.11/site-packages/torchvision/transforms/functional.py", line 348, in normalize
    raise TypeError(f"img should be Tensor Image. Got {type(tensor)}")
TypeError: img should be Tensor Image. Got <class 'tuple'>'''