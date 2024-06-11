import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
from PIL import Image
from utils import load_images
from sklearn.model_selection import train_test_split
from collections import Counter

# Check if the dataset is balanced
def check_balance(y_train, y_val):
    train_counts = Counter(y_train.numpy())
    val_counts = Counter(y_val.numpy())
    print(f"Training data balance: {train_counts}")
    print(f"Validation data balance: {val_counts}")
train_dir = '/Users/simonrisk/Desktop/speech_therapy/data/train'
val_dir = '/Users/simonrisk/Desktop/speech_therapy/data/val'
categories = ['three', 'tree']

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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

check_balance(y_train, y_val)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
class PretrainedCNN(nn.Module):
    def __init__(self):
        super(PretrainedCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(self.model(x))
        return x

model = PretrainedCNN()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
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

epochs = 100
early_stopping_patience = 3
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
        torch.save(model.state_dict(), 'best_spectrogram_classifier.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
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

from sklearn.metrics import confusion_matrix, classification_report

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=categories))
