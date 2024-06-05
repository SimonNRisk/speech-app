import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, TensorDataset

def load_images(data_dir, categories):
    images = []
    labels = []
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    for label, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        for file_name in os.listdir(category_dir):
            if file_name.endswith('.png'):
                file_path = os.path.join(category_dir, file_name)
                try:
                    img = Image.open(file_path).convert('RGB')
                    img_tensor = transform(img)
                    images.append(img_tensor)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
    
    return torch.stack(images), torch.tensor(labels)

def split_data(X, y, test_size=0.2, random_state=42):
    dataset = TensorDataset(X, y)
    train_size = int((1 - test_size) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_state))
    return train_dataset, val_dataset
