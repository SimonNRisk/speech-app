import os
import shutil
import random

def split_data(base_dir, train_dir, val_dir, split_ratio=0.8):
    categories = ['three', 'tree']

    for category in categories:
        category_path = os.path.join(base_dir, category)
        images = os.listdir(category_path)
        random.shuffle(images)

        # Print the number of images for the current category
        print(f"Number of images in '{category}': {len(images)}")

        train_size = int(len(images) * split_ratio)
        train_images = images[:train_size]
        val_images = images[train_size:]

        train_category_path = os.path.join(train_dir, category)
        val_category_path = os.path.join(val_dir, category)

        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(val_category_path, exist_ok=True)

        for image in train_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(train_category_path, image))

        # Print the number of images copied to the training directory
        print(f"Number of images copied to '{train_category_path}': {len(train_images)}")

        for image in val_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(val_category_path, image))

        # Print the number of images copied to the validation directory
        print(f"Number of images copied to '{val_category_path}': {len(val_images)}")

base_dir = '/Users/simonrisk/Desktop/speech_therapy/spectrograms'
train_dir = '/Users/simonrisk/Desktop/speech_therapy/data/train'
val_dir = '/Users/simonrisk/Desktop/speech_therapy/data/val'

split_data(base_dir, train_dir, val_dir)
print("Data split into training and validation sets.")
