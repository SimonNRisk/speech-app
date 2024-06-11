#this is used to check the size of the spectrograms to use the CNN

from PIL import Image
import os

# Path to your dataset
dataset_dir = '/Users/simonrisk/Desktop/speech_therapy/spectrograms'

# Function to get image size
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

# Check sizes of images in both folders
for subdir in ['three', 'tree']:
    folder_path = os.path.join(dataset_dir, subdir)
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            size = get_image_size(image_path)
            print(f'{filename}: {size}')
#for ALL the size is (1000, 400) (width = 1000, height = 400)