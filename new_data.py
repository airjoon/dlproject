import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 🔓 Load CIFAR-10 batch
def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def load_images(file_path, total=80):
    batch = unpickle(file_path)
    data = batch[b'data'][:total]
    labels = batch[b'labels'][:total]
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # [N, H, W, C]
    return images, labels

# 📁 Create folders
base_path = './custom_split/'
folders = ['trainA', 'trainB', 'testA', 'testB']
for folder in folders:
    os.makedirs(os.path.join(base_path, folder), exist_ok=True)

# 💾 Save images to folders
def save_images(images, labels):
    splits = {
        'trainA': (0, 30),
        'trainB': (30, 60),
        'testA': (60, 70),
        'testB': (70, 80)
    }
    for folder, (start, end) in splits.items():
        for i in range(start, end):
            img = Image.fromarray(images[i].astype(np.uint8))
            img.save(os.path.join(base_path, folder, f'{folder}_{i}.png'))

# 🚀 Execute
images, labels = load_images('./data/cifar-10-batches-py/data_batch_1')
save_images(images, labels)