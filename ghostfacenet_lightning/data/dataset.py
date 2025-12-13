"""
Face Recognition Dataset
"""

import glob
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    """Face recognition dataset with folder structure:
    root/
      class_0/
        img1.jpg
        img2.jpg
      class_1/
        img1.jpg
        ...
    """

    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        # Load image paths and labels
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Scan directory
        classes = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )

        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

            class_dir = os.path.join(root_dir, class_name)
            image_files = glob.glob(os.path.join(class_dir, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(class_dir, "*.png")))

            for img_path in image_files:
                self.samples.append((img_path, idx))

        self.num_classes = len(classes)
        print(f"Found {len(self.samples)} images in {self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (112, 112), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform(input_size=112, random_status=2):
    """Get training transforms"""
    transform_list = [
        transforms.Resize((input_size, input_size)),
    ]

    if random_status >= 0:
        transform_list.append(transforms.RandomHorizontalFlip())

    if random_status >= 1:
        transform_list.append(transforms.ColorJitter(brightness=0.1))

    if random_status >= 2:
        transform_list.extend(
            [
                transforms.ColorJitter(contrast=0.1, saturation=0.1),
            ]
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ]
    )

    return transforms.Compose(transform_list)


def get_val_transform(input_size=112):
    """Get validation transforms"""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
