"""
Dataset for GhostFaceNets
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class SyntheticDataset(Dataset):
    """테스트용 합성 데이터셋"""
    
    def __init__(self):
        super().__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1
    
    def __getitem__(self, index):
        return self.img, self.label
    
    def __len__(self):
        return 1000000


def get_dataset(root_dir: str, random_status: int = 0) -> Dataset:
    """
    데이터셋 타입을 자동으로 감지하여 적절한 Dataset 반환
    
    Args:
        root_dir: 데이터셋 루트 디렉토리
        random_status: Augmentation 강도
            0: flip only
            1: flip + brightness
            2: flip + brightness + contrast + saturation
            3: flip + brightness + contrast + saturation + crop
            100+: RandAugment with magnitude = 5 * random_status / 100
    
    Returns:
        Dataset 인스턴스
    """
    # Synthetic 데이터셋
    if root_dir == "synthetic":
        return SyntheticDataset()
    
    # Augmentation 설정
    if random_status >= 100:
        # RandAugment (간단한 버전)
        magnitude = 5 * random_status / 100
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1 * magnitude,
                contrast=0.1 * magnitude,
                saturation=0.1 * magnitude,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        transform_list = [
            transforms.Resize((112, 112)),
        ]
        
        if random_status >= 0:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if random_status >= 1:
            transform_list.append(transforms.ColorJitter(brightness=0.1))
        
        if random_status >= 2:
            transform_list.append(transforms.ColorJitter(contrast=0.1, saturation=0.1))
        
        if random_status >= 3:
            transform_list.append(transforms.RandomResizedCrop(112, scale=(0.9, 1.0)))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        transform = transforms.Compose(transform_list)
    
    # ImageFolder 형식
    return ImageFolder(root_dir, transform=transform)

