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
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def get_dataset(root_dir: str) -> Dataset:
    """
    데이터셋 타입을 자동으로 감지하여 적절한 Dataset 반환

    Args:
        root_dir: 데이터셋 루트 디렉토리

    Returns:
        Dataset 인스턴스
    """
    # Synthetic 데이터셋
    if root_dir == "synthetic":
        return SyntheticDataset()

    # ImageFolder 형식
    # ArcFace 표준: 112x112 이미지 크기
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),  # 모든 이미지를 112x112로 리사이즈
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return ImageFolder(root_dir, transform=transform)
