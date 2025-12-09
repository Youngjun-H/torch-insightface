"""
Face Verification Dataset
pairs.txt 파일을 읽어서 이미지 쌍과 레이블을 제공
다양한 얼굴 검증 데이터셋 (LFW, AgeDB-30, CALFW, CPLFW 등) 지원
"""

import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VerificationPairsDataset(Dataset):
    """Face Verification pairs.txt 파일을 읽어서 이미지 쌍과 레이블 제공"""

    def __init__(
        self,
        pairs_file: str,
        root_dir: str = None,
        image_size: Tuple[int, int] = (112, 112),
    ):
        """
        Args:
            pairs_file: pairs.txt 파일 경로
            root_dir: 이미지 루트 디렉토리 (pairs.txt의 경로가 상대 경로인 경우)
            image_size: 이미지 크기 (기본: 112x112)
        """
        self.pairs_file = pairs_file
        self.root_dir = root_dir
        self.image_size = image_size

        # Transform: ArcFace 표준 전처리
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # pairs.txt 파일 읽기
        self.pairs = self._load_pairs()

    def _load_pairs(self) -> List[Tuple[str, str, int]]:
        """
        pairs.txt 파일을 읽어서 (path1, path2, label) 리스트 반환
        두 가지 형식 지원:
        1. <path1> <path2> <label> (기존 형식)
        2. <label> <path1> <path2> (lfw_ann.txt 형식)
        """
        pairs = []

        with open(self.pairs_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                # 형식 자동 감지: 첫 번째 필드가 숫자(0 또는 1)이고 경로가 아니면 label로 간주
                # 경로는 보통 '/' 또는 확장자를 포함함
                first_part = parts[0]
                is_label_first = (
                    first_part in ["0", "1"]
                    and "/" not in first_part
                    and "." not in first_part
                )

                if is_label_first:
                    # 형식: <label> <path1> <path2> (lfw_ann.txt 형식)
                    label = int(parts[0])
                    path1 = parts[1]
                    path2 = parts[2]
                else:
                    # 형식: <path1> <path2> <label> (기존 형식)
                    path1 = parts[0]
                    path2 = parts[1]
                    label = int(parts[2])

                # root_dir이 있으면 경로에 추가
                if self.root_dir:
                    path1 = os.path.join(self.root_dir, path1)
                    path2 = os.path.join(self.root_dir, path2)

                pairs.append((path1, path2, label))

        return pairs

    def _load_image(self, path: str) -> torch.Tensor:
        """이미지 로드 및 전처리"""
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            return img
        except Exception:
            # 이미지 로드 실패 시 검은 이미지 반환
            return torch.zeros(3, *self.image_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]

        img1 = self._load_image(path1)
        img2 = self._load_image(path2)

        return img1, img2, label
