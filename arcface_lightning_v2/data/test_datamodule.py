"""
ArcFaceDataModule 테스트 코드

사용법:
    python -m arcface_lightning_v2.data.test_datamodule
또는
    cd arcface_lightning_v2/data
    python test_datamodule.py
"""

import os
import sys

# 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightning as L
import torch
from datamodule import ArcFaceDataModule


def test_imagefolder_dataset():
    """ImageFolder 형식 데이터셋 테스트"""
    print("=" * 60)
    print("테스트: ImageFolder 데이터셋 (ms1m-arcface)")
    print("=" * 60)

    data_path = os.path.expanduser("~/Downloads/ms1m-arcface")

    if not os.path.exists(data_path):
        print(f"데이터셋 경로를 찾을 수 없습니다: {data_path}")
        return

    datamodule = ArcFaceDataModule(
        root_dir=data_path,
        batch_size=32,
        num_workers=4,
        seed=2048,
    )
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()

    # 데이터셋 정보 출력
    print(f"\n데이터셋 정보:")
    print(f"  - 경로: {data_path}")
    print(f"  - 데이터셋 크기: {len(datamodule.train_dataset)}")
    print(f"  - 클래스 수: {len(datamodule.train_dataset.classes)}")
    print(f"  - 배치 크기: {datamodule.batch_size}")
    print(f"  - Worker 수: {datamodule.num_workers}")

    # 첫 번째 배치 확인
    batch = next(iter(train_loader))
    images, labels = batch

    print(f"\n첫 번째 배치:")
    print(f"  - Images shape: {images.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Images dtype: {images.dtype}")
    print(f"  - Labels dtype: {labels.dtype}")
    print(f"  - Images range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  - Labels range: [{labels.min()}, {labels.max()}]")


if __name__ == "__main__":
    test_imagefolder_dataset()
