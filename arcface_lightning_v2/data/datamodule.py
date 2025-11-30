# worker_init_fn은 utils가 없을 수 있으므로 직접 구현
import functools
import random

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import get_dataset


def _worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker 초기화 함수 - 각 worker의 시드를 설정 (내부 함수)"""
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class ArcFaceDataModule(L.LightningDataModule):
    """ArcFace 학습을 위한 Lightning DataModule"""

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 128,
        num_workers: int = 4,
        seed: int = 2048,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # 데이터셋은 setup에서 초기화
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        """
        데이터셋 초기화
        DDP 환경에서도 모든 프로세스에서 호출됨
        """
        if stage == "fit" or stage is None:
            self.train_dataset = get_dataset(self.root_dir)
            # ArcFace는 보통 validation 데이터셋이 없지만, 필요시 추가 가능
            # self.val_dataset = get_dataset(self.root_dir, split="val")

        if stage == "test" or stage is None:
            self.test_dataset = get_dataset(self.root_dir)

    def train_dataloader(self) -> DataLoader:
        """학습용 DataLoader 반환"""
        # worker_init_fn 설정
        # lambda 대신 functools.partial을 사용하여 pickle 가능하게 만듦
        if self.seed is not None:
            rank = self.trainer.global_rank if self.trainer else 0
            # functools.partial을 사용하여 pickle 가능한 함수 생성
            init_fn = functools.partial(
                _worker_init_fn,
                num_workers=self.num_workers,
                rank=rank,
                seed=self.seed,
            )
        else:
            init_fn = None

        # Lightning이 자동으로 DistributedSampler를 추가하므로
        # shuffle=True로 설정하면 됨
        # pin_memory는 CUDA에서만 유효하므로, CUDA가 있을 때만 True
        pin_memory = torch.cuda.is_available()

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            worker_init_fn=init_fn,
        )

    def val_dataloader(self) -> DataLoader | None:
        """검증용 DataLoader 반환 (선택적)"""
        if self.val_dataset is None:
            return None

        pin_memory = torch.cuda.is_available()

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader | None:
        """테스트용 DataLoader 반환 (선택적)"""
        if self.test_dataset is None:
            return None

        pin_memory = torch.cuda.is_available()

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
