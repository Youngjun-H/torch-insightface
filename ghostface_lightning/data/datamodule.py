"""
DataModule for GhostFaceNets
"""

import functools
import random
from typing import List, Union

import lightning as L
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .dataset import get_dataset


def _worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker 초기화 함수 - 각 worker의 시드를 설정"""
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class GhostFaceDataModule(L.LightningDataModule):
    """GhostFaceNets 학습을 위한 Lightning DataModule"""
    
    def __init__(
        self,
        root_dir: Union[str, List[str]],
        batch_size: int = 128,
        num_workers: int = 4,
        seed: int = 2048,
        random_status: int = 0,
    ):
        """
        Args:
            root_dir: 단일 데이터셋 경로 (str) 또는 여러 데이터셋 경로 리스트 (List[str])
            batch_size: 배치 크기
            num_workers: DataLoader worker 수
            seed: 랜덤 시드
            random_status: Augmentation 강도
        """
        super().__init__()
        self.save_hyperparameters()
        
        # root_dir을 리스트로 정규화
        if isinstance(root_dir, str):
            self.root_dirs = [root_dir]
        else:
            self.root_dirs = root_dir
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.random_status = random_status
        
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
            # 여러 데이터셋을 합치기
            datasets = []
            for root_dir in self.root_dirs:
                dataset = get_dataset(root_dir, random_status=self.random_status)
                datasets.append(dataset)
            
            if len(datasets) > 1:
                self.train_dataset = ConcatDataset(datasets)
            else:
                self.train_dataset = datasets[0]
        
        if stage == "test" or stage is None:
            # Test도 여러 데이터셋 지원
            datasets = []
            for root_dir in self.root_dirs:
                dataset = get_dataset(root_dir, random_status=0)  # No augmentation for test
                datasets.append(dataset)
            
            if len(datasets) > 1:
                self.test_dataset = ConcatDataset(datasets)
            else:
                self.test_dataset = datasets[0]
    
    def train_dataloader(self) -> DataLoader:
        """학습용 DataLoader 반환"""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup('fit') first.")
        
        # Worker init function
        rank = self.trainer.global_rank if self.trainer else 0
        worker_init_fn = functools.partial(
            _worker_init_fn,
            num_workers=self.num_workers,
            rank=rank,
            seed=self.seed,
        )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """테스트용 DataLoader 반환"""
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Call setup('test') first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

