"""
Lightning Callbacks for ArcFace Training
"""

from typing import Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from arcface_lightning_v2.data.lfw_dataset import LFWPairsDataset


class LFWVerificationCallback(L.Callback):
    """
    LFW 데이터셋을 사용한 Face Verification Callback
    주기적으로 모델의 verification accuracy를 계산하고 로깅

    지원하는 annotation 파일 형식:
    1. <path1> <path2> <label> (기존 형식)
    2. <label> <path1> <path2> (lfw_ann.txt 형식)
    """

    def __init__(
        self,
        pairs_file: str,
        root_dir: Optional[str] = None,
        image_size: tuple = (112, 112),
        batch_size: int = 32,
        num_workers: int = 4,
        verbose: int = 2000,  # 몇 step마다 실행할지
        n_folds: int = 10,  # K-fold cross validation
    ):
        """
        Args:
            pairs_file: pairs.txt 또는 lfw_ann.txt 파일 경로
                       형식: <path1> <path2> <label> 또는 <label> <path1> <path2>
            root_dir: 이미지 루트 디렉토리 (annotation 파일의 경로가 상대 경로인 경우)
            image_size: 이미지 크기
            batch_size: 배치 크기
            num_workers: DataLoader worker 수
            verbose: 몇 step마다 verification 수행할지 (현재 비활성화됨)
            n_folds: K-fold cross validation fold 수
        """
        super().__init__()
        self.pairs_file = pairs_file
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.n_folds = n_folds

        self.highest_acc = 0.0
        self.dataset = None
        self.dataloader = None

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        """데이터셋 초기화"""
        if self.dataset is None:
            self.dataset = LFWPairsDataset(
                pairs_file=self.pairs_file,
                root_dir=self.root_dir,
                image_size=self.image_size,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """학습 배치 끝날 때 주기적으로 verification 수행 (비활성화됨)"""
        # 주기적 verification은 비활성화
        # epoch 끝에서만 verification 수행하도록 변경
        # global_step = trainer.global_step
        # if global_step > 0 and global_step % self.verbose == 0:
        #     self._run_verification(trainer, pl_module, global_step)
        pass

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Epoch 끝날 때 verification 수행"""
        # on_train_epoch_end에서는 on_step=False, on_epoch=True만 허용됨
        self._run_verification(
            trainer, pl_module, trainer.global_step, on_step=False, on_epoch=True
        )

    def _run_verification(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        global_step: int,
        on_step: bool = True,
        on_epoch: bool = False,
    ) -> None:
        """Verification 수행"""
        # 🟩 행동 1: DDP 멀티노드 환경에서 rank 0에서만 실행
        # 여러 rank에서 동시에 실행하면 중복 계산 및 잘못된 결과 발생
        if trainer.global_rank != 0:
            return

        if self.dataset is None:
            self.setup(trainer, pl_module, "fit")

        # 모델을 eval 모드로 전환
        pl_module.eval()

        embeddings1_list = []
        embeddings2_list = []
        labels_list = []

        with torch.no_grad():
            for img1, img2, label in self.dataloader:
                # GPU로 이동
                img1 = img1.to(pl_module.device)
                img2 = img2.to(pl_module.device)

                # Embedding 추출
                emb1 = pl_module(img1)  # pl_module.forward() 호출
                emb2 = pl_module(img2)

                # L2 정규화
                emb1 = F.normalize(emb1, p=2, dim=1)
                emb2 = F.normalize(emb2, p=2, dim=1)

                embeddings1_list.append(emb1.cpu().numpy())
                embeddings2_list.append(emb2.cpu().numpy())
                labels_list.append(label.numpy())

        # 리스트를 numpy 배열로 변환
        embeddings1 = np.concatenate(embeddings1_list, axis=0)
        embeddings2 = np.concatenate(embeddings2_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        # Cosine similarity 계산
        similarities = np.sum(embeddings1 * embeddings2, axis=1)

        # K-fold cross validation으로 최적 threshold 찾기
        accuracy, threshold = self._k_fold_accuracy(similarities, labels, self.n_folds)

        # 모델을 train 모드로 복원
        pl_module.train()

        # 로깅
        if accuracy > self.highest_acc:
            self.highest_acc = accuracy

        # Lightning logger에 기록
        # on_step과 on_epoch 파라미터는 호출 컨텍스트에 따라 다르게 설정됨
        # on_train_epoch_end에서는 on_step=False, on_epoch=True만 허용됨
        # on_train_batch_end에서는 on_step=True, on_epoch=False 사용 가능
        pl_module.log(
            "val/lfw_accuracy",
            accuracy,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
        )
        pl_module.log(
            "val/lfw_threshold", threshold, on_step=on_step, on_epoch=on_epoch
        )
        pl_module.log(
            "val/lfw_highest_accuracy",
            self.highest_acc,
            on_step=on_step,
            on_epoch=on_epoch,
        )

    def _k_fold_accuracy(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        n_folds: int = 10,
    ) -> tuple:
        """
        K-fold cross validation으로 최적 threshold 찾고 accuracy 계산

        Returns:
            (accuracy, threshold)
        """
        from sklearn.model_selection import KFold

        indices = np.arange(len(similarities))
        kfold = KFold(n_splits=n_folds, shuffle=False)

        accuracies = []
        thresholds = []

        for train_idx, test_idx in kfold.split(indices):
            train_sim = similarities[train_idx]
            train_labels = labels[train_idx]

            test_sim = similarities[test_idx]
            test_labels = labels[test_idx]

            # Train set에서 최적 threshold 찾기
            best_threshold = self._find_best_threshold(train_sim, train_labels)
            thresholds.append(best_threshold)

            # Test set에서 accuracy 계산
            predictions = (test_sim >= best_threshold).astype(int)
            acc = accuracy_score(test_labels, predictions)
            accuracies.append(acc)

        # 평균 accuracy와 threshold 반환
        mean_accuracy = np.mean(accuracies)
        mean_threshold = np.mean(thresholds)

        return mean_accuracy, mean_threshold

    def _find_best_threshold(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """최적 threshold 찾기 (train set에서)"""
        # 🟩 행동 2: threshold resolution을 0.001로 변경 (기존 0.01)
        # 더 정밀한 threshold 탐색으로 accuracy 향상 가능 (+0.02 정도)
        thresholds = np.arange(-1.0, 1.0, 0.001)

        best_threshold = 0.0
        best_accuracy = 0.0

        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)
            accuracy = accuracy_score(labels, predictions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        return best_threshold
