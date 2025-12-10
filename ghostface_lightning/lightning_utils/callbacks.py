"""
Lightning Callbacks for GhostFaceNets Training
"""

import os
from typing import Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from ghostface_lightning.data.verification_dataset import VerificationPairsDataset


class FaceVerificationCallback(L.Callback):
    """
    Face Verification Callback for various datasets (LFW, AgeDB-30, CALFW, CPLFW, etc.)
    주기적으로 모델의 verification accuracy를 계산하고 로깅
    """
    
    def __init__(
        self,
        pairs_file: str,
        root_dir: Optional[str] = None,
        image_size: tuple = (112, 112),
        batch_size: int = 32,
        num_workers: int = 4,
        verbose: int = 2000,
        n_folds: int = 10,
        dataset_name: Optional[str] = None,
    ):
        super().__init__()
        self.pairs_file = pairs_file
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.n_folds = n_folds
        
        # 데이터셋 이름 자동 추출
        if dataset_name is None:
            filename = os.path.basename(pairs_file)
            if "_ann.txt" in filename:
                dataset_name = filename.replace("_ann.txt", "")
            elif filename.endswith(".txt"):
                dataset_name = filename.replace(".txt", "")
            else:
                dataset_name = "verification"
        self.dataset_name = dataset_name.lower()
        
        self.highest_acc = 0.0
        self.dataset = None
        self.dataloader = None
    
    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        """데이터셋 초기화"""
        if self.dataset is None:
            self.dataset = VerificationPairsDataset(
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
    
    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Epoch 끝날 때 verification 수행"""
        if trainer.global_rank != 0:
            return
        
        if self.dataset is None:
            self.setup(trainer, pl_module, "fit")
        
        pl_module.eval()
        
        embeddings1_list = []
        embeddings2_list = []
        labels_list = []
        
        with torch.no_grad():
            for img1, img2, label in self.dataloader:
                img1 = img1.to(pl_module.device)
                img2 = img2.to(pl_module.device)
                
                emb1 = pl_module(img1)
                emb2 = pl_module(img2)
                
                emb1 = F.normalize(emb1, p=2, dim=1)
                emb2 = F.normalize(emb2, p=2, dim=1)
                
                embeddings1_list.append(emb1.cpu().numpy())
                embeddings2_list.append(emb2.cpu().numpy())
                labels_list.append(label.numpy())
        
        embeddings1 = np.concatenate(embeddings1_list, axis=0)
        embeddings2 = np.concatenate(embeddings2_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        similarities = np.sum(embeddings1 * embeddings2, axis=1)
        accuracy, threshold = self._k_fold_accuracy(similarities, labels, self.n_folds)
        
        pl_module.train()
        
        if accuracy > self.highest_acc:
            self.highest_acc = accuracy
        
        log_prefix = f"val/{self.dataset_name}"
        pl_module.log(
            f"{log_prefix}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )
        pl_module.log(
            f"{log_prefix}_threshold",
            threshold,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        pl_module.log(
            f"{log_prefix}_highest_accuracy",
            self.highest_acc,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
    
    def _k_fold_accuracy(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        n_folds: int = 10,
    ) -> tuple:
        """K-fold cross validation으로 최적 threshold 찾고 accuracy 계산"""
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
            
            best_threshold = self._find_best_threshold(train_sim, train_labels)
            thresholds.append(best_threshold)
            
            predictions = (test_sim >= best_threshold).astype(int)
            acc = accuracy_score(test_labels, predictions)
            accuracies.append(acc)
        
        mean_accuracy = np.mean(accuracies)
        mean_threshold = np.mean(thresholds)
        
        return mean_accuracy, mean_threshold
    
    def _find_best_threshold(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """최적 threshold 찾기"""
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

