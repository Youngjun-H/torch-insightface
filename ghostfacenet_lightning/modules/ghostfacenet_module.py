"""
Lightning Module for GhostFaceNet
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.arcface import ArcFaceLoss, CombinedLoss
from models.ghostfacenet import GhostFaceNet
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
    StepLR,
)


class GhostFaceNetModule(L.LightningModule):
    """Lightning Module for GhostFaceNet training"""

    def __init__(
        self,
        num_classes: int,
        backbone_type: str = "ghostnetv1",
        width_mult: float = 1.0,
        embedding_size: int = 512,
        dropout: float = 0.0,
        input_size: int = 112,
        num_ghost_v1_stacks: int = 2,
        stem_strides: int = 1,
        margin: float = 0.5,
        scale: float = 64.0,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        lr_scheduler: str = "cosine",
        lr_decay_steps: int = 50,
        lr_min: float = 1e-5,
        warmup_epochs: int = 0,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = GhostFaceNet(
            backbone_type=backbone_type,
            width_mult=width_mult,
            embedding_size=embedding_size,
            dropout=dropout,
            input_size=input_size,
            num_ghost_v1_stacks=num_ghost_v1_stacks,
            stem_strides=stem_strides,
        )

        # Loss
        self.loss_fn = CombinedLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            margin=margin,
            scale=scale,
        )

        # Training parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_decay_steps = lr_decay_steps
        self.lr_min = lr_min
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        images, labels = batch

        # Forward
        embeddings = self.model(images)
        loss, logits = self.loss_fn(embeddings, labels)

        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Logging
        # sync_dist=True for distributed training (accumulate across devices)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return loss

    # Validation step removed - using verification callbacks instead
    # def validation_step(self, batch, batch_idx):
    #     """Validation is handled by FaceVerificationCallback"""
    #     pass

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Optimizer
        optimizer = SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler
        if self.lr_scheduler == "cosine":
            # T_max를 max_epochs - warmup_epochs로 설정하여 마지막 epoch에서 lr이 최소가 되도록 함
            # warmup이 있으면 warmup 이후부터 cosine annealing이 시작되므로
            cosine_epochs = self.max_epochs - self.warmup_epochs
            main_scheduler = CosineAnnealingLR(
                optimizer, T_max=cosine_epochs, eta_min=self.lr_min
            )
        elif self.lr_scheduler == "step":
            main_scheduler = StepLR(optimizer, step_size=self.lr_decay_steps, gamma=0.1)
        elif self.lr_scheduler == "exponential":

            def lr_lambda(epoch):
                return max(self.lr_min / self.lr, (0.95**epoch))

            main_scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            main_scheduler = None

        # Warmup 적용
        if self.warmup_epochs > 0 and main_scheduler is not None:
            # Warmup scheduler: linear warmup from very small lr to base_lr
            # start_factor must be > 0 and <= 1 (PyTorch requirement)
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-6,  # Start from very small lr (almost 0)
                end_factor=1.0,  # End at base_lr
                total_iters=self.warmup_epochs,
            )
            # Sequential scheduler: warmup -> main scheduler
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.warmup_epochs],
            )
        else:
            scheduler = main_scheduler

        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer
