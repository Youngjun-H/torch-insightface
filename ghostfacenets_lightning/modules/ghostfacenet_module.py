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
        strides: int = 2,
        stem_strides: int = 1,
        margin: float = 0.5,
        scale: float = 64.0,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        lr_scheduler: str = "cosine",
        lr_decay_steps: int = 50,
        lr_min: float = 1e-5,
        max_epochs: int = 100,
        warmup_epochs: int = 5,
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
            strides=strides,
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
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

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

    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping before optimizer step"""
        # Gradient clipping to prevent gradient explosion
        # ArcFace 학습에서 특히 중요: embedding과 weight matrix의 gradient가 불안정할 수 있음
        self.clip_gradients(
            optimizer,
            gradient_clip_val=1.0,  # 일반적인 ArcFace 학습에서 권장되는 값
            gradient_clip_algorithm="norm",
        )

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

        # Learning rate scheduler with warmup
        schedulers = []

        # Warmup scheduler (linear warmup)
        if self.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,  # 시작 learning rate는 최종 lr의 1%
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            schedulers.append(warmup_scheduler)

        # Main scheduler
        if self.lr_scheduler == "cosine":
            # T_max를 (max_epochs - warmup_epochs)로 설정하여 warmup 이후 cosine annealing 적용
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs - self.warmup_epochs,
                eta_min=self.lr_min,
            )
            schedulers.append(main_scheduler)
        elif self.lr_scheduler == "step":
            main_scheduler = StepLR(optimizer, step_size=self.lr_decay_steps, gamma=0.1)
            schedulers.append(main_scheduler)
        elif self.lr_scheduler == "exponential":

            def lr_lambda(epoch):
                # warmup 이후부터 exponential decay 적용
                if epoch < self.warmup_epochs:
                    return 1.0
                return max(
                    self.lr_min / self.lr, (0.95 ** (epoch - self.warmup_epochs))
                )

            main_scheduler = LambdaLR(optimizer, lr_lambda)
            schedulers.append(main_scheduler)

        # Combine schedulers if warmup is used
        if len(schedulers) > 1:
            scheduler = SequentialLR(
                optimizer, schedulers=schedulers, milestones=[self.warmup_epochs]
            )
        elif len(schedulers) == 1:
            scheduler = schedulers[0]
        else:
            scheduler = None

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
