"""
GhostFaceNets Lightning Module
"""

import os
from typing import Optional

import lightning as L
import torch
import torch.nn as nn

from .backbones import get_model
from .losses import (
    AdaFaceLoss,
    ArcFaceLoss,
    CombinedMarginLoss,
    CosFaceLoss,
    MagFaceLoss,
)
from .norm_dense import NormDense, NormDenseVPL


class GhostFaceModule(L.LightningModule):
    """
    GhostFaceNets Lightning Module
    """

    def __init__(
        self,
        # Backbone
        network: str = "ghostnetv1",
        embedding_size: int = 512,
        width: float = 1.3,
        strides: int = 2,
        use_prelu: bool = False,
        # Loss
        loss_type: str = "arcface",  # "arcface", "cosface", "adaface", "magface"
        margin_list: tuple = (1.0, 0.5, 0.0),  # (m1, m2, m3) for CombinedMarginLoss
        margin_s: float = 64.0,
        label_smoothing: float = 0.0,
        # Loss-specific parameters
        adaface_margin: float = 0.4,
        adaface_margin_alpha: float = 0.333,
        magface_min_norm: float = 10.0,
        magface_max_norm: float = 110.0,
        magface_min_margin: float = 0.45,
        magface_max_margin: float = 0.8,
        # Classification head
        num_classes: int = 93431,
        use_norm_dense: bool = True,
        loss_top_k: int = 1,
        append_norm: bool = False,  # For AdaFace, MagFace
        use_vpl: bool = False,
        vpl_lambda: float = 0.15,
        vpl_start_iters: int = 8000,
        vpl_allowed_delta: int = 200,
        # Optimizer
        optimizer: str = "sgd",
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        # Learning Rate Scheduler
        num_image: int = 5179510,
        num_epoch: int = 20,
        warmup_epoch: int = 0,
        batch_size: int = 128,
        lr_decay_type: str = "cosine",  # "cosine", "polynomial", "exponential"
        lr_decay_steps: int = 0,  # For exponential decay
        lr_min: float = 1e-6,
        # Training
        gradient_acc: int = 1,
        # Resume
        resume: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Backbone
        self.backbone = get_model(
            network,
            num_features=embedding_size,
            width=width,
            strides=strides,
        )

        # Replace ReLU with PReLU if needed
        if use_prelu:
            self._replace_relu_with_prelu()

        # Classification head
        self.use_norm_dense = use_norm_dense
        if use_norm_dense:
            if use_vpl:
                self.classifier = NormDenseVPL(
                    in_features=embedding_size,
                    out_features=num_classes,
                    batch_size=batch_size,
                    vpl_lambda=vpl_lambda,
                    start_iters=vpl_start_iters,
                    allowed_delta=vpl_allowed_delta,
                    loss_top_k=loss_top_k,
                    append_norm=append_norm,
                )
            else:
                self.classifier = NormDense(
                    in_features=embedding_size,
                    out_features=num_classes,
                    loss_top_k=loss_top_k,
                    append_norm=append_norm,
                )
        else:
            self.classifier = nn.Linear(embedding_size, num_classes)

        # Loss function
        self.loss_type = loss_type
        if loss_type == "arcface":
            self.loss_fn = ArcFaceLoss(
                margin1=margin_list[0],
                margin2=margin_list[1],
                margin3=margin_list[2],
                scale=margin_s,
                label_smoothing=label_smoothing,
            )
        elif loss_type == "cosface":
            self.loss_fn = CosFaceLoss(
                margin=margin_list[2] if margin_list[2] > 0 else 0.35,
                scale=margin_s,
                label_smoothing=label_smoothing,
            )
        elif loss_type == "adaface":
            self.loss_fn = AdaFaceLoss(
                margin=adaface_margin,
                margin_alpha=adaface_margin_alpha,
                scale=margin_s,
                label_smoothing=label_smoothing,
            )
        elif loss_type == "magface":
            self.loss_fn = MagFaceLoss(
                min_feature_norm=magface_min_norm,
                max_feature_norm=magface_max_norm,
                min_margin=magface_min_margin,
                max_margin=magface_max_margin,
                scale=margin_s,
                label_smoothing=label_smoothing,
            )
        else:
            # Default to CombinedMarginLoss
            self.loss_fn = CombinedMarginLoss(
                s=margin_s,
                m1=margin_list[0],
                m2=margin_list[1],
                m3=margin_list[2],
                label_smoothing=label_smoothing,
            )

        # Training parameters
        self.optimizer_name = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.gradient_acc = gradient_acc

        # Learning rate scheduler parameters
        self.num_image = num_image
        self.num_epoch = num_epoch
        self.warmup_epoch = warmup_epoch
        self.batch_size = batch_size
        self.lr_decay_type = lr_decay_type
        self.lr_decay_steps = lr_decay_steps
        self.lr_min = lr_min

        # Resume
        self.resume_path = resume

    def _replace_relu_with_prelu(self):
        """Replace ReLU with PReLU in backbone"""

        def replace_relu(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, nn.PReLU(num_parameters=1))
                else:
                    replace_relu(child)

        replace_relu(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone"""
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        img, labels = batch

        # Embedding 추출
        embeddings = self.backbone(img)

        # Classification head
        logits = self.classifier(embeddings)

        # Loss 계산
        if (
            self.loss_type in ["adaface", "magface"]
            and hasattr(self.classifier, "append_norm")
            and self.classifier.append_norm
        ):
            # Loss functions that need feature norm
            loss = self.loss_fn(logits, labels)
        else:
            loss = self.loss_fn(logits, labels)

        # 디버깅: loss 값이 정상 범위인지 확인
        # ArcFace loss에서 num_classes가 매우 클 때 초기 loss는 높을 수 있지만,
        # 학습이 진행되면 감소해야 합니다.
        if self.global_step % 100 == 0:
            # Log additional debugging info
            with torch.no_grad():
                # Logits 통계
                logits_mean = logits.mean().item()
                logits_std = logits.std().item()
                logits_max = logits.max().item()
                logits_min = logits.min().item()

                # Target logits 통계
                target_logits = logits[torch.arange(logits.size(0)), labels]
                target_mean = target_logits.mean().item()

                self.log("debug/logits_mean", logits_mean, on_step=True)
                self.log("debug/logits_std", logits_std, on_step=True)
                self.log("debug/logits_max", logits_max, on_step=True)
                self.log("debug/logits_min", logits_min, on_step=True)
                self.log("debug/target_logits_mean", target_mean, on_step=True)

        # Logging
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Optimizer 설정
        params = [
            {"params": self.backbone.parameters(), "name": "backbone"},
            {"params": self.classifier.parameters(), "name": "classifier"},
        ]

        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Learning rate scheduler 설정
        world_size = self.trainer.world_size if self.trainer else 1
        gradient_accumulation_steps = (
            getattr(self.trainer, "accumulate_grad_batches", 1) if self.trainer else 1
        )
        total_batch_size = self.batch_size * world_size * gradient_accumulation_steps

        warmup_step = self.num_image // total_batch_size * self.warmup_epoch
        total_step = self.num_image // total_batch_size * self.num_epoch

        if self.lr_decay_type == "polynomial":
            from .lr_scheduler import PolynomialLRWarmup

            scheduler = PolynomialLRWarmup(
                optimizer=optimizer,
                warmup_iters=warmup_step,
                total_iters=total_step,
                power=2.0,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.lr_decay_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_step - warmup_step,
                eta_min=self.lr_min,
            )
            # Warmup을 위한 wrapper
            from .lr_scheduler import WarmupLR

            scheduler = WarmupLR(scheduler, warmup_step)
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        else:
            # Exponential decay
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.lr_decay_steps if self.lr_decay_steps > 0 else 0.95,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def on_train_start(self) -> None:
        """Training 시작 시 호출"""
        # Resume 체크포인트 로드
        if self.resume_path and os.path.exists(self.resume_path):
            checkpoint = torch.load(self.resume_path, map_location=self.device)

            if "state_dict" in checkpoint:
                self.load_state_dict(checkpoint["state_dict"], strict=False)
            elif "model_state_dict" in checkpoint:
                self.load_state_dict(checkpoint["model_state_dict"], strict=False)

    def on_before_optimizer_step(self, optimizer):
        """Optimizer step 전에 gradient clipping"""
        # Gradient clipping
        backbone_norm = torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 5.0)
        classifier_norm = torch.nn.utils.clip_grad_norm_(
            self.classifier.parameters(), 5.0
        )

        # 주기적으로 로깅
        if self.global_step % 100 == 0:
            self.log("grad_norm/backbone", backbone_norm, on_step=True)
            self.log("grad_norm/classifier", classifier_norm, on_step=True)
