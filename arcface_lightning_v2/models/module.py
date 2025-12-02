"""
ArcFace Lightning Module
"""

import os
from typing import Optional

import lightning as L
import torch

from .backbones import get_model
from .losses import CombinedMarginLoss
from .lr_scheduler import PolynomialLRWarmup
from .partial_fc_v2 import PartialFC_V2


class ArcFaceModule(L.LightningModule):
    """
    ArcFace Lightning Module
    """

    def __init__(
        self,
        # Backbone
        network: str = "r50",
        embedding_size: int = 512,
        # Loss
        margin_list: tuple = (1.0, 0.5, 0.0),  # (m1, m2, m3)
        margin_s: float = 64.0,
        interclass_filtering_threshold: float = 0.0,
        # Partial FC
        num_classes: int = 93431,
        sample_rate: float = 1.0,
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
            dropout=0.0,
            fp16=False,  # Lightning이 mixed precision 처리
            num_features=embedding_size,
        )

        # Margin Loss
        self.margin_loss = CombinedMarginLoss(
            s=margin_s,
            m1=margin_list[0],
            m2=margin_list[1],
            m3=margin_list[2],
            interclass_filtering_threshold=interclass_filtering_threshold,
        )

        # Partial FC (lazy initialization - distributed가 준비된 후 초기화)
        self.partial_fc = PartialFC_V2(
            margin_loss=self.margin_loss,
            embedding_size=embedding_size,
            num_classes=num_classes,
            sample_rate=sample_rate,
            fp16=False,  # Lightning이 mixed precision 처리
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

        # Resume
        self.resume_path = resume

    def setup(self, stage: str) -> None:
        """Setup distributed environment for PartialFC"""
        if stage == "fit":
            # PartialFC의 distributed 초기화
            self.partial_fc.setup_distributed()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone"""
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        img, labels = batch

        # Embedding 추출
        embeddings = self.backbone(img)

        # Partial FC로 loss 계산
        loss = self.partial_fc(embeddings, labels)

        # Logging
        self.log(
            "train/loss",
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
        # Partial FC의 파라미터는 마지막에 위치해야 함
        params = [
            {"params": self.backbone.parameters()},
            {"params": self.partial_fc.parameters()},
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
        # Total batch size 계산 (world_size는 Lightning이 자동 처리)
        world_size = self.trainer.world_size if self.trainer else 1
        total_batch_size = self.batch_size * world_size

        warmup_step = self.num_image // total_batch_size * self.warmup_epoch
        total_step = self.num_image // total_batch_size * self.num_epoch

        scheduler = PolynomialLRWarmup(
            optimizer=optimizer,
            warmup_iters=warmup_step,
            total_iters=total_step,
            power=2.0,  # polynomial power
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # step마다 업데이트
                "frequency": 1,
            },
        }

    def on_train_start(self) -> None:
        """Training 시작 시 호출"""
        # PartialFC distributed 초기화 확인
        if not self.partial_fc._initialized:
            self.partial_fc.setup_distributed()

        # Resume 체크포인트 로드
        if self.resume_path and os.path.exists(self.resume_path):
            print(f"Loading checkpoint from {self.resume_path}")
            checkpoint = torch.load(self.resume_path, map_location=self.device)

            if "state_dict_backbone" in checkpoint:
                self.backbone.load_state_dict(checkpoint["state_dict_backbone"])
            if "state_dict_softmax_fc" in checkpoint:
                self.partial_fc.load_state_dict(checkpoint["state_dict_softmax_fc"])

            print("Checkpoint loaded successfully")

    def on_before_optimizer_step(self, optimizer):
        """Optimizer step 전에 gradient clipping"""
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 5.0)

    @property
    def _ddp_params_and_buffers_to_ignore(self):
        """DDP에서 PartialFC 파라미터 제외 (각 GPU마다 다른 weight shape)"""
        return ["partial_fc.weight"]
