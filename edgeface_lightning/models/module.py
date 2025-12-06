"""
EdgeFace Lightning Module
"""

import os
from typing import Optional

import lightning as L
import torch

from .backbones import get_model
from .losses import CombinedMarginLoss
from .lr_scheduler import PolynomialLRWarmup
from .partial_fc_v2 import PartialFC_V2


class EdgeFaceModule(L.LightningModule):
    """
    EdgeFace Lightning Module
    """

    def __init__(
        self,
        # Backbone
        network: str = "edgeface_xs_gamma_06",  # EdgeFace 모델 이름
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

        # Backbone - EdgeFace 모델 사용
        self.backbone = get_model(
            network,
            num_features=embedding_size,  # Lightning 호환
            batchnorm=False,
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
        # PartialFC 파라미터 확인
        backbone_params = list(self.backbone.parameters())
        partial_fc_params = list(self.partial_fc.parameters())

        if len(partial_fc_params) == 0:
            raise RuntimeError(
                "❌ CRITICAL: PartialFC parameters not initialized! "
                "Call setup_distributed() first. "
                f"PartialFC._initialized={self.partial_fc._initialized}, "
                f"PartialFC.weight is None={self.partial_fc.weight is None}"
            )

        # Optimizer 설정
        # Partial FC의 파라미터는 마지막에 위치해야 함
        params = [
            {"params": backbone_params, "name": "backbone"},
            {"params": partial_fc_params, "name": "partial_fc"},
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
        # gradient accumulation도 고려
        gradient_accumulation_steps = (
            getattr(self.trainer, "accumulate_grad_batches", 1) if self.trainer else 1
        )
        total_batch_size = self.batch_size * world_size * gradient_accumulation_steps

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
            checkpoint = torch.load(self.resume_path, map_location=self.device)

            if "state_dict_backbone" in checkpoint:
                self.backbone.load_state_dict(checkpoint["state_dict_backbone"])
            if "state_dict_softmax_fc" in checkpoint:
                self.partial_fc.load_state_dict(checkpoint["state_dict_softmax_fc"])

    def on_before_optimizer_step(self, optimizer):
        """Optimizer step 전에 gradient clipping"""
        # Backbone gradient clipping
        backbone_norm = torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 5.0)

        # PartialFC gradient 확인 및 clipping
        if self.partial_fc._initialized and self.partial_fc.weight is not None:
            # PartialFC weight에 gradient가 있는지 확인
            if self.partial_fc.weight.grad is not None:
                partial_fc_norm = torch.nn.utils.clip_grad_norm_(
                    self.partial_fc.parameters(), 5.0
                )

                # 주기적으로 로깅 (매 100 step마다)
                if self.global_step % 100 == 0:
                    self.log("grad_norm/backbone", backbone_norm, on_step=True)
                    self.log("grad_norm/partial_fc", partial_fc_norm, on_step=True)

                    # PartialFC weight 값 확인
                    weight_mean = self.partial_fc.weight.data.mean().item()
                    weight_std = self.partial_fc.weight.data.std().item()
                    self.log("partial_fc/weight_mean", weight_mean, on_step=True)
                    self.log("partial_fc/weight_std", weight_std, on_step=True)

            else:
                # ⚠️ 경고: PartialFC weight에 gradient가 없음
                if self.global_step % 100 == 0:
                    self.log("grad_norm/partial_fc", 0.0, on_step=True)

    @property
    def _ddp_params_and_buffers_to_ignore(self):
        """DDP에서 PartialFC 파라미터 제외 (각 GPU마다 다른 weight shape)"""
        return ["partial_fc.weight"]