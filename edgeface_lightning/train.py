"""
EdgeFace Training Script with PyTorch Lightning
Gamma 0.6 Model
"""

import argparse
import os
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from edgeface_lightning.data.datamodule import EdgeFaceDataModule
from edgeface_lightning.lightning_utils.callbacks import FaceVerificationCallback
from edgeface_lightning.lightning_utils.config import get_config
from edgeface_lightning.models.module import EdgeFaceModule


def main():
    parser = argparse.ArgumentParser(
        description="EdgeFace Training with PyTorch Lightning"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Config file path (e.g., 'configs/edgeface_xs_gamma_06.py')",
    )
    parser.add_argument("--num_nodes", type=int, default=None, help="Number of nodes")
    parser.add_argument(
        "--devices", type=int, default=None, help="Number of GPUs per node"
    )
    parser.add_argument(
        "--saveckp_freq", type=int, default=1, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--epoch", type=int, default=None, help="Number of training epochs"
    )
    args = parser.parse_args()

    # Config 로드
    cfg = get_config(args.config)

    # Output 디렉토리 생성
    os.makedirs(cfg.output, exist_ok=True)

    # DataModule 생성
    datamodule = EdgeFaceDataModule(
        root_dir=cfg.rec,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    # Resume 체크포인트 경로 설정
    resume_ckpt_path = None
    if cfg.resume:
        if isinstance(cfg.resume, str):
            resume_ckpt_path = cfg.resume
        else:
            ckpt_dir = os.path.join(cfg.output, "checkpoints")
            if os.path.exists(ckpt_dir):
                ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
                if ckpt_files:
                    resume_ckpt_path = os.path.join(ckpt_dir, sorted(ckpt_files)[-1])

    # Model 생성
    model = EdgeFaceModule(
        network=cfg.network,
        embedding_size=cfg.embedding_size,
        margin_list=cfg.margin_list,
        margin_s=64.0,
        interclass_filtering_threshold=cfg.interclass_filtering_threshold,
        num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate,
        optimizer=cfg.optimizer,
        lr=cfg.lr,
        momentum=getattr(cfg, "momentum", 0.9),
        weight_decay=cfg.weight_decay,
        num_image=cfg.num_image,
        num_epoch=args.epoch or cfg.num_epoch,
        warmup_epoch=cfg.warmup_epoch,
        batch_size=cfg.batch_size,
        gradient_acc=cfg.gradient_acc,
        resume=resume_ckpt_path,
    )

    # Callbacks
    callbacks = []

    # Verification Callbacks
    if cfg.verification_val_dir and cfg.verification_datasets:
        for filename, dataset_name in cfg.verification_datasets:
            pairs_file = os.path.join(cfg.verification_val_dir, filename)
            if os.path.exists(pairs_file):
                pairs_dir = os.path.dirname(os.path.abspath(pairs_file))
                callbacks.append(
                    FaceVerificationCallback(
                        pairs_file=pairs_file,
                        root_dir=pairs_dir,
                        image_size=(112, 112),
                        batch_size=32,
                        num_workers=4,
                        verbose=cfg.verbose,
                        n_folds=10,
                        dataset_name=dataset_name,
                    )
                )

    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Checkpoint Callback
    checkpoint_dir = os.path.join(cfg.output, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="edgeface-{epoch:02d}-{train_loss:.2f}",
            save_top_k=3,
            monitor="train_loss",
            mode="min",
            every_n_epochs=args.saveckp_freq,
            save_on_train_epoch_end=True,
        )
    )

    # Logger
    loggers = [
        WandbLogger(
            project="EdgeFace-Lightning",
            name=f"{datetime.now().strftime('%y%m%d_%H%M')}",
        )
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epoch or cfg.num_epoch,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="ddp",
        precision="bf16-mixed",
        accumulate_grad_batches=cfg.gradient_acc,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=cfg.output,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=cfg.frequent,
    )

    # Training 시작
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=(
            resume_ckpt_path
            if resume_ckpt_path and os.path.exists(resume_ckpt_path)
            else None
        ),
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    wandb.login(key="53f960c86b81377b89feb5d30c90ddc6c3810d3a")
    main()
