"""
ArcFace Training Script with PyTorch Lightning
"""

import argparse
import os
import sys
from datetime import datetime

import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from arcface_lightning_v2.data.datamodule import ArcFaceDataModule
from arcface_lightning_v2.lightning_utils.callbacks import LFWVerificationCallback
from arcface_lightning_v2.lightning_utils.config import get_config
from arcface_lightning_v2.models.module import ArcFaceModule


def main():
    parser = argparse.ArgumentParser(
        description="ArcFace Training with PyTorch Lightning"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Config file path (e.g., 'configs/ms1mv3_r50.py')",
    )
    parser.add_argument(
        "--pairs_file",
        type=str,
        default="datasets/pairs.txt",
        help="LFW pairs.txt file path for verification",
    )
    args = parser.parse_args()

    # Config 로드
    cfg = get_config(args.config)

    # Output 디렉토리 생성
    os.makedirs(cfg.output, exist_ok=True)

    # DataModule 생성
    datamodule = ArcFaceDataModule(
        root_dir=cfg.rec,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    # Resume 체크포인트 경로 설정
    resume_ckpt_path = None
    if cfg.resume:
        if isinstance(cfg.resume, str):
            # 경로가 직접 주어진 경우
            resume_ckpt_path = cfg.resume
        else:
            # True인 경우, output 디렉토리에서 찾기
            # Lightning의 체크포인트를 찾거나, 기존 형식의 체크포인트 찾기
            ckpt_dir = os.path.join(cfg.output, "checkpoints")
            if os.path.exists(ckpt_dir):
                # Lightning 체크포인트 찾기
                ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
                if ckpt_files:
                    resume_ckpt_path = os.path.join(
                        ckpt_dir, sorted(ckpt_files)[-1]
                    )  # 가장 최근 것

    # Model 생성
    model = ArcFaceModule(
        network=cfg.network,
        embedding_size=cfg.embedding_size,
        margin_list=cfg.margin_list,
        margin_s=64.0,  # ArcFace standard
        interclass_filtering_threshold=cfg.interclass_filtering_threshold,
        num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate,
        optimizer=cfg.optimizer,
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        num_image=cfg.num_image,
        num_epoch=cfg.num_epoch,
        warmup_epoch=cfg.warmup_epoch,
        batch_size=cfg.batch_size,
        gradient_acc=cfg.gradient_acc,
        resume=resume_ckpt_path,
    )

    # Callbacks
    callbacks = []

    # LFW Verification Callback
    if os.path.exists(args.pairs_file):
        lfw_callback = LFWVerificationCallback(
            pairs_file=args.pairs_file,
            root_dir=None,  # pairs.txt의 경로가 절대 경로
            image_size=(112, 112),
            batch_size=32,
            num_workers=4,
            verbose=cfg.verbose,
            n_folds=10,
        )
        callbacks.append(lfw_callback)
        print(f"[Info] LFW Verification Callback enabled: {args.pairs_file}")
    else:
        print(f"[Warning] LFW pairs file not found: {args.pairs_file}")

    # Loggers
    loggers = []

    # wandb 설정
    wandb_logger = WandbLogger(
        project="ArcFace-Lightning",
        name=f"{datetime.now().strftime("%y%m%d_%H%M")}",
    )
    loggers.append(wandb_logger)

    # Trainer 설정
    trainer = L.Trainer(
        max_epochs=cfg.num_epoch,
        accelerator="gpu" if cfg.fp16 else "auto",
        devices="auto",
        precision="16-mixed" if cfg.fp16 else "32",
        accumulate_grad_batches=cfg.gradient_acc,
        callbacks=callbacks,
        logger=loggers if loggers else True,  # 기본 logger 사용
        default_root_dir=cfg.output,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=cfg.frequent,
    )

    # Resume 체크포인트가 있으면 로드
    ckpt_path = None
    if resume_ckpt_path and os.path.exists(resume_ckpt_path):
        ckpt_path = resume_ckpt_path
        print(f"[Info] Resuming from checkpoint: {ckpt_path}")

    # 학습 시작
    print(f"[Info] Starting training with config: {args.config}")
    print(f"[Info] Output directory: {cfg.output}")
    print(f"[Info] Network: {cfg.network}")
    print(f"[Info] Batch size: {cfg.batch_size}")
    print(f"[Info] Learning rate: {cfg.lr}")
    print(f"[Info] Epochs: {cfg.num_epoch}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    wandb.login(key="53f960c86b81377b89feb5d30c90ddc6c3810d3a")
    main()
