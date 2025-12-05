"""
ArcFace Training Script with PyTorch Lightning
"""

import argparse
import os
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
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
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=None,
        help="Number of nodes for distributed training (auto-detected from SLURM if not specified)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of GPUs per node (auto-detected from SLURM if not specified). Use 'auto' for automatic detection.",
    )
    parser.add_argument(
        "--saveckp_freq",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Number of training epochs (overrides config file if specified)",
    )
    args = parser.parse_args()

    # Config 로드
    cfg = get_config(args.config)

    # Output 디렉토리 생성
    os.makedirs(cfg.output, exist_ok=True)

    # DataModule 생성
    datamodule = ArcFaceDataModule(
        root_dir=cfg.rec,  # 문자열 또는 리스트 모두 지원
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
        num_epoch=args.epoch,  # train.sh에서 전달된 값 또는 config 파일 값
        warmup_epoch=cfg.warmup_epoch,
        batch_size=cfg.batch_size,
        gradient_acc=cfg.gradient_acc,
        resume=resume_ckpt_path,
    )

    # Callbacks
    callbacks = []

    # LFW Verification Callback
    if os.path.exists(args.pairs_file):
        # annotation 파일의 디렉토리를 root_dir로 설정
        # 예: /path/to/lfw_ann.txt -> /path/to (이미지 루트 디렉토리)
        pairs_dir = os.path.dirname(os.path.abspath(args.pairs_file))
        lfw_callback = LFWVerificationCallback(
            pairs_file=args.pairs_file,
            root_dir=pairs_dir,  # annotation 파일과 같은 디렉토리를 이미지 루트로 사용
            image_size=(112, 112),
            batch_size=32,
            num_workers=4,
            verbose=cfg.verbose,
            n_folds=10,
        )
        callbacks.append(lfw_callback)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # 체크포인트 저장 디렉토리 설정
    checkpoint_dir = os.path.join(cfg.output, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="arcface-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",  # module.py에서 로깅하는 키와 일치
        mode="min",
        every_n_epochs=args.saveckp_freq,
        save_on_train_epoch_end=True,  # Epoch 끝에서 저장
    )
    callbacks.append(checkpoint_callback)

    # Loggers
    loggers = []

    # wandb 설정
    wandb_logger = WandbLogger(
        project="ArcFace-Lightning",
        name=f"{datetime.now().strftime("%y%m%d_%H%M")}",
    )
    loggers.append(wandb_logger)

    # Trainer 설정
    # 주의: PartialFC는 DDP에 포함되지 않음 (각 GPU마다 다른 weight shape)
    # ArcFaceModule의 parameters() 메서드가 PartialFC를 제외하므로
    # DDP는 Backbone 파라미터만 동기화하고, PartialFC는 별도로 관리됨
    trainer = L.Trainer(
        max_epochs=args.epoch,  # train.sh에서 전달된 값 또는 config 파일 값
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="ddp",  # DDP 사용 (PartialFC는 자동으로 제외됨)
        precision="bf16-mixed",  # A100/H100 최적화: bfloat16 사용
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

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    wandb.login(key="53f960c86b81377b89feb5d30c90ddc6c3810d3a")
    main()
