"""
Training script for GhostFaceNet using PyTorch Lightning
"""

import argparse
import os
import sys
from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import wandb
from callbacks.verification_callback import FaceVerificationCallback
from data.datamodule import FaceDataModule
from modules.ghostfacenet_module import GhostFaceNetModule


def main():
    parser = argparse.ArgumentParser(description="Train GhostFaceNet")

    # Data arguments
    parser.add_argument(
        "--data_dir", type=str, nargs="+", required=True, help="Path(s) to dataset directory(ies). Multiple directories can be specified."
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--input_size", type=int, default=112, help="Input image size")
    parser.add_argument(
        "--random_status", type=int, default=2, help="Data augmentation level"
    )

    # Model arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default="ghostnetv1",
        choices=["ghostnetv1", "ghostnetv2"],
        help="Backbone type",
    )
    parser.add_argument(
        "--width_mult", type=float, default=1.0, help="Width multiplier"
    )
    parser.add_argument(
        "--embedding_size", type=int, default=512, help="Embedding size"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--num_ghost_v1_stacks",
        type=int,
        default=2,
        help="Number of ghost_module v1 stacks (for GhostNetV2)",
    )
    parser.add_argument(
        "--stem_strides",
        type=int,
        default=1,
        choices=[1, 2],
        help="Stem layer strides (1 or 2)",
    )

    # Loss arguments
    parser.add_argument("--margin", type=float, default=0.5, help="ArcFace margin")
    parser.add_argument("--scale", type=float, default=64.0, help="ArcFace scale")

    # Training arguments
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "exponential"],
        help="LR scheduler type",
    )
    parser.add_argument("--lr_decay_steps", type=int, default=50, help="LR decay steps")
    parser.add_argument(
        "--lr_min", type=float, default=1e-5, help="Minimum learning rate"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Number of warmup epochs"
    )
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")

    # Lightning arguments
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Accelerator type (gpu, cpu, etc.)",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes for distributed training",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices per node (GPUs per node)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["16-mixed", "32-true", "bf16-mixed"],
        help="Precision (16-mixed, 32-true, bf16-mixed). Default: bf16-mixed for H100/A100",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--name", type=str, default="ghostfacenet", help="Experiment name"
    )
    parser.add_argument("--version", type=str, default=None, help="Experiment version")

    # Verification benchmark arguments
    parser.add_argument(
        "--verification_pairs_dir",
        type=str,
        default=None,
        help="Directory containing verification pairs files (e.g., lfw_ann.txt, agedb_ann.txt)",
    )
    parser.add_argument(
        "--verification_datasets",
        type=str,
        nargs="+",
        default=None,
        help="List of verification dataset pairs files (e.g., lfw_ann.txt agedb_ann.txt)",
    )
    parser.add_argument(
        "--verification_batch_size",
        type=int,
        default=32,
        help="Batch size for verification evaluation",
    )
    parser.add_argument(
        "--verification_num_workers",
        type=int,
        default=4,
        help="Number of workers for verification dataloader",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )

    args = parser.parse_args()

    # Setup data module
    data_module = FaceDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        random_status=args.random_status,
    )

    # Prepare data to get num_classes (Lightning will call setup automatically)
    data_module.prepare_data()
    num_classes = data_module.num_classes
    print(f"Number of classes: {num_classes}")

    # Setup model
    model = GhostFaceNetModule(
        num_classes=num_classes,
        backbone_type=args.backbone,
        width_mult=args.width_mult,
        embedding_size=args.embedding_size,
        dropout=args.dropout,
        input_size=args.input_size,
        num_ghost_v1_stacks=args.num_ghost_v1_stacks,
        stem_strides=args.stem_strides,
        margin=args.margin,
        scale=args.scale,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_decay_steps=args.lr_decay_steps,
        lr_min=args.lr_min,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
    )

    callbacks = []

    # Verification callbacks (벤치마크 평가)
    first_verification_dataset = None
    if args.verification_pairs_dir and args.verification_datasets:
        verification_datasets = args.verification_datasets
        for dataset_file in verification_datasets:
            pairs_file = os.path.join(args.verification_pairs_dir, dataset_file)
            if os.path.exists(pairs_file):
                # 데이터셋 이름 추출
                dataset_name = dataset_file.replace("_ann.txt", "").replace(".txt", "")

                # 첫 번째 verification dataset 저장 (ModelCheckpoint 모니터링용)
                if first_verification_dataset is None:
                    first_verification_dataset = dataset_name

                callback = FaceVerificationCallback(
                    pairs_file=pairs_file,
                    root_dir=args.verification_pairs_dir,
                    image_size=(args.input_size, args.input_size),
                    batch_size=args.verification_batch_size,
                    num_workers=args.verification_num_workers,
                    n_folds=10,
                    dataset_name=dataset_name,
                )
                callbacks.append(callback)
                print(f"Added verification callback for {dataset_name}")
            else:
                print(f"Warning: Verification pairs file not found: {pairs_file}")

    # 체크포인트 저장 디렉토리 설정
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="ghostfacenet-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",  # module.py에서 로깅하는 키와 일치
        mode="min",
        save_on_train_epoch_end=True,  # Epoch 끝에서 저장
    )
    callbacks.append(checkpoint_callback)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Logger
    loggers = [
        WandbLogger(
            project="GhostFaceNets-Lightning",
            name=f"{datetime.now().strftime('%y%m%d_%H%M')}",
            save_dir="./logs",
        )
    ]

    # Trainer
    # SLURM 환경에서는 Lightning이 자동으로 분산 학습을 감지하지만,
    # 명시적으로 설정하는 것이 더 안정적입니다.
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        num_nodes=args.num_nodes,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        strategy="ddp",
    )

    # Train (ckpt_path는 fit() 메서드에 전달)
    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    wandb.login(key="53f960c86b81377b89feb5d30c90ddc6c3810d3a")
    main()
