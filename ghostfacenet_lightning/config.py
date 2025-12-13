"""
Configuration file for GhostFaceNet training
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = "datasets/faces_emore_112x112_folders"
    batch_size: int = 128
    num_workers: int = 4
    input_size: int = 112
    random_status: int = 2


@dataclass
class ModelConfig:
    """Model configuration"""
    backbone: str = "ghostnetv1"  # ghostnetv1 or ghostnetv2
    width_mult: float = 1.0
    embedding_size: int = 512
    dropout: float = 0.0
    input_size: int = 112


@dataclass
class LossConfig:
    """Loss configuration"""
    margin: float = 0.5
    scale: float = 64.0


@dataclass
class TrainingConfig:
    """Training configuration"""
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_scheduler: str = "cosine"  # cosine, step, exponential
    lr_decay_steps: int = 50
    lr_min: float = 1e-5
    max_epochs: int = 100


@dataclass
class LightningConfig:
    """Lightning configuration"""
    gpus: int = 1
    precision: int = 32  # 16 or 32
    resume: Optional[str] = None
    name: str = "ghostfacenet"
    version: Optional[str] = None


@dataclass
class Config:
    """Complete configuration"""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    lightning: LightningConfig = LightningConfig()

