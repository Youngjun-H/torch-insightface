from easydict import EasyDict as edict

config = edict()

# ArcFace loss 사용: margin_list = (m1, m2, m3) where m1=1.0, m2=0.5, m3=0.0
config.margin_list = (1.0, 0.5, 0.0)  # ArcFace
config.network = "ghostnetv1"  # GhostNet V1
config.resume = False
config.output = "outputs/ghostface_v1_bs256_e50"
config.embedding_size = 512
config.width = 1.3
config.strides = 2
config.use_prelu = False

# Loss 설정
config.loss_type = "arcface"  # "arcface", "cosface", "adaface", "magface"
config.margin_s = 64.0
config.label_smoothing = 0.0

# Classification head
config.use_norm_dense = True
config.loss_top_k = 1
config.append_norm = False  # AdaFace, MagFace에서 True

# Optimizer 설정
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256

# Learning rate scheduler
config.lr_decay_type = "polynomial"  # "polynomial", "cosine", "exponential"
config.lr_decay_steps = 0
config.lr_min = 1e-6

config.verbose = 2000
config.num_workers = 8
config.seed = 2048
config.gradient_acc = 1
config.frequent = 10

# 데이터셋 경로
config.rec = [
    "/purestorage/AILAB/AI_2/yjhwang/work/face/datasets/ms1m-arcface",
]

config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 50
config.warmup_epoch = 2
config.random_status = 2  # Augmentation 강도

# Face Verification Datasets
config.verification_val_dir = (
    "/purestorage/AILAB/AI_2/yjhwang/work/face/datasets/FACE_VAL/val"
)
config.verification_datasets = [
    ("lfw_ann.txt", "lfw"),
    ("agedb_30_ann.txt", "agedb_30"),
    ("calfw_ann.txt", "calfw"),
    ("cplfw_ann.txt", "cplfw"),
]

