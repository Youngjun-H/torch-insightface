from easydict import EasyDict as edict

config = edict()

# Margin Base Softmax
# margin_list = (m1, m2, m3)
# ArcFace: (1.0, 0.5, 0.0) - m1=1.0, m2=0.5 (angular margin), m3=0.0
# CosFace: (1.0, 0.0, 0.35) - m1=1.0, m2=0.0, m3=0.35 (cosine margin)
config.margin_list = (1.0, 0.5, 0.0)  # ArcFace (기본값)
config.network = "edgeface_xs_gamma_06"  # EdgeFace 모델 이름
config.resume = False
config.save_all_states = False
config.output = "edgeface_xs_gamma_06"

config.embedding_size = 512

# Partial FC
config.sample_rate = 1
config.interclass_filtering_threshold = 0

config.fp16 = False
config.batch_size = 128

# For AdamW (논문에서 사용, 기본값)
config.optimizer = "adamw"
config.lr = 0.001
config.weight_decay = 0.01  # 0.1은 너무 큼, 일반적으로 0.01 사용
config.momentum = 0.9  # SGD용이지만 train.py에서 필요하므로 기본값 유지

# For SGD (원래 기본값)
# config.optimizer = "sgd"
# config.lr = 0.1
# config.momentum = 0.9
# config.weight_decay = 5e-4

config.verbose = 2000
config.frequent = 10

# For Large Scale Dataset, such as WebFace42M
config.dali = False
config.dali_aug = False

# Gradient ACC
config.gradient_acc = 1

# setup seed
config.seed = 2048

# dataload numworkers
config.num_workers = 2

# Face Verification Datasets
# verification_datasets: List of (filename, dataset_name) tuples
# 예: [("lfw_ann.txt", "lfw"), ("agedb_30_ann.txt", "agedb_30")]
config.verification_val_dir = (
    None  # 검증 데이터셋 루트 디렉토리 (None이면 verification 비활성화)
)
config.verification_datasets = []  # [(filename, dataset_name), ...]

