from easydict import EasyDict as edict

config = edict()

# Margin Base Softmax
config.margin_list = (1.0, 0.5, 0.0)
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

# For SGD
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

# For AdamW
# config.optimizer = "adamw"
# config.lr = 0.001
# config.weight_decay = 0.1

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

