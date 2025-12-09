from easydict import EasyDict as edict

config = edict()
# CosFace loss 사용: margin_list = (m1, m2, m3) where m3 > 0
config.margin_list = (1.0, 0.0, 0.35)  # CosFace: m1=1.0, m2=0.0, m3=0.35 / ArcFace: m1=1.0, m2=0.5, m3=0.0
config.network = "edgeface_base"  # EdgeFace Base
config.resume = False
config.output = "outputs/edgeface_base_bs256_e20"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
# AdamW 설정 (base.py에서 상속받지만 명시적으로 설정)
config.optimizer = "adamw"
config.lr = 0.001  # AdamW용 learning rate
config.weight_decay = 0.01  # AdamW용 weight decay (0.1은 너무 큼, 0.01로 수정)
config.batch_size = 512
# SGD 설정 (주석 처리)
# config.optimizer = "sgd"
# config.momentum = 0.9
# config.weight_decay = 5e-4
# config.lr = 0.1
config.verbose = 2000
config.dali = False

# 단일 데이터셋 경로 (문자열) 또는 여러 데이터셋 경로 리스트
config.rec = [
    "/purestorage/AILAB/AI_2/yjhwang/work/face/datasets/ms1m-arcface",
    # "/purestorage/AILAB/AI_2/yjhwang/work/face/datasets/casia_webface",
    # "/purestorage/AILAB/AI_2/yjhwang/work/face/datasets/vgg2face_train",
    # "/purestorage/AILAB/AI_2/yjhwang/work/face/datasets/asian_celeb_112x112_folders",
]

config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 20
config.warmup_epoch = 0

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

