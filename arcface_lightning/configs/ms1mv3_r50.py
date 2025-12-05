from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = "outputs/1205_r50_bs256_e20_4val"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256
config.lr = 0.1
config.verbose = 2000
config.dali = False

# 단일 데이터셋 경로 (문자열)
config.rec = [
    "/purestorage/AILAB/AI_2/yjhwang/work/face/datasets/ms1m-arcface",
    # "/purestorage/AILAB/AI_2/yjhwang/work/face/datasets/casia_webface",
    # "/purestorage/AILAB/AI_2/yjhwang/work/face/datasets/vgg2face_train",
    # "/purestorage/AILAB/AI_2/yjhwang/work/face/datasets/asian_celeb_112x112_folders",
]

config.num_classes = 85742
config.num_image = 5822623
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
