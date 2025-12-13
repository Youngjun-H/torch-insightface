# GhostFaceNet - PyTorch Lightning Implementation

PyTorch Lightning 기반 GhostFaceNet 얼굴 인식 모델입니다.

## 설치

```bash
uv sync
```

## 모델 학습

### 기본 학습

```bash
python train.py \
    --data_dir /path/to/dataset \
    --backbone ghostnetv1 \
    --batch_size 128 \
    --lr 0.1 \
    --max_epochs 100
```

### 여러 데이터셋 사용

```bash
python train.py \
    --data_dir /path/to/dataset1 /path/to/dataset2 \
    --backbone ghostnetv1 \
    --batch_size 128 \
    --max_epochs 100
```

### GhostNetV2 사용

```bash
python train.py \
    --data_dir /path/to/dataset \
    --backbone ghostnetv2 \
    --width_mult 1.3 \
    --num_ghost_v1_stacks 2 \
    --stem_strides 1 \
    --batch_size 128 \
    --max_epochs 100
```

### 벤치마크 평가 포함

```bash
python train.py \
    --data_dir /path/to/dataset \
    --verification_pairs_dir /path/to/benchmarks \
    --verification_datasets lfw_ann.txt agedb_ann.txt \
    --batch_size 128 \
    --max_epochs 100
```

### 주요 파라미터

**모델:**
- `--backbone`: `ghostnetv1` 또는 `ghostnetv2` (기본: `ghostnetv1`)
- `--width_mult`: 채널 수 배율 (기본: `1.0`)
- `--embedding_size`: 임베딩 차원 (기본: `512`)
- `--stem_strides`: Stem stride (기본: `1`, 선택: `1` 또는 `2`)
- `--num_ghost_v1_stacks`: GhostNetV2에서 v1 모듈 사용 스택 수 (기본: `2`)

**학습:**
- `--batch_size`: 배치 크기 (기본: `128`)
- `--lr`: 학습률 (기본: `0.1`)
- `--max_epochs`: 최대 epoch 수 (기본: `100`)
- `--lr_scheduler`: `cosine`, `step`, `exponential` (기본: `cosine`)
- `--warmup_epochs`: Warmup epoch 수 (기본: `5`)
- `--margin`: ArcFace margin (기본: `0.5`)
- `--scale`: ArcFace scale (기본: `64.0`)

**하드웨어:**
- `--accelerator`: `gpu` 또는 `cpu` (기본: `gpu`)
- `--devices`: GPU 개수 (기본: `1`)
- `--precision`: `bf16-mixed`, `16-mixed`, `32-true` (기본: `bf16-mixed`)

**벤치마크:**
- `--verification_pairs_dir`: 벤치마크 pairs 파일 디렉토리
- `--verification_datasets`: 평가할 데이터셋 파일 목록 (예: `lfw_ann.txt agedb_ann.txt`)

## 데이터셋 형식

```
dataset/
├── class_0/
│   ├── img1.jpg
│   └── ...
├── class_1/
│   ├── img1.jpg
│   └── ...
└── ...
```

## 체크포인트

체크포인트는 `--output_dir` (기본: `./checkpoints`)에 저장됩니다.

## 구조

- `models/`: GhostNet 백본 및 GhostFaceNet 모델
- `losses/`: ArcFace 손실 함수
- `data/`: 데이터셋 및 DataModule
- `modules/`: Lightning Module
- `callbacks/`: 벤치마크 평가 콜백
- `train.py`: 학습 스크립트
