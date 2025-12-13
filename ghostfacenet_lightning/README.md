# GhostFaceNet - PyTorch Lightning Implementation

Lightning 기반의 GhostFaceNet 얼굴 인식 모델 구현입니다.

## 구조

```
lightning_ghostfacenets/
├── models/
│   ├── backbone.py          # GhostNetV1/V2 백본 네트워크
│   └── ghostfacenet.py      # 전체 모델 (백본 + GDC)
├── losses/
│   └── arcface.py           # ArcFace, CosFace 손실 함수
├── data/
│   ├── dataset.py           # 데이터셋 클래스
│   ├── datamodule.py        # Lightning DataModule
│   └── verification_dataset.py  # Face Verification pairs 데이터셋
├── modules/
│   └── ghostfacenet_module.py  # Lightning Module
├── callbacks/
│   └── verification_callback.py  # Face Verification Callback
├── utils/
│   └── evaluation.py        # 평가 유틸리티
├── train.py                 # 학습 스크립트
└── config.py                # 설정 파일
```

## 설치

```bash
pip install -r requirements.txt
# 또는
pip install torch torchvision lightning wandb
```

## 사용법

### 기본 학습

```bash
python train.py \
    --data_dir /path/to/dataset \
    --backbone ghostnetv1 \
    --width_mult 1.0 \
    --batch_size 128 \
    --lr 0.1 \
    --max_epochs 100
```

### 고급 옵션

```bash
python train.py \
    --data_dir /path/to/dataset \
    --backbone ghostnetv2 \
    --width_mult 1.3 \
    --stem_strides 1 \
    --num_ghost_v1_stacks 2 \
    --embedding_size 512 \
    --margin 0.5 \
    --scale 64.0 \
    --batch_size 128 \
    --lr 0.1 \
    --lr_scheduler cosine \
    --lr_decay_steps 50 \
    --weight_decay 5e-4 \
    --max_epochs 100 \
    --accelerator gpu \
    --devices 1 \
    --precision bf16-mixed \
    --name ghostfacenet_v2
```

### 모델 아키텍처 파라미터

- `--width_mult`: 채널 수를 조정하는 width multiplier (기본값: 1.0)
  - GhostNetV1: `width` 파라미터와 동일
  - GhostNetV2: `width_mul` 파라미터와 동일
  - 예: `1.0`, `1.3`, `1.5` 등

- `--stem_strides`: 첫 번째 stem 레이어의 stride (기본값: 1, 선택: 1 또는 2)
  - GhostNetV1: `strides` 파라미터와 동일
  - GhostNetV2: `stem_strides` 파라미터와 동일
  - `1`: 입력 크기 유지 (112x112 → 112x112)
  - `2`: 입력 크기 절반 (112x112 → 56x56)

- `--num_ghost_v1_stacks`: GhostNetV2에서 ghost_module v1을 사용할 스택 수 (기본값: 2)
  - 나머지는 `ghost_module_multiply` 사용

### Precision 설정

- `--precision`: 학습 정밀도 설정 (기본값: `bf16-mixed`)
  - `bf16-mixed`: **H100/A100 권장** - bfloat16 mixed precision (최고 성능)
  - `16-mixed`: float16 mixed precision (일부 GPU에서 호환성 문제 가능)
  - `32-true`: float32 full precision (가장 안정적이지만 느림)

**H100/A100 환경에서는 `bf16-mixed`를 사용하는 것을 강력히 권장합니다.**

### 벤치마크 평가 (Face Verification)

학습 중 자동으로 벤치마크 데이터셋(LFW, AgeDB-30, CALFW, CPLFW 등)에 대한 verification accuracy를 평가합니다:

```bash
python train.py \
    --data_dir /path/to/train/dataset \
    --verification_pairs_dir /path/to/verification/benchmarks \
    --verification_datasets lfw_ann.txt agedb_ann.txt calfw_ann.txt cplfw_ann.txt \
    --verification_batch_size 32 \
    --verification_num_workers 4 \
    --batch_size 128 \
    --max_epochs 100
```

**Verification 인자:**
- `--verification_pairs_dir`: 벤치마크 pairs 파일이 있는 디렉토리
- `--verification_datasets`: 평가할 데이터셋 pairs 파일 목록 (예: `lfw_ann.txt agedb_ann.txt`)
- `--verification_batch_size`: Verification 평가 시 배치 크기 (기본값: 32)
- `--verification_num_workers`: Verification DataLoader worker 수 (기본값: 4)

**주의사항:**
- Epoch 끝날 때마다 자동으로 평가됩니다
- Wandb에 자동으로 로깅됩니다 (`val/{dataset_name}_accuracy`, `val/{dataset_name}_threshold`, `val/{dataset_name}_highest_accuracy`)
- 분산 학습 환경에서는 메인 프로세스(rank 0)에서만 실행됩니다

**Pairs 파일 형식:**
- `<label> <path1> <path2>` 형식 (예: `lfw_ann.txt`)
- 또는 `<path1> <path2> <label>` 형식

## 데이터셋 형식

데이터셋은 다음과 같은 폴더 구조를 가져야 합니다:

```
dataset/
├── class_0/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── class_1/
│   ├── img1.jpg
│   └── ...
└── ...
```

## 주요 기능

- **Lightning 표준**: PyTorch Lightning 표준 코드 사용
- **간단한 구조**: 최대한 심플하게 구현
- **유연한 설정**: 명령줄 인자로 모든 설정 가능
- **자동 체크포인트**: 최고 성능 모델 자동 저장
- **Wandb 로깅**: 학습 과정 및 벤치마크 결과 자동 로깅
- **벤치마크 평가**: Epoch마다 자동으로 LFW, AgeDB-30 등 벤치마크 평가

## 모델 아키텍처

- **Backbone**: GhostNetV1 또는 GhostNetV2
- **Feature Extraction**: GDC (Global Depthwise Convolution)
- **Loss**: ArcFace Loss
- **Output**: 512차원 정규화된 임베딩

## 체크포인트

학습 중 체크포인트는 `./logs/{name}/version_{version}/checkpoints/`에 저장됩니다.

## 평가

### 자동 벤치마크 평가

학습 중 `FaceVerificationCallback`이 자동으로 벤치마크를 평가합니다:
- **K-fold Cross Validation**: 10-fold CV로 정확도 계산
- **최적 Threshold 탐색**: Grid search로 최적 threshold 찾기
- **Wandb 로깅**: 모든 결과가 자동으로 Wandb에 기록됨

### 수동 평가

평가 유틸리티는 `utils/evaluation.py`에 구현되어 있습니다.

