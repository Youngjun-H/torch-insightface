# GhostFaceNets Lightning

PyTorch Lightning 기반 GhostFaceNets 학습 프레임워크

## 개요

이 프로젝트는 TensorFlow/Keras 기반의 GhostFaceNets를 PyTorch Lightning으로 마이그레이션한 버전입니다.

## 주요 기능

- **GhostNet V1/V2 백본**: 경량 얼굴 인식 백본 네트워크
- **다양한 Loss 함수**: ArcFace, CosFace, AdaFace, MagFace 지원
- **NormDense 레이어**: ArcFace/CosFace를 위한 정규화된 Dense 레이어
- **GDC 출력 레이어**: Global Depthwise Convolution 기반 embedding 추출
- **Lightning 기반 학습**: PyTorch Lightning을 활용한 분산 학습 지원

## 설치

```bash
# 가상환경 활성화
source .venv/bin/activate

# 필요한 패키지 설치 (이미 설치되어 있을 수 있음)
pip install torch torchvision lightning wandb easydict scikit-learn
```

## 사용법

### 기본 학습

```bash
python ghostface_lightning/train.py configs/ghostface_base.py
```

### 설정 파일 수정

`configs/ghostface_base.py` 파일을 수정하여 모델, 데이터셋, 학습 파라미터를 설정할 수 있습니다.

### 주요 설정 옵션

- `network`: "ghostnetv1" 또는 "ghostnetv2"
- `loss_type`: "arcface", "cosface", "adaface", "magface"
- `margin_list`: (m1, m2, m3) margin 설정
- `width`: GhostNet width multiplier (기본: 1.3)
- `use_prelu`: PReLU 사용 여부
- `random_status`: Augmentation 강도 (0~3 또는 100+)

## 디렉토리 구조

```
ghostface_lightning/
├── models/
│   ├── backbones/
│   │   ├── __init__.py
│   │   └── ghostnet.py          # GhostNet V1/V2 구현
│   ├── losses.py                 # Loss 함수들
│   ├── norm_dense.py             # NormDense 레이어
│   ├── lr_scheduler.py           # Learning rate scheduler
│   └── module.py                 # Lightning Module
├── data/
│   ├── dataset.py                # Dataset 구현
│   ├── datamodule.py             # Lightning DataModule
│   └── verification_dataset.py   # Verification용 Dataset
├── lightning_utils/
│   ├── callbacks.py              # Lightning Callbacks
│   └── config.py                 # Config 로더
├── configs/
│   └── ghostface_base.py         # 기본 설정 파일
└── train.py                      # 메인 학습 스크립트
```

## 원본 코드와의 차이점

1. **프레임워크**: TensorFlow/Keras → PyTorch Lightning
2. **학습 루프**: `train.Train` 클래스 → `LightningModule`
3. **데이터 처리**: TensorFlow Dataset → PyTorch DataLoader
4. **Loss 함수**: Keras Loss → PyTorch `nn.Module`
5. **콜백**: Keras Callbacks → Lightning Callbacks

## 마이그레이션 완료 항목

- ✅ GhostNet V1/V2 백본 구현
- ✅ NormDense 및 NormDenseVPL 레이어
- ✅ GDC 출력 레이어
- ✅ ArcFace, CosFace, AdaFace, MagFace Loss
- ✅ Lightning Module 및 DataModule
- ✅ Verification Callbacks
- ✅ Learning Rate Schedulers
- ✅ 설정 파일 및 학습 스크립트

## 참고

원본 GhostFaceNets 코드는 `GhostFaceNets/` 디렉토리에 있습니다.

