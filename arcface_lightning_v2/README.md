# ArcFace Lightning V2

PyTorch Lightning 기반 ArcFace 학습 코드

## 구조

```
arcface_lightning_v2/
├── configs/                # Config 파일들
│   ├── base.py            # 기본 설정
│   ├── ms1mv3_r50.py      # ResNet50 설정
│   └── ms1mv3_r100.py     # ResNet100 설정
├── data/
│   ├── datamodule.py      # Lightning DataModule
│   ├── dataset.py          # Dataset 클래스들
│   └── lfw_dataset.py      # LFW Verification Dataset
├── models/
│   ├── backbones/          # Backbone 네트워크들
│   ├── losses.py           # CombinedMarginLoss
│   ├── lr_scheduler.py     # PolynomialLRWarmup
│   ├── module.py           # ArcFaceModule (LightningModule)
│   └── partial_fc_v2.py    # Partial FC V2
├── lightning/
│   ├── callbacks.py        # LFW Verification Callback
│   └── config.py           # Config 로드 유틸리티
└── train.py                # 학습 스크립트
```

## 사용 방법

### 기본 학습

```bash
# ResNet50
python -m arcface_lightning_v2.train configs/ms1mv3_r50.py

# ResNet100
python -m arcface_lightning_v2.train configs/ms1mv3_r100.py
```

### LFW Verification 포함

```bash
python -m arcface_lightning_v2.train \
    configs/ms1mv3_r50.py \
    --pairs_file datasets/pairs.txt
```

### Config 파일

- `configs/base.py`: 기본 설정 (모든 config가 상속)
- `configs/ms1mv3_r50.py`: ResNet50 백본
- `configs/ms1mv3_r100.py`: ResNet100 백본

## 주요 기능

- ✅ PyTorch Lightning 기반 학습
- ✅ Distributed Training 지원 (자동)
- ✅ Mixed Precision 지원 (FP16)
- ✅ Gradient Accumulation
- ✅ LFW Verification Callback
- ✅ TensorBoard & WandB 로깅
- ✅ Checkpoint 저장/복구
- ✅ **insightface 의존성 없음** (독립 실행 가능)

