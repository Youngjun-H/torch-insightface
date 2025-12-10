# EdgeFace XS Gamma 0.6 - Lightning Training

EdgeNeXt XS Gamma 0.6 모델을 PyTorch Lightning으로 학습하는 코드입니다.

## 구조

```
new/
├── train.py                   # 학습 스크립트
├── models/
│   ├── module.py             # Lightning Module
│   ├── backbones/            # 백본 모델
│   │   ├── __init__.py       # 백본 팩토리
│   │   └── edgenext_lowrank.py  # EdgeNeXt with Low-rank Linear 구현
│   ├── losses.py             # Loss 함수
│   ├── lr_scheduler.py       # LR Scheduler
│   └── partial_fc_v2.py       # Partial FC
├── data/
│   ├── datamodule.py         # Lightning DataModule
│   ├── dataset.py            # Dataset
│   └── verification_dataset.py # Verification Dataset
├── configs/
│   └── edgeface_xs_gamma_06.py # 학습 설정
└── lightning_utils/
    ├── callbacks.py          # Callbacks (Verification 등)
    └── config.py             # Config 로더
```

## 사용 방법

### 1. 학습 시작

```bash
cd new
python train.py configs/edgeface_xs_gamma_06.py --devices 4 --num_nodes 1
```

### 2. 설정 파일 수정

`configs/edgeface_xs_gamma_06.py` 파일을 수정하여 학습 설정을 변경할 수 있습니다:

- `batch_size`: 배치 크기
- `lr`: 학습률
- `num_epoch`: 에폭 수
- `rec`: 데이터셋 경로
- `num_classes`: 클래스 수
- `num_image`: 전체 이미지 수

### 3. 체크포인트 재개

```bash
python train.py configs/edgeface_xs_gamma_06.py --devices 4 --resume outputs/edgeface_xs_gamma_06_bs512_e50/checkpoints/latest.ckpt
```

## 주요 특징

1. **LoRaLin 사용**: 모든 Linear 레이어를 Low-rank Linear로 대체하여 파라미터 수 감소
2. **Partial FC**: 대규모 클래스 수를 위한 Partial Fully Connected 레이어
3. **Distributed Training**: DDP를 통한 멀티 GPU 학습 지원
4. **Face Verification**: LFW, AgeDB-30, CALFW, CPLFW 등 검증 데이터셋 지원
5. **Mixed Precision**: bf16-mixed precision으로 학습 속도 향상

## 모델 구조

- **Backbone**: EdgeNeXt X-Small with LoRaLin (rank_ratio=0.6)
- **Embedding Size**: 512
- **Loss**: CosFace (margin_list=(1.0, 0.0, 0.35))
- **Optimizer**: AdamW (lr=0.005, weight_decay=0.01)

## 출력

- **Checkpoints**: `outputs/edgeface_xs_gamma_06_bs512_e50/checkpoints/`
- **Logs**: WandB에 자동 로깅
- **Verification Results**: WandB에 자동 로깅 (각 epoch마다)

