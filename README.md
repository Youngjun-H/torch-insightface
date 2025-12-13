# Face Recognition Training

PyTorch Lightning 기반 얼굴 인식 모델 학습 프레임워크 (ArcFace, EdgeFace, GhostFaceNet)

## 목적

대규모 얼굴 인식 모델을 분산 학습하기 위한 통합 프레임워크입니다. ArcFace, EdgeFace, GhostFaceNet 세 가지 모델을 지원하며, SLURM 환경에서 멀티 노드/멀티 GPU 학습을 지원합니다.

## 모델 학습

### ArcFace

Config 파일 기반 학습:

```bash
python -m arcface_lightning.train configs/ms1mv3_r50.py --epoch 20
```

### EdgeFace

Config 파일 기반 학습:

```bash
python -m edgeface_lightning.train edgeface_lightning/configs/edgeface_xs_gamma_06.py --epoch 50
```

### GhostFaceNet

Command-line arguments 기반 학습:

```bash
python -m ghostfacenet_lightning.train \
    --data_dir /path/to/dataset \
    --backbone ghostnetv1 \
    --width_mult 1.3 \
    --batch_size 256 \
    --max_epochs 100
```

## SLURM 환경에서 학습

### ArcFace 학습

```bash
sbatch train_arc.sh
```

`train_arc.sh`에서 노드 수와 GPU 수 수정:
```bash
#SBATCH --nodes=4                    # 노드 수
#SBATCH --gres=gpu:8                 # 노드당 GPU 수
#SBATCH --ntasks-per-node=8          # 노드당 태스크 수
```

### EdgeFace 학습

```bash
sbatch train_edge.sh
```

`train_edge.sh`에서 노드 수와 GPU 수 수정 (위와 동일)

### GhostFaceNet 학습

```bash
sbatch train_ghost.sh
```

`train_ghost.sh`에서 노드 수와 GPU 수 수정 (위와 동일)

## 출력

- 체크포인트: `outputs/{config_name}/checkpoints/`
- 로그: WandB에 자동 업로드
- Verification: 매 epoch 종료 시 자동 평가 (LFW, AgeDB-30, CALFW, CPLFW)
