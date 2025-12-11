# Face Recognition Training

PyTorch Lightning 기반 얼굴 인식 모델 학습 (ArcFace, EdgeFace, GhostFaceNets)

## 빠른 시작

### ArcFace 학습

```bash
python -m arcface_lightning.train configs/ms1mv3_r50.py --epoch 20
```

### EdgeFace 학습

```bash
python -m edgeface_lightning.train configs/edgeface_xs.py --epoch 20
```

### GhostFaceNets 학습

```bash
python -m ghostfacenets_lightning.train \
    --data_dir /path/to/dataset \
    --backbone ghostnetv1 \
    --width_mult 1.3 \
    --batch_size 256 \
    --max_epochs 30
```

### SLURM 환경에서 학습

```bash
sbatch train.sh
```

`train.sh`에서 노드 수와 GPU 수를 수정:
```bash
#SBATCH --nodes=4                    # 노드 수
#SBATCH --gres=gpu:8                 # 노드당 GPU 수
#SBATCH --ntasks-per-node=8          # 노드당 태스크 수
```

## 주요 옵션

- `config`: Config 파일 경로 (필수)
- `--num_nodes`: 노드 수
- `--devices`: 노드당 GPU 수
- `--epoch`: 학습 epoch 수
- `--saveckp_freq`: 체크포인트 저장 주기 (기본: 1 epoch)

## EdgeFace 모델 변형

- `edgeface_xs_gamma_06`: X-Small with low-rank (rank_ratio=0.6)
- `edgeface_s_gamma_05`: Small with low-rank (rank_ratio=0.5)
- `edgeface_xxs`: XX-Small
- `edgeface_base`: Base

## GhostFaceNets 주요 옵션

- `--backbone`: 백본 타입 (`ghostnetv1`, `ghostnetv2`)
- `--width_mult`: 채널 수 조정 (기본: 1.0)
- `--strides`: 첫 번째 stem 레이어 stride (1 또는 2)
- `--max_epochs`: 학습 epoch 수

## 출력

- 체크포인트: `outputs/{config_name}/checkpoints/`
- 로그: WandB에 자동 업로드
- Verification: 매 epoch 종료 시 자동 평가 (LFW, AgeDB-30, CALFW, CPLFW)

