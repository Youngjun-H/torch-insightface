# ArcFace Training

PyTorch Lightning 기반 ArcFace 얼굴 인식 모델 학습

## 빠른 시작

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

### 로컬 환경에서 학습

```bash
python -m arcface_lightning_v2.train \
    configs/ms1mv3_r50.py \
    --pairs_file /path/to/lfw_ann.txt \
    --epoch 20
```

## 주요 옵션

- `--pairs_file`: LFW 검증용 annotation 파일 경로
- `--num_nodes`: 노드 수 (SLURM 자동 감지)
- `--devices`: 노드당 GPU 수 (SLURM 자동 감지)
- `--epoch`: 학습 epoch 수
- `--saveckp_freq`: 체크포인트 저장 주기 (기본: 1 epoch)

## 출력

- 체크포인트: `outputs/{config_name}/checkpoints/`
- 로그: WandB에 자동 업로드
- LFW 검증: 매 epoch 종료 시 자동 평가

