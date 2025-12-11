#!/bin/bash
#SBATCH --job-name=CCTV
#SBATCH --partition=hopper
#SBATCH --nodes=4                    # 노드 수 (필요시 수정)
#SBATCH --gres=gpu:8                 # 노드당 GPU 수 (필요시 수정)
#SBATCH --ntasks-per-node=8          # 노드당 태스크 수 (보통 GPU 수와 동일)
#SBATCH --cpus-per-task=14
#SBATCH --mem=2000G
#SBATCH --comment="person_reid_training"
#SBATCH --output=model_%A.log

# =============================================================================
# 학습 설정 정보 출력
# =============================================================================
echo "================================================================"
echo "Job name: $SLURM_JOB_NAME"
echo "Nodelist: $SLURM_JOB_NODELIST"
echo "Number of nodes: ${SLURM_NNODES:-1}"
echo "GPUs per node: ${SLURM_NTASKS_PER_NODE:-8}"
echo "Total GPUs: $((${SLURM_NNODES:-1} * ${SLURM_NTASKS_PER_NODE:-8}))"
echo "================================================================"

echo "Run started at:- "
date
hostname -I;

# SLURM 환경 변수 출력 (디버깅용)
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# NCCL 환경 변수 설정 (네트워크 연결 문제 해결)
# cubox 서버는 InfiniBand가 없고 IPv6 링크-로컬 주소로 연결 시도 시 실패하므로 설정 필요

# 1. InfiniBand 비활성화 (Ethernet만 사용) - 가장 중요!
export NCCL_IB_DISABLE=1

# 2. 네트워크 인터페이스 지정 (IPv4 주소를 가진 인터페이스 사용)
# hostname -I의 첫 번째 IP 주소를 가진 인터페이스 찾기
PRIMARY_IP=$(hostname -I | awk '{print $1}')
if [ -n "$PRIMARY_IP" ]; then
    # ip 명령어로 해당 IP를 가진 인터페이스 찾기
    IFACE=$(ip -4 addr show | grep -B 2 "$PRIMARY_IP" | grep -oP '^\d+:\s\K[^:]+' | head -1)
    if [ -z "$IFACE" ]; then
        # ifconfig 사용 (ip 명령어가 없는 경우)
        IFACE=$(ifconfig | grep -B 1 "$PRIMARY_IP" | grep -oP '^\S+' | head -1 | tr -d ':')
    fi
    if [ -n "$IFACE" ]; then
        echo "Using network interface: $IFACE (IP: $PRIMARY_IP)"
        export NCCL_SOCKET_IFNAME=$IFACE
    else
        echo "Warning: Could not detect network interface, using default"
        # 일반적인 인터페이스 이름 시도
        export NCCL_SOCKET_IFNAME=eth0,eth1,ens,eno,enp
    fi
else 
    echo "Warning: Could not detect primary IP"
    export NCCL_SOCKET_IFNAME=eth0,eth1,ens,eno,enp
fi

# 3. 디버깅 레벨 설정 (문제 발생 시 INFO로 변경하여 상세 로그 확인 가능)
export NCCL_DEBUG=WARN

# 4. 추가 네트워크 정보 출력 (디버깅용)
echo "All IP addresses:"
hostname -I
echo "Network interface details:"
ip -4 addr show 2>/dev/null | head -20 || ifconfig 2>/dev/null | head -20

# =============================================================================
# 학습 실행
# PyTorch Lightning이 SLURM 환경 변수를 자동으로 감지하여 분산 학습 설정
# =============================================================================

# SLURM에서 실제 할당된 노드 수와 GPU 수 확인
# SLURM 환경 변수를 직접 사용 (가장 정확함)
ACTUAL_NODES=${SLURM_NNODES:-1}
ACTUAL_GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE:-8}

echo "Starting training with:"
echo "  - Nodes: $ACTUAL_NODES"
echo "  - GPUs per node: $ACTUAL_GPUS_PER_NODE"
echo "  - Total GPUs: $((ACTUAL_NODES * ACTUAL_GPUS_PER_NODE))"
echo ""

srun python -m lightning_ghostfacenets.train \
        --data_dir /purestorage/AILAB/AI_2/yjhwang/work/face/datasets/ms1m-arcface \
        --backbone ghostnetv1 \
        --width_mult 1.3 \
        --strides 1 \
        --batch_size 256 \
        --lr 0.1 \
        --max_epochs 30 \
        --num_nodes ${ACTUAL_NODES} \
        --devices ${ACTUAL_GPUS_PER_NODE} \
        --precision bf16-mixed \
        --verification_pairs_dir /purestorage/AILAB/AI_2/yjhwang/work/face/datasets/FACE_VAL/val \
        --verification_datasets lfw_ann.txt agedb_30_ann.txt calfw_ann.txt cplfw_ann.txt \
        --verification_batch_size 32