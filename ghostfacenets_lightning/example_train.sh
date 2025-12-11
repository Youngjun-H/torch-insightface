#!/bin/bash
# Example training script

python train.py \
    --data_dir /path/to/your/dataset \
    --backbone ghostnetv1 \
    --width_mult 1.0 \
    --embedding_size 512 \
    --batch_size 128 \
    --lr 0.1 \
    --margin 0.5 \
    --scale 64.0 \
    --lr_scheduler cosine \
    --lr_decay_steps 50 \
    --weight_decay 5e-4 \
    --max_epochs 100 \
    --accelerator gpu \
    --devices 1 \
    --precision bf16-mixed \
    --name ghostfacenet_experiment \
    --verification_pairs_dir /path/to/verification/benchmarks \
    --verification_datasets lfw_ann.txt agedb_ann.txt calfw_ann.txt cplfw_ann.txt \
    --verification_batch_size 32

