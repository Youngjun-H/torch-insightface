import os

import pandas as pd

# Kaggle LFW root
IMG_ROOT = "/purestorage/AILAB/AI_2/yjhwang/work/face/torch-insightface/datasets/LFW/lfw-deepfunneled/lfw-deepfunneled"
# 예: lfw-deepfunneled/George_W_Bush/George_W_Bush_0001.jpg

df = pd.read_csv("pairs.csv")

out_lines = []

for _, row in df.iterrows():
    row = row.dropna().tolist()

    # same pair (3 columns)
    if len(row) == 3:
        name, n1, n2 = row
        path1 = f"{IMG_ROOT}/{name}/{name}_{int(n1):04d}.jpg"
        path2 = f"{IMG_ROOT}/{name}/{name}_{int(n2):04d}.jpg"
        label = 1

    # diff pair (4 columns)
    elif len(row) == 4:
        name1, n1, name2, n2 = row
        path1 = f"{IMG_ROOT}/{name1}/{name1}_{int(n1):04d}.jpg"
        path2 = f"{IMG_ROOT}/{name2}/{name2}_{int(n2):04d}.jpg"
        label = 0

    else:
        raise ValueError(f"Invalid row length: {row}")

    out_lines.append(f"{path1} {path2} {label}\n")

with open("lfw_pairs_from_kaggle.txt", "w") as f:
    f.writelines(out_lines)

print(f"[Done] Saved {len(out_lines)} pairs to lfw_pairs_from_kaggle.txt")
