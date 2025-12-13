# ONNX Tools

## convert_to_onnx.py

PyTorch Lightning 체크포인트를 ONNX 모델로 변환합니다.

### 사용법

```bash
python convert_to_onnx.py \
    --checkpoint_path <checkpoint.ckpt> \
    --output_path <output.onnx>
```

### Arguments

- `--checkpoint_path` (required): 체크포인트 파일 경로
- `--output_path` (optional): 출력 ONNX 파일 경로 (기본값: `lightning_ghostfacenets/ghostfacenet.onnx`)

### 예시

```bash
python convert_to_onnx.py \
    --checkpoint_path checkpoints/ghostfacenet-epoch=99.ckpt \
    --output_path models/ghostfacenet.onnx
```

---

## evaluate_onnx.py

ONNX 모델을 face verification 벤치마크 데이터셋으로 평가합니다.

### 사용법

```bash
python evaluate_onnx.py \
    --onnx_path <model.onnx> \
    --pairs_file <pairs.txt> [<pairs2.txt> ...] \
    --pairs_dir <pairs_directory> \
    --root_dir <images_directory> \
    --device <cpu|cuda>
```

### Arguments

- `--onnx_path` (required): ONNX 모델 파일 경로
- `--pairs_file` (required): pairs.txt 파일 경로 (여러 개 지정 가능)
- `--pairs_dir` (optional): pairs 파일들의 공통 디렉토리
- `--root_dir` (optional): 이미지 루트 디렉토리
- `--device` (optional): 사용할 디바이스 (기본값: `cpu`)
- `--batch_size` (optional): 배치 크기 (기본값: `32`)
- `--image_size` (optional): 입력 이미지 크기 (기본값: `112`)

### 예시

```bash
# 단일 벤치마크
python evaluate_onnx.py \
    --onnx_path model.onnx \
    --pairs_file /path/to/lfw_ann.txt \
    --root_dir /path/to/images

# 여러 벤치마크 (pairs_dir 사용)
python evaluate_onnx.py \
    --onnx_path model.onnx \
    --pairs_dir /path/to/benchmarks \
    --pairs_file lfw_ann.txt agedb_30_ann.txt \
    --root_dir /path/to/images \
    --device cuda \
    --batch_size 128
```

