import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import time
import warnings

import numpy as np
import onnxruntime as ort
from data.verification_dataset import VerificationPairsDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# ONNX Runtime 경고 메시지 억제
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


def find_best_threshold(similarities: np.ndarray, labels: np.ndarray) -> float:
    """최적 threshold 찾기 (Grid Search)"""
    # 코사인 유사도 범위: -1.0 ~ 1.0
    thresholds = np.arange(-1.0, 1.0, 0.001)

    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in thresholds:
        predictions = (similarities >= threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold


def k_fold_accuracy(
    similarities: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 10,
) -> tuple:
    """K-fold cross validation으로 최적 threshold 찾고 accuracy 계산"""
    indices = np.arange(len(similarities))
    kfold = KFold(n_splits=n_folds, shuffle=False)

    accuracies = []
    thresholds = []

    for train_idx, test_idx in kfold.split(indices):
        train_sim = similarities[train_idx]
        train_labels = labels[train_idx]

        test_sim = similarities[test_idx]
        test_labels = labels[test_idx]

        best_threshold = find_best_threshold(train_sim, train_labels)
        thresholds.append(best_threshold)

        predictions = (test_sim >= best_threshold).astype(int)
        acc = accuracy_score(test_labels, predictions)
        accuracies.append(acc)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_threshold = np.mean(thresholds)

    return mean_accuracy, std_accuracy, mean_threshold


def load_onnx_session(onnx_path: str, device: str = "cpu"):
    """ONNX 모델 세션 로드"""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")

    if device == "cuda":
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    actual_provider = providers[0]
    print(f"Using providers: {providers}")

    session = ort.InferenceSession(onnx_path, providers=providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")

    return session, input_name, output_name, actual_provider


def evaluate_onnx(
    onnx_path: str,
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    actual_provider: str,
    pairs_file: str,
    root_dir: str = None,
    image_size: tuple = (112, 112),
    batch_size: int = 32,
    num_workers: int = 4,
    n_folds: int = 10,
    device: str = "cpu",
):
    """ONNX 모델로 face verification 평가"""

    print(f"\n{'=' * 80}")
    print(f"Evaluating: {os.path.basename(pairs_file)}")
    print("=" * 80)
    print(f"Loading dataset from: {pairs_file}")
    dataset = VerificationPairsDataset(
        pairs_file=pairs_file,
        root_dir=root_dir,
        image_size=image_size,
    )

    if len(dataset) == 0:
        raise ValueError(f"No pairs loaded from {pairs_file}")

    print(f"Total pairs: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    print("\nRunning inference...")
    embeddings1_list = []
    embeddings2_list = []
    labels_list = []

    inference_start_time = time.time()
    use_cpu_fallback = False
    cpu_session = None

    model_input_shape = session.get_inputs()[0].shape
    expected_batch_size = model_input_shape[0] if isinstance(model_input_shape[0], int) else None
    
    for batch_idx, (img1, img2, label) in enumerate(dataloader):
        img1_np = img1.numpy().astype(np.float32)
        img2_np = img2.numpy().astype(np.float32)

        actual_batch_size = img1_np.shape[0]
        needs_padding = (
            expected_batch_size is not None
            and actual_batch_size != expected_batch_size
            and actual_batch_size < expected_batch_size
        )

        if needs_padding:
            pad_size = expected_batch_size - actual_batch_size
            img1_pad = np.concatenate([img1_np, img1_np[-pad_size:]], axis=0)
            img2_pad = np.concatenate([img2_np, img2_np[-pad_size:]], axis=0)
        else:
            img1_pad = img1_np
            img2_pad = img2_np

        try:
            emb1_full = session.run([output_name], {input_name: img1_pad})[0]
            emb2_full = session.run([output_name], {input_name: img2_pad})[0]

            if needs_padding:
                emb1 = emb1_full[:actual_batch_size]
                emb2 = emb2_full[:actual_batch_size]
            else:
                emb1 = emb1_full
                emb2 = emb2_full

        except Exception as e:
            error_str = str(e)
            is_cudnn_error = "CUDNN" in error_str or "CUDNN_STATUS" in error_str

            if not use_cpu_fallback:
                if is_cudnn_error:
                    print(f"\n⚠ CUDNN error detected with {actual_provider} (likely due to dynamic_shapes model)")
                    print("This model is not compatible with CUDA provider. Switching to CPU...")
                else:
                    print(f"\n⚠ Inference error with {actual_provider}: {e}")
                    print("Falling back to CPU provider for inference...")

                use_cpu_fallback = True
                cpu_session = ort.InferenceSession(
                    onnx_path,
                    providers=["CPUExecutionProvider"]
                )
                print("✓ CPU provider loaded, continuing inference...")

            try:
                if needs_padding:
                    emb1_full = cpu_session.run([output_name], {input_name: img1_pad})[0]
                    emb2_full = cpu_session.run([output_name], {input_name: img2_pad})[0]
                    emb1 = emb1_full[:actual_batch_size]
                    emb2 = emb2_full[:actual_batch_size]
                else:
                    emb1 = cpu_session.run([output_name], {input_name: img1_np})[0]
                    emb2 = cpu_session.run([output_name], {input_name: img2_np})[0]
            except Exception as cpu_e:
                print(f"⚠ Error even with CPU provider: {cpu_e}")
                print("Skipping current batch...")
                continue

        embeddings1_list.append(emb1)
        embeddings2_list.append(emb2)
        labels_list.append(label.numpy())

        if (batch_idx + 1) % 10 == 0:
            current_provider = "CPU (fallback)" if use_cpu_fallback else actual_provider
            print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches (using {current_provider})")

    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    embeddings1 = np.concatenate(embeddings1_list, axis=0)
    embeddings2 = np.concatenate(embeddings2_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    print(f"\nEmbeddings shape: {embeddings1.shape}")
    print(f"Labels shape: {labels.shape}")

    total_images = len(embeddings1) * 2
    total_batches = len(dataloader)
    images_per_second = total_images / inference_time
    batches_per_second = total_batches / inference_time

    print(f"\n{'=' * 60}")
    print("Inference Speed")
    print("=" * 60)
    print(f"Total inference time: {inference_time:.2f} seconds ({inference_time/60:.2f} minutes)")
    print(f"Total images processed: {total_images} (pairs: {len(embeddings1)})")
    print(f"Total batches: {total_batches}")
    print(f"Speed: {images_per_second:.2f} images/second ({batches_per_second:.2f} batches/second)")
    print(f"Average time per image: {inference_time/total_images*1000:.2f} ms")
    print(f"Average time per batch: {inference_time/total_batches*1000:.2f} ms")
    print("=" * 60)

    similarities = np.sum(embeddings1 * embeddings2, axis=1)
    print(f"Similarities range: [{similarities.min():.4f}, {similarities.max():.4f}]")

    print(f"\nComputing {n_folds}-fold cross validation accuracy...")
    mean_accuracy, std_accuracy, mean_threshold = k_fold_accuracy(
        similarities, labels, n_folds
    )

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Dataset: {os.path.basename(pairs_file)}")
    print(f"Total pairs: {len(dataset)}")
    print(f"K-fold CV: {n_folds} folds")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Mean Threshold: {mean_threshold:.4f}")
    print("=" * 60)

    return mean_accuracy, std_accuracy, mean_threshold


def main():
    parser = argparse.ArgumentParser(description="Evaluate ONNX model on face verification datasets")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to ONNX model file")
    parser.add_argument("--pairs_file", type=str, nargs="+", required=True, help="Path(s) to pairs.txt file(s). If --pairs_dir is specified, can be just filenames (e.g., lfw_ann.txt). Multiple files can be specified.")
    parser.add_argument("--pairs_dir", type=str, default=None, help="Common directory for pairs files. If specified, pairs_file can be relative paths or filenames.")
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory for images (if paths in pairs_file are relative)")
    parser.add_argument("--image_size", type=int, default=112, help="Input image size (default: 112)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference (default: 32)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers (default: 4)")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of folds for cross validation (default: 10)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for inference (default: cpu)")

    args = parser.parse_args()

    print(f"Loading ONNX model from: {args.onnx_path}")
    session, input_name, output_name, actual_provider = load_onnx_session(args.onnx_path, args.device)

    results = []
    for pairs_file in args.pairs_file:
        # pairs_dir이 지정되면 경로 조합
        if args.pairs_dir:
            if os.path.isabs(pairs_file):
                # 절대 경로인 경우 그대로 사용
                full_pairs_file = pairs_file
            else:
                # 상대 경로인 경우 pairs_dir과 조합
                full_pairs_file = os.path.join(args.pairs_dir, pairs_file)
        else:
            full_pairs_file = pairs_file
        try:
            mean_accuracy, std_accuracy, mean_threshold = evaluate_onnx(
                onnx_path=args.onnx_path,
                session=session,
                input_name=input_name,
                output_name=output_name,
                actual_provider=actual_provider,
                pairs_file=full_pairs_file,
                root_dir=args.root_dir,
                image_size=(args.image_size, args.image_size),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                n_folds=args.n_folds,
                device=args.device,
            )
            results.append({
                "dataset": os.path.basename(full_pairs_file),
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "mean_threshold": mean_threshold,
            })
        except Exception as e:
            print(f"\n❌ Error evaluating {full_pairs_file}: {e}")
            results.append({
                "dataset": os.path.basename(full_pairs_file),
                "error": str(e),
            })

    # 전체 결과 요약
    print("\n" + "=" * 80)
    print("SUMMARY - All Benchmarks")
    print("=" * 80)
    print(f"{'Dataset':<30} {'Accuracy':<20} {'Threshold':<15}")
    print("-" * 80)
    for result in results:
        if "error" in result:
            print(f"{result['dataset']:<30} {'ERROR':<20} {'-':<15}")
        else:
            print(f"{result['dataset']:<30} {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}  {result['mean_threshold']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
