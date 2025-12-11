"""
Evaluation utilities for face recognition
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold


def calculate_accuracy(threshold, dist, actual_issame):
    """Calculate accuracy at given threshold"""
    predict_issame = dist < threshold
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame))
    )
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def evaluate_verification(embeddings1, embeddings2, actual_issame, nrof_folds=10):
    """
    Evaluate face verification performance

    Args:
        embeddings1: First set of embeddings (N, embedding_size)
        embeddings2: Second set of embeddings (N, embedding_size)
        actual_issame: Boolean array indicating if pairs are same person
        nrof_folds: Number of folds for cross-validation
    """
    # Normalize embeddings
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # Calculate cosine similarity
    dist = np.sum(embeddings1 * embeddings2, axis=1)

    # K-fold cross-validation
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    thresholds = np.arange(0, 1, 0.01)

    accuracies = []
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(dist)):
        # Find best threshold on training set
        acc_train = np.zeros(len(thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set]
            )
        best_threshold_idx = np.argmax(acc_train)
        best_threshold = thresholds[best_threshold_idx]

        # Evaluate on test set
        _, _, acc = calculate_accuracy(
            best_threshold, dist[test_set], actual_issame[test_set]
        )
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    return mean_acc, std_acc, accuracies


def evaluate_on_pairs(model, pairs_loader, device="cuda"):
    """
    Evaluate model on verification pairs

    Args:
        model: Trained model
        pairs_loader: DataLoader with pairs of images
        device: Device to run evaluation on
    """
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in pairs_loader:
            images, labels = batch
            images = images.to(device)

            embeddings = model(images)
            embeddings = embeddings.cpu().numpy()

            embeddings_list.append(embeddings)
            labels_list.append(labels.numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Split into pairs
    embeddings1 = embeddings[::2]
    embeddings2 = embeddings[1::2]
    actual_issame = labels[::2] == labels[1::2]

    # Evaluate
    mean_acc, std_acc, _ = evaluate_verification(
        embeddings1, embeddings2, actual_issame
    )

    return mean_acc, std_acc
