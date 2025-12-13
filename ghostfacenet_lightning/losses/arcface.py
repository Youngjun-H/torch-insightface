"""
ArcFace and related loss functions
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """ArcFace Loss"""

    def __init__(self, num_classes, embedding_size=512, margin=0.5, scale=64.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale

        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch_size, embedding_size)
            labels: (batch_size,)
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight)  # (batch_size, num_classes)
        cosine = cosine.clamp(-1, 1)

        # Calculate theta
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply margin
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Output
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class CosFaceLoss(nn.Module):
    """CosFace Loss"""

    def __init__(self, num_classes, embedding_size=512, margin=0.35, scale=64.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weight)
        cosine = cosine.clamp(-1, 1)

        # Apply margin
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = cosine - one_hot * self.margin
        output *= self.scale

        return output


class CombinedLoss(nn.Module):
    """Combined loss: ArcFace + CrossEntropy"""

    def __init__(self, num_classes, embedding_size=512, margin=0.5, scale=64.0):
        super().__init__()
        self.arcface = ArcFaceLoss(num_classes, embedding_size, margin, scale)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        logits = self.arcface(embeddings, labels)
        loss = self.criterion(logits, labels)
        return loss, logits
