"""
Loss Functions for GhostFaceNets
PyTorch implementation based on TensorFlow/Keras version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss
    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    """
    
    def __init__(self, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0, 
                 label_smoothing=0.0, easy_margin=True):
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.margin3 = margin3
        self.scale = scale
        self.label_smoothing = label_smoothing
        self.easy_margin = easy_margin
        
        # Threshold for easy margin
        self.threshold = np.cos((np.pi - margin2) / margin1) if margin1 != 1.0 else -1.0
        self.theta_min = (-1 - margin3) * 2
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [batch_size, num_classes] normalized logits (cosine similarity)
            labels: [batch_size] class labels
        """
        # Convert labels to one-hot if needed
        if labels.dim() == 1:
            labels_one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        else:
            labels_one_hot = labels
        
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            labels_one_hot = labels_one_hot * (1 - self.label_smoothing) + \
                           self.label_smoothing / logits.size(1)
        
        # Get target logits
        target_mask = labels_one_hot.bool()
        target_logits = logits[target_mask]
        
        # Apply margin
        if self.margin1 == 1.0 and self.margin2 == 0.0 and self.margin3 == 0.0:
            theta = target_logits
        elif self.margin1 == 1.0 and self.margin3 == 0.0:
            # Standard ArcFace: cos(theta + m)
            eps = 1e-7
            target_logits_clamped = target_logits.clamp(-1 + eps, 1 - eps)
            target_angle = target_logits_clamped.acos()
            
            if self.easy_margin:
                theta = torch.where(
                    target_logits > self.threshold,
                    (target_angle + self.margin2).cos(),
                    self.theta_min - (target_angle + self.margin2).cos()
                )
            else:
                theta = (target_angle + self.margin2).cos()
        else:
            # General case: cos(m1 * theta + m2) - m3
            eps = 1e-7
            target_logits_clamped = target_logits.clamp(-1 + eps, 1 - eps)
            target_angle = target_logits_clamped.acos()
            theta = (self.margin1 * target_angle + self.margin2).cos() - self.margin3
        
        # Update logits
        arcface_logits = logits.clone()
        arcface_logits[target_mask] = theta
        
        # Scale logits
        arcface_logits = arcface_logits * self.scale
        
        # Compute cross entropy loss
        loss = F.cross_entropy(arcface_logits, labels, label_smoothing=self.label_smoothing)
        
        return loss


class CosFaceLoss(nn.Module):
    """
    CosFace Loss
    Paper: CosFace: Large Margin Cosine Loss for Deep Face Recognition
    """
    
    def __init__(self, margin=0.35, scale=64.0, label_smoothing=0.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [batch_size, num_classes] normalized logits (cosine similarity)
            labels: [batch_size] class labels
        """
        # Convert labels to one-hot if needed
        if labels.dim() == 1:
            labels_one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        else:
            labels_one_hot = labels
        
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            labels_one_hot = labels_one_hot * (1 - self.label_smoothing) + \
                           self.label_smoothing / logits.size(1)
        
        # Apply margin: subtract margin from target logits
        cosface_logits = logits.clone()
        target_mask = labels_one_hot.bool()
        cosface_logits[target_mask] = cosface_logits[target_mask] - self.margin
        
        # Scale logits
        cosface_logits = cosface_logits * self.scale
        
        # Compute cross entropy loss
        loss = F.cross_entropy(cosface_logits, labels, label_smoothing=self.label_smoothing)
        
        return loss


class AdaFaceLoss(nn.Module):
    """
    AdaFace Loss
    Paper: AdaFace: Quality Adaptive Margin for Face Recognition
    """
    
    def __init__(self, margin=0.4, margin_alpha=0.333, mean_std_alpha=0.01, 
                 scale=64.0, label_smoothing=0.0):
        super().__init__()
        self.margin = margin
        self.margin_alpha = margin_alpha
        self.mean_std_alpha = mean_std_alpha
        self.scale = scale
        self.label_smoothing = label_smoothing
        
        self.min_feature_norm = 0.001
        self.max_feature_norm = 100
        self.epsilon = 1e-3
        
        # Running statistics
        self.register_buffer('batch_mean', torch.tensor(20.0))
        self.register_buffer('batch_std', torch.tensor(100.0))
    
    def forward(self, logits_with_norm, labels):
        """
        Args:
            logits_with_norm: [batch_size, num_classes + 1] 
                             (logits + feature_norm appended with -1)
            labels: [batch_size] class labels
        """
        # Split logits and feature norm
        logits = logits_with_norm[:, :-1]
        feature_norm = logits_with_norm[:, -1:] * -1  # Reverse the -1 multiplication
        
        # Clip values
        logits = logits.clamp(-1 + self.epsilon, 1 - self.epsilon)
        feature_norm = feature_norm.clamp(self.min_feature_norm, self.max_feature_norm)
        
        # Update running statistics
        norm_mean = feature_norm.mean()
        norm_std = feature_norm.std()
        self.batch_mean = self.mean_std_alpha * norm_mean + (1 - self.mean_std_alpha) * self.batch_mean
        self.batch_std = self.mean_std_alpha * norm_std + (1 - self.mean_std_alpha) * self.batch_std
        
        # Compute adaptive margin
        margin_scaler = (feature_norm - self.batch_mean) / (self.batch_std + self.epsilon)
        margin_scaler = margin_scaler.clamp(-1, 1) * self.margin_alpha
        scaled_margin = self.margin * margin_scaler
        
        # Convert labels to one-hot if needed
        if labels.dim() == 1:
            labels_one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        else:
            labels_one_hot = labels
        
        # Apply adaptive margin
        target_mask = labels_one_hot.bool()
        eps = self.epsilon
        cos_max_eps = np.pi - eps
        
        adaface_logits = logits.clone()
        target_logits = logits[target_mask]
        target_angles = target_logits.clamp(-1 + eps, 1 - eps).acos()
        target_margins = scaled_margin[target_mask]
        
        # Apply margin: cos(angle - margin) - (margin + scaled_margin)
        target_angles_with_margin = (target_angles - target_margins).clamp(eps, cos_max_eps)
        adaface_logits[target_mask] = target_angles_with_margin.cos() - (self.margin + target_margins)
        
        # Scale logits
        adaface_logits = adaface_logits * self.scale
        
        # Compute cross entropy loss
        loss = F.cross_entropy(adaface_logits, labels, label_smoothing=self.label_smoothing)
        
        return loss


class MagFaceLoss(nn.Module):
    """
    MagFace Loss
    Paper: MagFace: A Universal Representation for Face Recognition and Quality Assessment
    """
    
    def __init__(self, min_feature_norm=10.0, max_feature_norm=110.0,
                 min_margin=0.45, max_margin=0.8, scale=64.0,
                 regularizer_loss_lambda=35.0, use_cosface_margin=False,
                 label_smoothing=0.0):
        super().__init__()
        self.min_feature_norm = min_feature_norm
        self.max_feature_norm = max_feature_norm
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.scale = scale
        self.regularizer_loss_lambda = regularizer_loss_lambda
        self.use_cosface_margin = use_cosface_margin
        self.label_smoothing = label_smoothing
        
        self.margin_scale = (max_margin - min_margin) / (max_feature_norm - min_feature_norm)
        self.regularizer_loss_scale = 1.0 / (max_feature_norm ** 2)
        self.epsilon = 1e-3
    
    def forward(self, logits_with_norm, labels):
        """
        Args:
            logits_with_norm: [batch_size, num_classes + 1] 
                             (logits + feature_norm appended with -1)
            labels: [batch_size] class labels
        """
        # Split logits and feature norm
        logits = logits_with_norm[:, :-1]
        feature_norm = logits_with_norm[:, -1:] * -1  # Reverse the -1 multiplication
        
        # Clip values
        logits = logits.clamp(-1 + self.epsilon, 1 - self.epsilon)
        feature_norm = feature_norm.clamp(self.min_feature_norm, self.max_feature_norm)
        
        # Compute adaptive margin based on feature norm
        margin = self.margin_scale * (feature_norm - self.min_feature_norm) + self.min_margin
        
        # Convert labels to one-hot if needed
        if labels.dim() == 1:
            labels_one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        else:
            labels_one_hot = labels
        
        target_mask = labels_one_hot.bool()
        
        if self.use_cosface_margin:
            # CosFace style margin
            magface_logits = logits.clone()
            magface_logits[target_mask] = magface_logits[target_mask] - margin
        else:
            # ArcFace style margin
            magface_logits = logits.clone()
            target_logits = logits[target_mask]
            target_angles = target_logits.clamp(-1 + self.epsilon, 1 - self.epsilon).acos()
            margin_cos = margin.cos()
            margin_sin = margin.sin()
            
            target_logits_with_margin = target_logits * margin_cos - \
                                      (1 - target_logits.pow(2)).clamp(min=0.0).sqrt() * margin_sin
            magface_logits[target_mask] = torch.minimum(target_logits_with_margin, target_logits)
        
        # Scale logits
        magface_logits = magface_logits * self.scale
        
        # Compute cross entropy loss
        arcface_loss = F.cross_entropy(magface_logits, labels, label_smoothing=self.label_smoothing)
        
        # Regularizer loss: g = 1/(u_a^2) * x_norm + 1/(x_norm)
        regularizer_loss = self.regularizer_loss_scale * feature_norm + 1.0 / feature_norm
        
        # Total loss
        loss = arcface_loss + regularizer_loss.mean() * self.regularizer_loss_lambda
        
        return loss


class CombinedMarginLoss(nn.Module):
    """
    Combined Margin Loss (ArcFace/CosFace unified)
    Supports (m1, m2, m3) margin configuration:
    - ArcFace: (1.0, 0.5, 0.0)
    - CosFace: (1.0, 0.0, 0.35)
    """
    
    def __init__(self, s=64.0, m1=1.0, m2=0.5, m3=0.0, 
                 interclass_filtering_threshold=0.0, label_smoothing=0.0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        self.label_smoothing = label_smoothing
        
        self.easy_margin = True
        if m1 != 1.0:
            self.threshold = np.cos((np.pi - m2) / m1)
        else:
            self.threshold = np.cos(np.pi - m2) if m2 > 0 else -1.0
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [batch_size, num_classes] normalized logits (cosine similarity)
            labels: [batch_size] class labels
        """
        # Convert labels to one-hot if needed
        if labels.dim() == 1:
            labels_one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        else:
            labels_one_hot = labels
        
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            labels_one_hot = labels_one_hot * (1 - self.label_smoothing) + \
                           self.label_smoothing / logits.size(1)
        
        target_mask = labels_one_hot.bool()
        target_logits = logits[target_mask]
        
        # Apply margin
        if self.m1 == 1.0 and self.m2 == 0.0 and self.m3 == 0.0:
            theta = target_logits
        elif self.m1 == 1.0 and self.m3 == 0.0:
            # Standard ArcFace or CosFace
            if self.m2 > 0:
                # ArcFace: cos(theta + m2)
                eps = 1e-7
                target_logits_clamped = target_logits.clamp(-1 + eps, 1 - eps)
                target_angle = target_logits_clamped.acos()
                
                if self.easy_margin:
                    theta = torch.where(
                        target_logits > self.threshold,
                        (target_angle + self.m2).cos(),
                        (-1 - self.m3) * 2 - (target_angle + self.m2).cos()
                    )
                else:
                    theta = (target_angle + self.m2).cos()
            else:
                # CosFace: cos(theta) - m3 (but m3 is margin, not m2)
                theta = target_logits - self.m3
        else:
            # General case: cos(m1 * theta + m2) - m3
            eps = 1e-7
            target_logits_clamped = target_logits.clamp(-1 + eps, 1 - eps)
            target_angle = target_logits_clamped.acos()
            theta = (self.m1 * target_angle + self.m2).cos() - self.m3
        
        # Update logits
        margin_logits = logits.clone()
        margin_logits[target_mask] = theta
        
        # Scale logits
        margin_logits = margin_logits * self.s
        
        # Compute cross entropy loss
        loss = F.cross_entropy(margin_logits, labels, label_smoothing=self.label_smoothing)
        
        return loss

