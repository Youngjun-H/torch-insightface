"""
NormDense Layer for ArcFace/CosFace Loss
PyTorch implementation based on TensorFlow/Keras version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormDense(nn.Module):
    """
    Normalized Dense Layer for ArcFace/CosFace
    L2 normalizes both weights and inputs before computing dot product
    """

    def __init__(
        self, in_features, out_features, loss_top_k=1, append_norm=False, bias=False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.loss_top_k = loss_top_k
        self.append_norm = append_norm

        # Weight initialization: glorot normal (Xavier normal)
        self.weight = nn.Parameter(torch.empty(out_features * loss_top_k, in_features))
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # L2 normalize weights and inputs
        # Original: norm_w = tf.nn.l2_normalize(self.w, axis=0, epsilon=1e-5)
        # axis=0 means normalize each column (each feature dimension)
        # In PyTorch: weight shape is (out_features, in_features), so dim=0 normalizes each column
        norm_weight = F.normalize(self.weight, p=2, dim=0, eps=1e-5)
        # Original: norm_inputs = tf.nn.l2_normalize(inputs, axis=1, epsilon=1e-5)
        # axis=1 means normalize each row (each sample)
        norm_inputs = F.normalize(x, p=2, dim=1, eps=1e-5)

        # Compute dot product
        # Original: output = K.dot(norm_inputs, norm_w)
        # norm_inputs: (batch_size, in_features), norm_w: (in_features, out_features)
        # In PyTorch: norm_inputs @ norm_weight.t() = (batch_size, in_features) @ (in_features, out_features)
        output = F.linear(norm_inputs, norm_weight, self.bias)

        # Top-K max pooling if loss_top_k > 1
        if self.loss_top_k > 1:
            output = output.view(-1, self.out_features, self.loss_top_k)
            output, _ = torch.max(output, dim=2)

        # Append norm value if needed (for MagFace, AdaFace, etc.)
        if self.append_norm:
            feature_norm = torch.norm(x, p=2, dim=1, keepdim=True)
            # Keep norm value low by * -1, so will not affect accuracy metrics
            output = torch.cat([output, feature_norm * -1], dim=1)

        return output

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"loss_top_k={self.loss_top_k}, append_norm={self.append_norm}"
        )


class NormDenseVPL(NormDense):
    """
    NormDense with Variational Prototype Learning (VPL)
    """

    def __init__(
        self,
        in_features,
        out_features,
        batch_size,
        vpl_lambda=0.15,
        start_iters=8000,
        allowed_delta=200,
        loss_top_k=1,
        append_norm=False,
    ):
        super().__init__(in_features, out_features, loss_top_k, append_norm)
        self.batch_size = batch_size
        self.vpl_lambda = vpl_lambda
        self.start_iters = start_iters
        self.allowed_delta = allowed_delta

        # Queue for storing prototype features
        self.register_buffer("queue_features", torch.zeros(out_features, in_features))
        self.register_buffer("queue_iters", torch.zeros(out_features, dtype=torch.long))
        self.register_buffer("iters", torch.tensor(0, dtype=torch.long))
        self.register_buffer("norm_features", torch.zeros(batch_size, in_features))

    def forward(self, x):
        self.iters += 1

        # Compute queue lambda based on iterations
        if self.iters > self.start_iters:
            queue_lambda = torch.where(
                (self.iters - self.queue_iters) <= self.allowed_delta,
                torch.tensor(self.vpl_lambda, device=x.device),
                torch.tensor(0.0, device=x.device),
            )
        else:
            queue_lambda = torch.zeros(self.out_features, device=x.device)

        # Normalize inputs
        norm_inputs = F.normalize(x, p=2, dim=1, eps=1e-5)
        self.norm_features[: x.size(0)] = norm_inputs

        # Normalize weights
        # Original: norm_w = tf.nn.l2_normalize(self.w, axis=0, epsilon=1e-5)
        # axis=0 means normalize each column (each feature dimension)
        norm_w = F.normalize(self.weight, p=2, dim=0, eps=1e-5)

        # Inject queue features
        queue_lambda_expanded = queue_lambda.unsqueeze(1)  # [out_features, 1]
        injected_weight = (
            norm_w * (1 - queue_lambda_expanded)
            + self.queue_features.t() * queue_lambda_expanded
        )
        injected_norm_weight = F.normalize(injected_weight, p=2, dim=0, eps=1e-5)

        # Compute output
        output = F.linear(norm_inputs, injected_norm_weight.t())

        # Top-K max pooling if loss_top_k > 1
        if self.loss_top_k > 1:
            output = output.view(-1, self.out_features, self.loss_top_k)
            output, _ = torch.max(output, dim=2)

        # Append norm value if needed
        if self.append_norm:
            feature_norm = torch.norm(x, p=2, dim=1, keepdim=True)
            output = torch.cat([output, feature_norm * -1], dim=1)

        return output
