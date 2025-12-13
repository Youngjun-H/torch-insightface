"""
GhostFaceNet Model
Complete face recognition model with backbone and GDC layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import GhostNetV1, GhostNetV2


class GDC(nn.Module):
    """Global Depthwise Convolution for feature extraction"""

    def __init__(self, in_channels, embedding_size=512, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Depthwise convolution
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=7, groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
        )

        # Pointwise convolution
        self.pw_conv = nn.Conv2d(in_channels, embedding_size, kernel_size=1, bias=False)

        # Final batch norm
        self.bn = nn.BatchNorm2d(embedding_size, affine=False)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.dropout(x)
        x = self.pw_conv(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x


class GhostFaceNet(nn.Module):
    """Complete GhostFaceNet model"""

    def __init__(
        self,
        backbone_type="ghostnetv1",
        width_mult=1.0,
        embedding_size=512,
        dropout=0.0,
        input_size=112,
        num_ghost_v1_stacks=2,
        stem_strides=1,
        use_prelu=False,
    ):
        super().__init__()
        self.embedding_size = embedding_size

        # Backbone
        if backbone_type.lower() == "ghostnetv1":
            self.backbone = GhostNetV1(
                width_mult=width_mult, input_size=input_size, stem_strides=stem_strides
            )
            # Get output channels from backbone
            with torch.no_grad():
                dummy = torch.zeros(1, 3, input_size, input_size)
                backbone_out = self.backbone(dummy)
                backbone_channels = backbone_out.shape[1]
        elif backbone_type.lower() == "ghostnetv2":
            self.backbone = GhostNetV2(
                width_mult=width_mult,
                num_ghost_v1_stacks=num_ghost_v1_stacks,
                input_size=input_size,
                stem_strides=stem_strides,
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, input_size, input_size)
                backbone_out = self.backbone(dummy)
                backbone_channels = backbone_out.shape[1]
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # GDC layer
        self.gdc = GDC(backbone_channels, embedding_size, dropout)

        # Replace ReLU with PReLU if needed
        if use_prelu:
            self._replace_relu_with_prelu()

    def _replace_relu_with_prelu(self):
        """Replace ReLU with PReLU in the model"""

        def replace_relu(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, nn.PReLU())
                else:
                    replace_relu(child)

        replace_relu(self)

    def forward(self, x):
        """Forward pass"""
        x = self.backbone(x)
        x = self.gdc(x)
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        return x

    def extract_features(self, x):
        """Extract features without normalization (for some losses)"""
        x = self.backbone(x)
        x = self.gdc(x)
        return x
