"""
GhostNet Backbone Networks (V1 and V2)
PyTorch implementation for Lightning
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v, divisor=4, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels, se_ratio=0.25):
        super().__init__()
        reduction = make_divisible(channels * se_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channels, 1, bias=True),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        return x * self.se(x)


class GhostModule(nn.Module):
    """Ghost Module for efficient feature extraction"""

    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, use_relu=True
    ):
        super().__init__()
        conv_out_channels = out_channels // 2

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                conv_out_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True) if use_relu else nn.Identity(),
        )

        cheap_channels = out_channels - conv_out_channels
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(
                conv_out_channels,
                cheap_channels,
                3,
                1,
                1,
                groups=conv_out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(cheap_channels),
            nn.ReLU(inplace=True) if use_relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_conv(x1)
        return torch.cat([x1, x2], dim=1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck Block"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        se_ratio=0,
        use_relu=True,
    ):
        super().__init__()
        self.stride = stride

        # Expansion
        self.ghost1 = GhostModule(in_channels, hidden_channels, use_relu=use_relu)

        # Depthwise
        if stride > 1:
            self.dw = nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
            )
        else:
            self.dw = nn.Identity()

        # SE
        self.se = SEBlock(hidden_channels, se_ratio) if se_ratio > 0 else nn.Identity()

        # Projection
        self.ghost2 = GhostModule(hidden_channels, out_channels, use_relu=False)

        # Shortcut
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ghost1(x)
        x = self.dw(x)
        x = self.se(x)
        x = self.ghost2(x)
        return x + residual


class GhostNetV1(nn.Module):
    """GhostNet V1 Backbone"""

    def __init__(self, width_mult=1.0, input_size=112, stem_strides=2):
        super().__init__()
        self.input_size = input_size

        # First layer (stem)
        stem_channels = make_divisible(16 * width_mult, 4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stem_strides, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # Building blocks
        self.blocks = nn.ModuleList()

        # Stage 1
        self.blocks.append(GhostBottleneck(stem_channels, 16, 16, stride=1, se_ratio=0))

        # Stage 2
        self.blocks.append(GhostBottleneck(16, 48, 24, stride=2, se_ratio=0))
        self.blocks.append(GhostBottleneck(24, 72, 24, stride=1, se_ratio=0))

        # Stage 3
        self.blocks.append(GhostBottleneck(24, 72, 40, stride=2, se_ratio=0.25))
        self.blocks.append(GhostBottleneck(40, 120, 40, stride=1, se_ratio=0.25))

        # Stage 4
        self.blocks.append(GhostBottleneck(40, 240, 80, stride=2, se_ratio=0))
        self.blocks.append(GhostBottleneck(80, 200, 80, stride=1, se_ratio=0))
        self.blocks.append(GhostBottleneck(80, 184, 80, stride=1, se_ratio=0))
        self.blocks.append(GhostBottleneck(80, 184, 80, stride=1, se_ratio=0))
        self.blocks.append(GhostBottleneck(80, 480, 112, stride=1, se_ratio=0.25))
        self.blocks.append(GhostBottleneck(112, 672, 112, stride=1, se_ratio=0.25))

        # Stage 5
        self.blocks.append(GhostBottleneck(112, 672, 160, stride=2, se_ratio=0))
        self.blocks.append(GhostBottleneck(160, 960, 160, stride=1, se_ratio=0))
        self.blocks.append(GhostBottleneck(160, 960, 160, stride=1, se_ratio=0.25))
        self.blocks.append(GhostBottleneck(160, 960, 160, stride=1, se_ratio=0))
        self.blocks.append(GhostBottleneck(160, 960, 160, stride=1, se_ratio=0.25))

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x


class GhostModuleV2(nn.Module):
    """Ghost Module V2 with attention mechanism"""

    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, use_relu=True
    ):
        super().__init__()
        conv_out_channels = out_channels // 2

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                conv_out_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True) if use_relu else nn.Identity(),
        )

        cheap_channels = out_channels - conv_out_channels
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(
                conv_out_channels,
                cheap_channels,
                3,
                1,
                1,
                groups=conv_out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(cheap_channels),
            nn.ReLU(inplace=True) if use_relu else nn.Identity(),
        )

        # Attention branch
        self.attention = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                (1, 5),
                1,
                (0, 2),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                (5, 1),
                1,
                (2, 0),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_conv(x1)
        feat = torch.cat([x1, x2], dim=1)

        # Attention
        att = self.attention(x)
        att = F.interpolate(
            att, size=feat.shape[2:], mode="bilinear", align_corners=False
        )

        return feat * att


class GhostBottleneckV2(nn.Module):
    """Ghost Bottleneck Block V2"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        se_ratio=0,
        use_ghost_v2=True,
    ):
        super().__init__()
        self.stride = stride

        # Expansion
        if use_ghost_v2:
            self.ghost1 = GhostModuleV2(in_channels, hidden_channels)
        else:
            self.ghost1 = GhostModule(in_channels, hidden_channels)

        # Depthwise
        if stride > 1:
            self.dw = nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
            )
        else:
            self.dw = nn.Identity()

        # SE
        self.se = SEBlock(hidden_channels, se_ratio) if se_ratio > 0 else nn.Identity()

        # Projection
        self.ghost2 = GhostModule(hidden_channels, out_channels, use_relu=False)

        # Shortcut
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ghost1(x)
        x = self.dw(x)
        x = self.se(x)
        x = self.ghost2(x)
        return x + residual


class GhostNetV2(nn.Module):
    """GhostNet V2 Backbone"""

    def __init__(
        self, width_mult=1.0, num_ghost_v1_stacks=2, input_size=112, stem_strides=1
    ):
        super().__init__()
        self.input_size = input_size

        # First layer (stem)
        stem_channels = make_divisible(16 * width_mult, 4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stem_strides, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # Building blocks
        self.blocks = nn.ModuleList()

        # Stage 1
        use_v2 = num_ghost_v1_stacks < 1
        self.blocks.append(
            GhostBottleneckV2(
                stem_channels, 16, 16, stride=1, se_ratio=0, use_ghost_v2=use_v2
            )
        )

        # Stage 2
        use_v2 = num_ghost_v1_stacks < 2
        self.blocks.append(
            GhostBottleneckV2(16, 48, 24, stride=2, se_ratio=0, use_ghost_v2=use_v2)
        )
        self.blocks.append(
            GhostBottleneckV2(24, 72, 24, stride=1, se_ratio=0, use_ghost_v2=use_v2)
        )

        # Stage 3
        self.blocks.append(
            GhostBottleneckV2(24, 72, 40, stride=2, se_ratio=0.25, use_ghost_v2=True)
        )
        self.blocks.append(
            GhostBottleneckV2(40, 120, 40, stride=1, se_ratio=0.25, use_ghost_v2=True)
        )

        # Stage 4
        self.blocks.append(
            GhostBottleneckV2(40, 240, 80, stride=2, se_ratio=0, use_ghost_v2=True)
        )
        self.blocks.append(
            GhostBottleneckV2(80, 200, 80, stride=1, se_ratio=0, use_ghost_v2=True)
        )
        self.blocks.append(
            GhostBottleneckV2(80, 184, 80, stride=1, se_ratio=0, use_ghost_v2=True)
        )
        self.blocks.append(
            GhostBottleneckV2(80, 184, 80, stride=1, se_ratio=0, use_ghost_v2=True)
        )
        self.blocks.append(
            GhostBottleneckV2(80, 480, 112, stride=1, se_ratio=0.25, use_ghost_v2=True)
        )
        self.blocks.append(
            GhostBottleneckV2(112, 672, 112, stride=1, se_ratio=0.25, use_ghost_v2=True)
        )

        # Stage 5
        self.blocks.append(
            GhostBottleneckV2(112, 672, 160, stride=2, se_ratio=0, use_ghost_v2=True)
        )
        self.blocks.append(
            GhostBottleneckV2(160, 960, 160, stride=1, se_ratio=0, use_ghost_v2=True)
        )
        self.blocks.append(
            GhostBottleneckV2(160, 960, 160, stride=1, se_ratio=0.25, use_ghost_v2=True)
        )
        self.blocks.append(
            GhostBottleneckV2(160, 960, 160, stride=1, se_ratio=0, use_ghost_v2=True)
        )
        self.blocks.append(
            GhostBottleneckV2(160, 960, 160, stride=1, se_ratio=0.25, use_ghost_v2=True)
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x
