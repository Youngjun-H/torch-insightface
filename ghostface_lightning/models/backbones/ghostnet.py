"""
GhostNet Backbone for Face Recognition
PyTorch implementation based on TensorFlow/Keras version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=4, min_value=None):
    """
    This function ensures that all layers have a channel number that is divisible by divisor
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels, se_ratio=0.25):
        super().__init__()
        reduction = _make_divisible(channels * se_ratio)
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
    """Ghost Module"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        dw_kernel_size=3,
        stride=1,
        activation=True,
    ):
        super().__init__()
        conv_out_channel = out_channels // 2

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                conv_out_channel,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(conv_out_channel),
        )
        if activation:
            self.primary_conv.add_module("relu", nn.ReLU(inplace=True))

        cheap_channel = out_channels - conv_out_channel
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                conv_out_channel,
                cheap_channel,
                dw_kernel_size,
                1,
                dw_kernel_size // 2,
                groups=conv_out_channel,
                bias=False,
            ),
            nn.BatchNorm2d(cheap_channel),
        )
        if activation:
            self.cheap_operation.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck"""

    def __init__(
        self,
        in_channels,
        dw_kernel_size,
        stride,
        exp_channels,
        out_channels,
        se_ratio=0,
        shortcut=True,
    ):
        super().__init__()
        self.shortcut = shortcut

        # Ghost module 1
        self.ghost1 = GhostModule(in_channels, exp_channels, activation=True)

        # Depthwise conv if stride > 1
        if stride > 1:
            self.dw_conv = nn.Sequential(
                nn.Conv2d(
                    exp_channels,
                    exp_channels,
                    dw_kernel_size,
                    stride,
                    dw_kernel_size // 2,
                    groups=exp_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(exp_channels),
            )
        else:
            self.dw_conv = nn.Identity()

        # SE block
        if se_ratio > 0:
            self.se = SEBlock(exp_channels, se_ratio)
        else:
            self.se = nn.Identity()

        # Ghost module 2
        self.ghost2 = GhostModule(exp_channels, out_channels, activation=False)

        # Shortcut
        if shortcut:
            self.shortcut_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    dw_kernel_size,
                    stride,
                    dw_kernel_size // 2,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut_conv = nn.Identity()

    def forward(self, x):
        residual = x

        # Ghost module 1
        out = self.ghost1(x)

        # Depthwise conv
        out = self.dw_conv(out)

        # SE block
        out = self.se(out)

        # Ghost module 2
        out = self.ghost2(out)

        # Shortcut
        if self.shortcut:
            residual = self.shortcut_conv(x)

        return out + residual


class GhostNetV1(nn.Module):
    """GhostNet V1 Backbone"""

    def __init__(
        self, input_size=112, width=1.3, strides=2, featdim=512, use_prelu=False
    ):
        super().__init__()
        self.input_size = input_size

        # Stem
        out_channel = _make_divisible(16 * width, 4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channel, 3, strides, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True) if not use_prelu else nn.PReLU(num_parameters=1),
        )

        # Ghost bottlenecks
        dwkernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
        exps = [
            16,
            48,
            72,
            72,
            120,
            240,
            200,
            184,
            184,
            480,
            672,
            672,
            960,
            960,
            960,
            512,
        ]
        outs = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]
        use_ses = [0, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0.25, 0, 0.25]
        bottleneck_strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]

        self.blocks = nn.ModuleList()
        pre_out = out_channel

        for dwk, stride, exp, out, se in zip(
            dwkernels, bottleneck_strides, exps, outs, use_ses
        ):
            out = _make_divisible(out * width, 4)
            exp = _make_divisible(exp * width, 4)
            shortcut = False if out == pre_out and stride == 1 else True

            self.blocks.append(
                GhostBottleneck(pre_out, dwk, stride, exp, out, se, shortcut)
            )
            pre_out = out

        # Final conv
        final_channel = _make_divisible(exps[-1] * width, 4)
        self.final_conv = nn.Sequential(
            nn.Conv2d(pre_out, final_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(final_channel),
            nn.ReLU(inplace=True) if not use_prelu else nn.PReLU(num_parameters=1),
        )

        # GDC (Global Depthwise Convolution) for face recognition
        self.gdc = nn.Sequential(
            nn.Conv2d(
                final_channel,
                final_channel,
                input_size,
                1,
                0,
                groups=final_channel,
                bias=False,
            ),
            nn.BatchNorm2d(final_channel),
            nn.Conv2d(final_channel, featdim, 1, 1, 0, bias=False),
            nn.Flatten(),
            nn.BatchNorm1d(featdim),
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        x = self.gdc(x)
        return x


class GhostModuleMultiply(nn.Module):
    """Ghost Module with Attention (for GhostNet V2)"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        dw_kernel_size=3,
        stride=1,
        activation=True,
    ):
        super().__init__()
        self.ghost = GhostModule(
            in_channels, out_channels, kernel_size, dw_kernel_size, stride, activation
        )

        # Attention branch
        self.attention = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
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
        ghost_out = self.ghost(x)
        attention = self.attention(x)
        # Upsample attention to match ghost_out size
        if attention.shape[2:] != ghost_out.shape[2:]:
            attention = F.interpolate(
                attention,
                size=ghost_out.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return ghost_out * attention


class GhostBottleneckV2(nn.Module):
    """Ghost Bottleneck V2 with Attention"""

    def __init__(
        self,
        in_channels,
        dw_kernel_size,
        stride,
        exp_channels,
        out_channels,
        se_ratio=0,
        shortcut=True,
        use_ghost_multiply=False,
    ):
        super().__init__()
        self.shortcut = shortcut

        # Ghost module 1 (with or without attention)
        if use_ghost_multiply:
            self.ghost1 = GhostModuleMultiply(
                in_channels, exp_channels, activation=True
            )
        else:
            self.ghost1 = GhostModule(in_channels, exp_channels, activation=True)

        # Depthwise conv if stride > 1
        if stride > 1:
            self.dw_conv = nn.Sequential(
                nn.Conv2d(
                    exp_channels,
                    exp_channels,
                    dw_kernel_size,
                    stride,
                    dw_kernel_size // 2,
                    groups=exp_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(exp_channels),
            )
        else:
            self.dw_conv = nn.Identity()

        # SE block
        if se_ratio > 0:
            self.se = SEBlock(exp_channels, se_ratio)
        else:
            self.se = nn.Identity()

        # Ghost module 2
        self.ghost2 = GhostModule(exp_channels, out_channels, activation=False)

        # Shortcut
        if shortcut:
            self.shortcut_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    dw_kernel_size,
                    stride,
                    dw_kernel_size // 2,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut_conv = nn.Identity()

    def forward(self, x):
        residual = x

        # Ghost module 1
        out = self.ghost1(x)

        # Depthwise conv
        out = self.dw_conv(out)

        # SE block
        out = self.se(out)

        # Ghost module 2
        out = self.ghost2(out)

        # Shortcut
        if self.shortcut:
            residual = self.shortcut_conv(x)

        return out + residual


class GhostNetV2(nn.Module):
    """GhostNet V2 Backbone"""

    def __init__(
        self,
        input_size=112,
        width=1.3,
        strides=2,
        featdim=512,
        num_ghost_module_v1_stacks=2,
        use_prelu=False,
    ):
        super().__init__()
        self.input_size = input_size

        # Stem
        out_channel = _make_divisible(16 * width, 4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channel, 3, strides, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True) if not use_prelu else nn.PReLU(num_parameters=1),
        )

        # Ghost bottlenecks
        dwkernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
        exps = [
            16,
            48,
            72,
            72,
            120,
            240,
            200,
            184,
            184,
            480,
            672,
            672,
            960,
            960,
            960,
            960,
        ]
        outs = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]
        use_ses = [0, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0.25, 0, 0.25]
        bottleneck_strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]

        self.blocks = nn.ModuleList()
        pre_out = out_channel

        for stack_id, (dwk, stride, exp, out, se) in enumerate(
            zip(dwkernels, bottleneck_strides, exps, outs, use_ses)
        ):
            out = _make_divisible(out * width, 4)
            exp = _make_divisible(exp * width, 4)
            shortcut = False if out == pre_out and stride == 1 else True
            use_ghost_multiply = (
                num_ghost_module_v1_stacks >= 0
                and stack_id >= num_ghost_module_v1_stacks
            )

            self.blocks.append(
                GhostBottleneckV2(
                    pre_out, dwk, stride, exp, out, se, shortcut, use_ghost_multiply
                )
            )
            pre_out = out

        # Final conv
        final_channel = _make_divisible(exps[-1] * width, 4)
        self.final_conv = nn.Sequential(
            nn.Conv2d(pre_out, final_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(final_channel),
            nn.ReLU(inplace=True) if not use_prelu else nn.PReLU(num_parameters=1),
        )

        # GDC (Global Depthwise Convolution) for face recognition
        self.gdc = nn.Sequential(
            nn.Conv2d(
                final_channel,
                final_channel,
                input_size,
                1,
                0,
                groups=final_channel,
                bias=False,
            ),
            nn.BatchNorm2d(final_channel),
            nn.Conv2d(final_channel, featdim, 1, 1, 0, bias=False),
            nn.Flatten(),
            nn.BatchNorm1d(featdim),
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        x = self.gdc(x)
        return x


def ghostnet_v1(featdim=512, width=1.3, strides=2, use_prelu=False):
    """GhostNet V1 factory function"""
    return GhostNetV1(
        input_size=112,
        width=width,
        strides=strides,
        featdim=featdim,
        use_prelu=use_prelu,
    )


def ghostnet_v2(
    featdim=512, width=1.3, strides=2, num_ghost_module_v1_stacks=2, use_prelu=False
):
    """GhostNet V2 factory function"""
    return GhostNetV2(
        input_size=112,
        width=width,
        strides=strides,
        featdim=featdim,
        num_ghost_module_v1_stacks=num_ghost_module_v1_stacks,
        use_prelu=use_prelu,
    )
