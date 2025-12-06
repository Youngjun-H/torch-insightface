"""
EdgeNeXt Model Implementation (timm 의존성 제거)
EdgeNeXt는 ConvNeXt 기반의 경량 모델
"""

from typing import Tuple

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """2D Layer Normalization"""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.num_channels = num_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u)/torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class EdgeNeXtBlock(nn.Module):
    """EdgeNeXt Block"""
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # B C H W -> B H W C
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = input + self.drop_path(x)
        return x
    
class EdgeNeXt(nn.Module):
    """EdgeNeXt"""
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        dims: Tuple[int, ...] = (48, 96, 192, 384),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
    ):
        super().__init__()

        # Stem 4x4 conv, stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, padding=0, bias=False),
            LayerNorm2d(dims[0])
        )

        # Stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[EdgeNeXtBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
            
            if i < 3:  # Downsample between stages
                self.stages.append(
                    nn.Sequential(
                        LayerNorm2d(dims[i]),
                        nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2, bias=False)
                    )
                )
        
        # Head
        self.norm = LayerNorm2d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
    
    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean([-2, -1])  # Global average pooling
        x = self.head(x)
        return x


def edgenext_x_small(**kwargs):
    """EdgeNeXt X-Small"""
    model = EdgeNeXt(
        depths=[2, 6, 15, 2],
        dims=[52, 104, 208, 416],
        **kwargs
    )
    return model

def edgenext_small(**kwargs):
    """EdgeNeXt Small"""
    model = EdgeNeXt(
        depths=[3, 3, 9, 3],
        dims=[48, 96, 192, 384],
        **kwargs
    )
    return model


def edgenext_xx_small(**kwargs):
    """EdgeNeXt XX-Small"""
    model = EdgeNeXt(
        depths=[2, 2, 6, 2],
        dims=[24, 48, 96, 192],
        **kwargs
    )
    return model


def edgenext_base(**kwargs):
    """EdgeNeXt Base"""
    model = EdgeNeXt(
        depths=[3, 3, 9, 3],
        dims=[80, 160, 288, 584],
        **kwargs
    )
    return model