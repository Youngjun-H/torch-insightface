"""
EdgeNeXt Model Implementation (timm 의존성 제거)
timm의 실제 EdgeNeXt 구조를 정확히 구현
"""

import math
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
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNorm(nn.Module):
    """1D Layer Normalization"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
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


class Mlp(nn.Module):
    """MLP with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, norm_layer=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PositionalEncodingFourier(nn.Module):
    """Positional encoding using Fourier features"""
    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1, bias=False)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        device = self.token_projection.weight.device
        y_embed = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).repeat(H, 1)
        y_embed = y_embed / (H + 1e-6) * self.scale
        x_embed = x_embed / (W + 1e-6) * self.scale
        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        pos = self.token_projection(pos.unsqueeze(0)).expand(B, -1, -1, -1)
        return pos


class CrossCovarianceAttn(nn.Module):
    """Cross-Covariance Attention (XCA)"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv output shape: (B, N, dim * 3)
        qkv = self.qkv(x)  # (B, N, C * 3)
        
        # Reshape to (B, N, 3, num_heads, head_dim)
        # head_dim = C // num_heads
        # Note: C must be divisible by num_heads for proper reshaping
        head_dim = C // self.num_heads
        # qkv shape: (B, N, C * 3)
        # We need to split into 3 parts: q, k, v
        # Each part has shape (B, N, C), which we reshape to (B, N, num_heads, head_dim)
        # Total: (B, N, 3 * C) -> (B, N, 3, C) -> (B, N, 3, num_heads, head_dim)
        qkv = qkv.reshape(B, N, 3, C)  # (B, N, 3, C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)  # (B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ConvBlock(nn.Module):
    """ConvBlock with depthwise conv and MLP"""
    def __init__(self, dim, kernel_size=3, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=LayerNorm):
        super().__init__()
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, bias=False)
        self.norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # Depthwise conv
        x_conv = self.conv_dw(x)  # (B, C, H, W)
        # Permute to (B, H, W, C) for LayerNorm
        x_conv = x_conv.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        # LayerNorm on last dimension
        x_norm = self.norm(x_conv)  # (B, H, W, C)
        # Reshape to (B, H*W, C) for MLP
        x_norm = x_norm.reshape(B, H * W, C)  # (B, H, W, C) -> (B, H*W, C)
        # MLP
        x_mlp = self.mlp(x_norm)  # (B, H*W, C)
        # Reshape back to (B, H, W, C)
        x_mlp = x_mlp.reshape(B, H, W, C)  # (B, H*W, C) -> (B, H, W, C)
        # Permute back to (B, C, H, W)
        x_mlp = x_mlp.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        # Residual connection
        x = x + self.drop_path(x_mlp)
        return x


class SplitTransposeBlock(nn.Module):
    """SplitTransposeBlock with XCA"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, split_size=1, drop=0.0, drop_path=0.0, 
                 act_layer=nn.GELU, norm_layer=LayerNorm, num_splits=2):
        super().__init__()
        self.num_splits = num_splits
        self.split_size = split_size
        
        # Split convolutions - 각 split에 대해 conv 적용
        # 로그를 보면 Stage 1: 1개 conv, Stage 2: 2개 conv, Stage 3: 3개 conv
        self.convs = nn.ModuleList([
            nn.Conv2d(dim // num_splits, dim // num_splits, kernel_size=3, padding=1, 
                     groups=dim // num_splits, bias=False)
            for _ in range(num_splits)
        ])
        
        # Positional encoding
        self.pos_embd = PositionalEncodingFourier(hidden_dim=dim // num_splits, dim=dim)
        
        # XCA
        self.norm_xca = norm_layer(dim)
        self.xca = CrossCovarianceAttn(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        
        # MLP
        self.norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Split and apply convs
        x_splits = torch.split(x, C // self.num_splits, dim=1)
        x_splits = [conv(x_split) for conv, x_split in zip(self.convs, x_splits)]
        x = torch.cat(x_splits, dim=1)
        
        # Add positional encoding
        pos = self.pos_embd(B, H, W)
        x = x + pos
        
        # XCA
        x = x.flatten(2).transpose(1, 2)  # B, C, H, W -> B, H*W, C
        x = x + self.drop_path(self.xca(self.norm_xca(x)))
        x = x.transpose(1, 2).reshape(B, C, H, W)  # B, H*W, C -> B, C, H, W
        
        # MLP
        x = x.flatten(2).transpose(1, 2)  # B, C, H, W -> B, H*W, C
        x = x + self.drop_path(self.mlp(self.norm(x)))
        x = x.transpose(1, 2).reshape(B, C, H, W)  # B, H*W, C -> B, C, H, W
        
        return x


class EdgeNeXtStage(nn.Module):
    """EdgeNeXt Stage"""
    def __init__(self, dim, depth, kernel_size=3, mlp_ratio=4.0, num_heads=8, split_size=1, 
                 drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=LayerNorm, 
                 use_split_transpose=False, num_splits=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        drop_path_list = drop_path if isinstance(drop_path, list) else [drop_path] * depth
        
        for i in range(depth):
            if use_split_transpose and i == depth - 1:
                # Last block is SplitTransposeBlock
                self.blocks.append(
                    SplitTransposeBlock(
                        dim, num_heads=num_heads, mlp_ratio=mlp_ratio, split_size=split_size,
                        drop=drop, drop_path=drop_path_list[i],
                        act_layer=act_layer, norm_layer=norm_layer, num_splits=num_splits
                    )
                )
            else:
                # ConvBlock
                self.blocks.append(
                    ConvBlock(
                        dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio,
                        drop=drop, drop_path=drop_path_list[i],
                        act_layer=act_layer, norm_layer=norm_layer
                    )
                )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SelectAdaptivePool2d(nn.Module):
    """Select adaptive pooling"""
    def __init__(self, output_size=1, pool_type='avg', flatten=False):
        super().__init__()
        self.pool_type = pool_type
        self.flatten = flatten
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x


class NormMlpClassifierHead(nn.Module):
    """NormMlpClassifierHead"""
    def __init__(self, in_features, num_classes, pool_type='avg', drop_rate=0.0, norm_layer=LayerNorm2d):
        super().__init__()
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=True)
        self.norm = norm_layer(in_features)
        self.flatten = nn.Flatten(1)
        self.pre_logits = nn.Identity()
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.norm(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        x = self.flatten(x)
        x = self.pre_logits(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


class EdgeNeXt(nn.Module):
    """EdgeNeXt - timm의 실제 구조"""
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        dims: Tuple[int, ...] = (32, 64, 100, 192),
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9),
        mlp_ratio: float = 4.0,
        num_heads: int = 8,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.0,
        head_init_scale: float = 1.0,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, padding=0, bias=False),
            LayerNorm2d(dims[0])
        )

        # Stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            # Determine if this stage uses SplitTransposeBlock
            use_split_transpose = (i > 0)  # Stage 1, 2, 3 use SplitTransposeBlock
            
            # Calculate num_splits based on stage
            if i == 1:
                num_splits = 1
            elif i == 2:
                num_splits = 2
            elif i == 3:
                num_splits = 3
            else:
                num_splits = 1
            
            # Determine num_heads for each stage based on dim
            # dim must be divisible by num_heads
            stage_dim = dims[i]
            if stage_dim % 8 == 0:
                stage_num_heads = 8
            elif stage_dim % 4 == 0:
                stage_num_heads = 4
            elif stage_dim % 5 == 0:
                stage_num_heads = 5
            else:
                stage_num_heads = 1  # Fallback
            
            stage = EdgeNeXtStage(
                dim=dims[i],
                depth=depths[i],
                kernel_size=kernel_sizes[i],
                mlp_ratio=mlp_ratio,
                num_heads=stage_num_heads,  # dim에 맞는 num_heads 사용
                drop=drop_rate,
                drop_path=dp_rates[cur:cur+depths[i]],
                use_split_transpose=use_split_transpose,
                num_splits=num_splits
            )
            self.stages.append(stage)
            cur += depths[i]
            
            # Downsample between stages
            if i < 3:
                downsample = nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2, bias=False)
                )
                self.stages.append(downsample)
        
        # Head
        self.norm_pre = nn.Identity()
        self.head = NormMlpClassifierHead(
            in_features=dims[-1],
            num_classes=num_classes,
            pool_type='avg',
            drop_rate=drop_rate
        )
        
        # Initialize head
        if head_init_scale != 1.0:
            if hasattr(self.head.fc, 'weight'):
                self.head.fc.weight.data.mul_(head_init_scale)
            if hasattr(self.head.fc, 'bias') and self.head.fc.bias is not None:
                self.head.fc.bias.data.mul_(head_init_scale)
    
    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm_pre(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def edgenext_x_small(**kwargs):
    """EdgeNeXt X-Small - timm의 실제 구조"""
    model = EdgeNeXt(
        dims=[32, 64, 100, 192],
        depths=[3, 3, 9, 3],  # 각 stage에 ConvBlock + 마지막에 SplitTransposeBlock
        kernel_sizes=[3, 5, 7, 9],
        **kwargs
    )
    return model


def edgenext_small(**kwargs):
    """EdgeNeXt Small"""
    model = EdgeNeXt(
        dims=[48, 96, 192, 384],
        depths=[3, 3, 9, 3],
        kernel_sizes=[3, 5, 7, 9],
        **kwargs
    )
    return model


def edgenext_xx_small(**kwargs):
    """EdgeNeXt XX-Small"""
    model = EdgeNeXt(
        dims=[24, 48, 96, 192],
        depths=[2, 2, 6, 2],
        kernel_sizes=[3, 5, 7, 9],
        **kwargs
    )
    return model


def edgenext_base(**kwargs):
    """EdgeNeXt Base"""
    model = EdgeNeXt(
        dims=[80, 160, 288, 584],
        depths=[3, 3, 9, 3],
        kernel_sizes=[3, 5, 7, 9],
        **kwargs
    )
    return model
