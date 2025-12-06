"""
EdgeFace Backbone (timm 의존성 제거)
기존 edgeface/backbones/timmfr.py를 기반으로 timm 없이 구현
"""

import torch
import torch.nn as nn

from .edgenext import edgenext_base, edgenext_small, edgenext_x_small, edgenext_xx_small


class LoRaLin(nn.Module):
    """Low-rank Linear layer"""
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LoRaLin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        return x


def replace_linear_with_lowrank_recursive_2(model, rank_ratio=0.2):
    """Recursively replace Linear layers with Low-rank Linear layers"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and 'head' not in name:
            in_features = module.in_features
            out_features = module.out_features
            rank = max(2, int(min(in_features, out_features) * rank_ratio))
            bias = False
            if module.bias is not None:
                bias = True
            lowrank_module = LoRaLin(in_features, out_features, rank, bias)
            setattr(model, name, lowrank_module)
        else:
            replace_linear_with_lowrank_recursive_2(module, rank_ratio)


def replace_linear_with_lowrank_2(model, rank_ratio=0.2):
    """Replace Linear layers with Low-rank Linear layers"""
    replace_linear_with_lowrank_recursive_2(model, rank_ratio)
    return model


class EdgeFaceBackbone(nn.Module):
    """
    EdgeFace Backbone (timm 없이)
    EdgeNeXt 모델을 래핑하여 face recognition에 맞게 수정
    """
    def __init__(self, model_name='edgenext_x_small', featdim=512, batchnorm=False):
        super().__init__()
        self.featdim = featdim
        self.model_name = model_name
        
        # EdgeNeXt 모델 생성
        if model_name == 'edgenext_x_small':
            self.model = edgenext_x_small(num_classes=featdim)
        elif model_name == 'edgenext_small':
            self.model = edgenext_small(num_classes=featdim)
        elif model_name == 'edgenext_xx_small':
            self.model = edgenext_xx_small(num_classes=featdim)
        elif model_name == 'edgenext_base':
            self.model = edgenext_base(num_classes=featdim)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def forward(self, x):
        x = self.model(x)
        return x


def get_edgeface_backbone(model_name, **kwargs):
    """
    EdgeFace backbone 생성 (timm 없이)
    
    Args:
        model_name: 모델 이름 ('edgenext_x_small', 'edgenext_small', etc.)
        **kwargs: 추가 인자 (featdim, batchnorm 등)
    
    Returns:
        EdgeFaceBackbone 인스턴스
    """
    featdim = kwargs.get('featdim', 512)
    batchnorm = kwargs.get('batchnorm', False)
    return EdgeFaceBackbone(model_name=model_name, featdim=featdim, batchnorm=batchnorm)