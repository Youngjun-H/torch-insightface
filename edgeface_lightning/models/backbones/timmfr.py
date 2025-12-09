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
    """
    Recursively replace Linear layers with Low-rank Linear layers
    공식 edgeface/backbones/timmfr.py와 정확히 동일한 구현
    """
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


class TimmFRWrapperV2(nn.Module):
    """
    Wraps timm model
    원본 edgeface/backbones/timmfr.py의 TimmFRWrapperV2와 정확히 동일한 구현
    """
    def __init__(self, model_name='edgenext_x_small', featdim=512, batchnorm=False, use_timm=True):
        super().__init__()
        self.featdim = featdim
        self.model_name = model_name
        
        # timm 사용 가능하면 timm 모델 사용 (공식 구현과 정확히 동일)
        # 주의: 공식 구현은 pretrained=False (처음부터 학습)
        if use_timm:
            try:
                import timm
                # 공식 구현과 정확히 동일: timm.create_model (pretrained 없음) 후 reset_classifier
                self.model = timm.create_model(model_name)
                self.model.reset_classifier(featdim)
                self._using_timm = True
            except (ImportError, Exception) as e:
                # timm을 사용할 수 없으면 직접 구현한 모델 사용
                if hasattr(self, 'model'):
                    del self.model
                self._using_timm = False
                use_timm = False
        
        # timm을 사용할 수 없으면 직접 구현한 모델 사용
        if not use_timm or not hasattr(self, '_using_timm') or not self._using_timm:
            self._using_timm = False
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


class EdgeFaceBackbone(nn.Module):
    """
    EdgeFace Backbone
    TimmFRWrapperV2의 별칭 (호환성 유지)
    """
    def __init__(self, model_name='edgenext_x_small', featdim=512, batchnorm=False, use_timm=True):
        super().__init__()
        self.wrapper = TimmFRWrapperV2(model_name=model_name, featdim=featdim, batchnorm=batchnorm, use_timm=use_timm)
        self.featdim = featdim
        self.model_name = model_name
        self.model = self.wrapper.model

    def forward(self, x):
        return self.wrapper(x)


def get_timmfrv2(model_name, **kwargs):
    """
    Create an instance of TimmFRWrapperV2 with the specified `model_name` and additional arguments passed as `kwargs`.
    공식 edgeface/backbones/timmfr.py와 정확히 동일한 구현
    """
    return TimmFRWrapperV2(model_name=model_name, **kwargs)


def get_edgeface_backbone(model_name, **kwargs):
    """
    EdgeFace backbone 생성 (호환성 유지용)
    
    Args:
        model_name: 모델 이름 ('edgenext_x_small', 'edgenext_small', etc.)
        **kwargs: 추가 인자 (featdim, batchnorm, use_timm 등)
            - use_timm: timm 사용 여부 (기본: True)
    
    Returns:
        EdgeFaceBackbone 인스턴스
    """
    featdim = kwargs.get('featdim', 512)
    batchnorm = kwargs.get('batchnorm', False)
    use_timm = kwargs.get('use_timm', True)
    return EdgeFaceBackbone(model_name=model_name, featdim=featdim, batchnorm=batchnorm, use_timm=use_timm)