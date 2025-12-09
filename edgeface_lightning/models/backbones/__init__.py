"""
EdgeFace Backbone Factory
공식 edgeface/backbones/__init__.py와 정확히 동일한 구현
"""

import torch

from .timmfr import get_timmfrv2, replace_linear_with_lowrank_2


def get_model(name, **kwargs):
    """
    EdgeFace 모델 생성
    공식 edgeface/backbones/__init__.py와 정확히 동일한 구현
    
    Args:
        name: 모델 이름
            - 'edgeface_xs_gamma_06': EdgeNeXt X-Small with low-rank (rank_ratio=0.6)
            - 'edgeface_s_gamma_05': EdgeNeXt Small with low-rank (rank_ratio=0.5)
            - 'edgeface_xxs': EdgeNeXt XX-Small
            - 'edgeface_base': EdgeNeXt Base
            - 'edgeface_xs_q': EdgeNeXt X-Small (quantized, training에서는 사용 안 함)
            - 'edgeface_xxs_q': EdgeNeXt XX-Small (quantized, training에서는 사용 안 함)
        **kwargs: 추가 인자
            - num_features: embedding size (기본: 512) - Lightning 호환을 위해 num_features로 받음
            - batchnorm: batch normalization 사용 여부 (기본: False)
            - use_timm: timm 사용 여부 (기본: True)
    
    Returns:
        모델 인스턴스
    """
    # Lightning 호환: num_features를 featdim으로 변환
    embedding_size = kwargs.get('num_features', kwargs.get('featdim', 512))
    batchnorm = kwargs.get('batchnorm', False)
    use_timm = kwargs.get('use_timm', True)
    
    # 공식 구현과 정확히 동일한 순서와 방식
    if name == 'edgeface_xs_gamma_06':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_x_small', featdim=embedding_size, batchnorm=batchnorm, use_timm=use_timm), rank_ratio=0.6)
    elif name == 'edgeface_xs_q':
        model = get_timmfrv2('edgenext_x_small', featdim=embedding_size, batchnorm=batchnorm, use_timm=use_timm)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model
    elif name == 'edgeface_xxs':
        return get_timmfrv2('edgenext_xx_small', featdim=embedding_size, batchnorm=batchnorm, use_timm=use_timm)
    elif name == 'edgeface_base':
        return get_timmfrv2('edgenext_base', featdim=embedding_size, batchnorm=batchnorm, use_timm=use_timm)
    elif name == 'edgeface_xxs_q':
        model = get_timmfrv2('edgenext_xx_small', featdim=embedding_size, batchnorm=batchnorm, use_timm=use_timm)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model
    elif name == 'edgeface_s_gamma_05':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_small', featdim=embedding_size, batchnorm=batchnorm, use_timm=use_timm), rank_ratio=0.5)
    else:
        raise ValueError(f"Unknown EdgeFace model: {name}")