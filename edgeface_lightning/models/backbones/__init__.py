"""
EdgeFace Backbone Factory (timm 의존성 제거)
"""

from .timmfr import get_edgeface_backbone, replace_linear_with_lowrank_2


def get_model(name, **kwargs):
    """
    EdgeFace 모델 생성 (timm 없이)
    
    Args:
        name: 모델 이름
            - 'edgeface_xs_gamma_06': EdgeNeXt X-Small with low-rank (rank_ratio=0.6)
            - 'edgeface_s_gamma_05': EdgeNeXt Small with low-rank (rank_ratio=0.5)
            - 'edgeface_xxs': EdgeNeXt XX-Small
            - 'edgeface_base': EdgeNeXt Base
            - 'edgeface_xs_q': EdgeNeXt X-Small (quantized, training에서는 사용 안 함)
            - 'edgeface_xxs_q': EdgeNeXt XX-Small (quantized, training에서는 사용 안 함)
        **kwargs: 추가 인자
            - num_features: embedding size (기본: 512)
            - batchnorm: batch normalization 사용 여부 (기본: False)
    
    Returns:
        모델 인스턴스
    """
    embedding_size = kwargs.get('num_features', 512)  # Lightning 호환
    batchnorm = kwargs.get('batchnorm', False)
    
    if name == 'edgeface_xs_gamma_06':
        model = get_edgeface_backbone('edgenext_x_small', featdim=embedding_size, batchnorm=batchnorm)
        return replace_linear_with_lowrank_2(model, rank_ratio=0.6)
    elif name == 'edgeface_xs_q':
        # 양자화는 inference 전용이므로 training에서는 일반 모델 사용
        model = get_edgeface_backbone('edgenext_x_small', featdim=embedding_size, batchnorm=batchnorm)
        return model
    elif name == 'edgeface_xxs':
        return get_edgeface_backbone('edgenext_xx_small', featdim=embedding_size, batchnorm=batchnorm)
    elif name == 'edgeface_base':
        return get_edgeface_backbone('edgenext_base', featdim=embedding_size, batchnorm=batchnorm)
    elif name == 'edgeface_xxs_q':
        # 양자화는 inference 전용이므로 training에서는 일반 모델 사용
        model = get_edgeface_backbone('edgenext_xx_small', featdim=embedding_size, batchnorm=batchnorm)
        return model
    elif name == 'edgeface_s_gamma_05':
        model = get_edgeface_backbone('edgenext_small', featdim=embedding_size, batchnorm=batchnorm)
        return replace_linear_with_lowrank_2(model, rank_ratio=0.5)
    else:
        raise ValueError(f"Unknown EdgeFace model: {name}")