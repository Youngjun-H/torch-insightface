"""
Backbone Factory for EdgeFace Models
"""

from .edgenext_lowrank import (
    edgenext_base,
    edgenext_s_gamma_05,
    edgenext_xs_gamma_06,
)


def get_model(name, **kwargs):
    """
    EdgeFace 모델 생성

    Args:
        name: 모델 이름
            - 'edgeface_xs_gamma_06': EdgeNeXt X-Small with low-rank (rank_ratio=0.6)
            - 'edgeface_s_gamma_05': EdgeNeXt Small with low-rank (rank_ratio=0.5)
            - 'edgeface_base': EdgeNeXt Base (rank_ratio=1.0, 일반 Linear)
        **kwargs: 추가 인자
            - num_features: embedding size (기본: 512) - Lightning 호환을 위해 num_features로 받음
            - featdim: embedding size (기본: 512) - 호환성 유지
            - batchnorm: batch normalization 사용 여부 (기본: False)
            - rank_ratio: LoRA rank ratio (기본: 모델별로 다름)

    Returns:
        모델 인스턴스
    """
    # Lightning 호환: num_features를 featdim으로 변환
    embedding_size = kwargs.get("num_features", kwargs.get("featdim", 512))

    if name == "edgeface_xs_gamma_06":
        rank_ratio = kwargs.get("rank_ratio", 0.6)
        return edgenext_xs_gamma_06(featdim=embedding_size, rank_ratio=rank_ratio)
    elif name == "edgeface_s_gamma_05":
        rank_ratio = kwargs.get("rank_ratio", 0.5)
        return edgenext_s_gamma_05(featdim=embedding_size, rank_ratio=rank_ratio)
    elif name == "edgeface_base":
        rank_ratio = kwargs.get("rank_ratio", 1.0)
        return edgenext_base(featdim=embedding_size, rank_ratio=rank_ratio)
    else:
        raise ValueError(f"Unknown EdgeFace model: {name}")
