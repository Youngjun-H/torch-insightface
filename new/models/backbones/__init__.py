"""
Backbone Factory for EdgeFace Gamma 0.6 Model
"""

from .edgenext_gamma_06 import edgenext_xs_gamma_06


def get_model(name, **kwargs):
    """
    EdgeFace 모델 생성

    Args:
        name: 모델 이름
            - 'edgeface_xs_gamma_06': EdgeNeXt X-Small with low-rank (rank_ratio=0.6)
        **kwargs: 추가 인자
            - num_features: embedding size (기본: 512) - Lightning 호환을 위해 num_features로 받음
            - featdim: embedding size (기본: 512) - 호환성 유지
            - batchnorm: batch normalization 사용 여부 (기본: False)
            - rank_ratio: LoRA rank ratio (기본: 0.6)

    Returns:
        모델 인스턴스
    """
    # Lightning 호환: num_features를 featdim으로 변환
    embedding_size = kwargs.get("num_features", kwargs.get("featdim", 512))
    batchnorm = kwargs.get("batchnorm", False)
    rank_ratio = kwargs.get("rank_ratio", 0.6)

    if name == "edgeface_xs_gamma_06":
        return edgenext_xs_gamma_06(featdim=embedding_size, rank_ratio=rank_ratio)
    else:
        raise ValueError(f"Unknown EdgeFace model: {name}")
