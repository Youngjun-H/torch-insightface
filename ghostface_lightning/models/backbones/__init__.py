"""
GhostNet Backbone Factory
"""

from .ghostnet import ghostnet_v1, ghostnet_v2


def get_model(name, **kwargs):
    """
    GhostNet 모델 생성

    Args:
        name: 모델 이름
            - 'ghostnetv1': GhostNet V1
            - 'ghostnetv2': GhostNet V2
        **kwargs: 추가 인자
            - num_features: embedding size (기본: 512)
            - width: width multiplier (기본: 1.3)
            - strides: stem strides (기본: 2)

    Returns:
        모델 인스턴스
    """
    embedding_size = kwargs.get("num_features", kwargs.get("featdim", 512))
    width = kwargs.get("width", 1.3)
    strides = kwargs.get("strides", 2)

    if name.lower() == "ghostnetv1":
        return ghostnet_v1(
            featdim=embedding_size,
            width=width,
            strides=strides,
        )
    elif name.lower() == "ghostnetv2":
        return ghostnet_v2(
            featdim=embedding_size,
            width=width,
            strides=strides,
        )
    else:
        raise ValueError(f"Unknown GhostNet model: {name}")
