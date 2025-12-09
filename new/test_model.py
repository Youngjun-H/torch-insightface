"""
모델 구조 검증 스크립트
"""

import torch
from models.backbones.edgenext_gamma_06 import edgenext_xs_gamma_06


def test_model_structure():
    """모델 구조를 출력하고 검증"""
    model = edgenext_xs_gamma_06(featdim=512, rank_ratio=0.6)

    print("=" * 80)
    print("EdgeNeXt XS Gamma 0.6 Model Structure")
    print("=" * 80)
    print(model)
    print("\n" + "=" * 80)

    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 80)

    # Forward pass 테스트
    x = torch.randn(1, 3, 112, 112)
    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")

    print("\n✓ Model test passed!")
    return model


if __name__ == "__main__":
    model = test_model_structure()
