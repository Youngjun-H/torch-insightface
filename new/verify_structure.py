"""
모델 구조를 사용자가 제공한 구조와 비교하여 검증
"""

import torch
from models.backbones.edgenext_gamma_06 import edgenext_xs_gamma_06


def verify_structure():
    """모델 구조 검증"""
    model = edgenext_xs_gamma_06(featdim=512, rank_ratio=0.6)

    print("=" * 80)
    print("모델 구조 검증")
    print("=" * 80)

    # Stem 확인
    print("\n[Stem]")
    print(f"  Conv2d: {model.model.stem[0]}")
    print(f"  LayerNorm2d: {model.model.stem[1]}")

    # Stage 0 확인
    print("\n[Stage 0]")
    stage0 = model.model.stages[0]
    print(f"  Downsample: {type(stage0.downsample).__name__}")
    print(f"  Blocks 개수: {len(stage0.blocks)}")
    print(f"  첫 번째 ConvBlock dim: {stage0.blocks[0].conv_dw.in_channels}")

    # Stage 1 확인
    print("\n[Stage 1]")
    stage1 = model.model.stages[1]
    print(f"  Downsample: {type(stage1.downsample).__name__}")
    if isinstance(stage1.downsample, torch.nn.Sequential):
        print(f"    - LayerNorm2d: {stage1.downsample[0]}")
        print(f"    - Conv2d: {stage1.downsample[1]}")
    print(f"  Blocks 개수: {len(stage1.blocks)}")
    print(f"  ConvBlocks: {len([b for b in stage1.blocks if hasattr(b, 'conv_dw')])}")
    split_block1 = stage1.blocks[2]
    print(f"  SplitTransposeBlock:")
    print(f"    - num_splits: {split_block1.num_splits}")
    print(f"    - convs 개수: {len(split_block1.convs)}")
    print(f"    - 첫 번째 conv: {split_block1.convs[0]}")
    print(f"    - pos_embd hidden_dim: {split_block1.pos_embd.hidden_dim}")
    print(f"    - pos_embd dim: {split_block1.pos_embd.dim}")

    # Stage 2 확인
    print("\n[Stage 2]")
    stage2 = model.model.stages[2]
    print(f"  Downsample: {type(stage2.downsample).__name__}")
    print(f"  Blocks 개수: {len(stage2.blocks)}")
    print(f"  ConvBlocks: {len([b for b in stage2.blocks if hasattr(b, 'conv_dw')])}")
    split_block2 = stage2.blocks[8]
    print(f"  SplitTransposeBlock:")
    print(f"    - num_splits: {split_block2.num_splits}")
    print(f"    - convs 개수: {len(split_block2.convs)}")
    print(f"    - 첫 번째 conv: {split_block2.convs[0]}")

    # Stage 3 확인
    print("\n[Stage 3]")
    stage3 = model.model.stages[3]
    print(f"  Downsample: {type(stage3.downsample).__name__}")
    print(f"  Blocks 개수: {len(stage3.blocks)}")
    print(f"  ConvBlocks: {len([b for b in stage3.blocks if hasattr(b, 'conv_dw')])}")
    split_block3 = stage3.blocks[2]
    print(f"  SplitTransposeBlock:")
    print(f"    - num_splits: {split_block3.num_splits}")
    print(f"    - convs 개수: {len(split_block3.convs)}")
    print(f"    - 첫 번째 conv: {split_block3.convs[0]}")

    # Head 확인
    print("\n[Head]")
    print(f"  NormMlpClassifierHead: {model.model.head}")
    print(f"  FC: {model.model.head.fc}")

    # Forward pass 테스트
    print("\n" + "=" * 80)
    print("Forward Pass 테스트")
    print("=" * 80)
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Forward pass 성공!")

    return model


if __name__ == "__main__":
    model = verify_structure()
