import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.ghostfacenet_module import GhostFaceNetModule


class GhostFaceNetIneference(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.gdc = model.gdc


    def forward(self, x):
        x = self.backbone(x)
        x = self.gdc(x)        

        x = x.reshape(x.shape[0], -1)
        x = F.normalize(x, dim=1)

        return x

def convert_to_onnx(onnx_model: GhostFaceNetIneference, output_path: str):
    dummy = torch.randn(1, 3, 112, 112)

    # 출력 디렉토리가 없으면 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    torch.onnx.export(
        onnx_model,
        dummy,
        output_path,

        input_names=["input"],
        output_names=["embedding"],

        opset_version=11, # 중요함!
        dynamo=False, # 중요함!

        dynamic_axes={
            "input": {0: "batch"},
            "embedding": {0: "batch"},
        },
    )

    onnx.checker.check_model(output_path)
    print(f"✅ GhostFaceNet ONNX export 성공: {output_path}")

def validate_onnx_model(onnx_path: str, use_cuda: bool = False):
    providers = []

    if use_cuda:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                },
            )
        )

    providers.append("CPUExecutionProvider")

    print(f"📦 Loading ONNX model: {onnx_path}")
    print(f"🔧 Providers: {providers}")

    sess = ort.InferenceSession(onnx_path, providers=providers)

    input_meta = sess.get_inputs()[0]
    output_meta = sess.get_outputs()[0]

    print("\n=== Model IO Info ===")
    print(f"Input name  : {input_meta.name}")
    print(f"Input shape : {input_meta.shape}")
    print(f"Output name : {output_meta.name}")
    print(f"Output shape: {output_meta.shape}")

    # 기대 shape 체크
    assert output_meta.shape[-1] == 512, "❌ Output dim is not 512"

    # -----------------------------
    # 3. Batch size 변경 테스트
    # -----------------------------
    print("\n=== Dynamic Batch Test ===")

    for b in [1, 8, 32, 64, 128]:
        x = np.random.randn(b, 3, 112, 112).astype(np.float32)

        y = sess.run(
            None,
            {input_meta.name: x},
        )

        emb = y[0]
        print(f"Batch {b:>2d} → output shape: {emb.shape}")

        # shape 검증
        assert emb.shape == (b, 512), (
            f"❌ Shape mismatch: expected {(b, 512)}, got {emb.shape}"
        )

        # L2 normalize 체크 (선택)
        norms = np.linalg.norm(emb, axis=1)
        print(f"          L2 norm (mean): {norms.mean():.4f}")

    print("\n✅ ONNX model validation PASSED")


def convert_and_validate(checkpoint_path: str, output_path: str):
    # model load
    lit_model = GhostFaceNetModule.load_from_checkpoint(checkpoint_path)
    lit_model.eval()

    onnx_model = GhostFaceNetIneference(lit_model.model)
    onnx_model.eval()
    onnx_model.to("cpu")

    convert_to_onnx(onnx_model, output_path)

    validate_onnx_model(output_path, use_cuda=True)

    print("completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GhostFaceNet checkpoint to ONNX")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--output_path", type=str, default="lightning_ghostfacenets/ghostfacenet.onnx", help="Path to output ONNX model file (default: lightning_ghostfacenets/ghostfacenet.onnx)")
    args = parser.parse_args()
    convert_and_validate(args.checkpoint_path, args.output_path)