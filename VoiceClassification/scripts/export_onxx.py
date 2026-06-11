"""
export_onnx.py

PyTorch 모델 → ONNX 변환 스크립트

Usage:
    python3 export_onnx.py
    python3 export_onnx.py --checkpoint best_5.pt --output model.onnx
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.model import VoiceClassifier


def export(checkpoint_path: str, output_path: str):
    # 모델 로드
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = VoiceClassifier()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"체크포인트 : {checkpoint_path}")
    print(f"epoch      : {ckpt.get('epoch', '?')}")
    print(f"val_loss   : {ckpt.get('val_loss', float('nan')):.4f}")
    print(f"val_acc    : {ckpt.get('val_acc', float('nan')):.1%}")

    # 더미 입력 (raw waveform 1초 @ 16kHz)
    dummy_input = torch.randn(1, 16000)

    # ONNX 변환
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names  = ["waveform"],
        output_names = ["logits"],
        dynamic_axes = {
            "waveform": {0: "batch_size"},
            "logits"  : {0: "batch_size"},
        },
        opset_version   = 17,
        do_constant_folding = True,
    )
    print(f"\n✅ ONNX 저장 완료: {output_path}")

    # 검증
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(output_path)
        dummy   = np.random.randn(1, 16000).astype(np.float32)
        result  = session.run(None, {"waveform": dummy})
        print(f"✅ ONNX 검증 완료 | output shape: {result[0].shape}")
    except ImportError:
        print("⚠️  onnxruntime 없어서 검증 스킵 (pip install onnxruntime)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_5.pt")
    parser.add_argument("--output",     default="voice_classifier.onnx")
    args = parser.parse_args()

    export(args.checkpoint, args.output)