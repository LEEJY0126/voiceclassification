"""
inferencer_onnx.py

ONNX Runtime 기반 음성 명령 추론.
PyTorch 불필요 → 라즈베리파이에서 훨씬 빠름.

Usage:
    inferencer = InferencerOnnx("voice_classifier.onnx")
    label, confidence = inferencer.predict(window)  # window: np.ndarray [16000]
"""

import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime 설치 필요: pip install onnxruntime")


LABEL_NAMES = {
    0: "기타",
    1: "일로와",
    2: "가",
    3: "멈춰",
}


class InferencerOnnx:
    """
    Args:
        model_path : voice_classifier.onnx 경로
    """

    def __init__(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX 모델 없음: {path}")

        self.session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        print(f"[InferencerOnnx] 모델 로드 완료: {path.name}")
        print(f"[InferencerOnnx] input='{self.input_name}' shape={self.session.get_inputs()[0].shape}")

    # ------------------------------------------------------------------

    def predict(self, window: np.ndarray) -> tuple[int, float]:
        """
        Args:
            window : np.ndarray [16000] float32

        Returns:
            label      : int   (0 / 1 / 2 / 3)
            confidence : float (0.0 ~ 1.0)
        """
        x      = window.astype(np.float32).reshape(1, -1)
        logits = self.session.run(None, {self.input_name: x})[0]  # [1, 4]
        probs  = self._softmax(logits[0])

        label      = int(np.argmax(probs))
        confidence = float(probs[label])
        return label, confidence

    def predict_batch(self, windows: list[np.ndarray]) -> list[tuple[int, float]]:
        """여러 window 한 번에 추론"""
        x      = np.stack([w.astype(np.float32) for w in windows])
        logits = self.session.run(None, {self.input_name: x})[0]

        results = []
        for logit in logits:
            probs      = self._softmax(logit)
            label      = int(np.argmax(probs))
            confidence = float(probs[label])
            results.append((label, confidence))
        return results

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()