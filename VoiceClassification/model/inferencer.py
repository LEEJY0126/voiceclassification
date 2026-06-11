"""
inferencer.py

학습된 모델을 로드하고 window 단위로 추론.
ROS2 노드, CLI 스크립트 등 어디서든 재사용 가능.

Usage:
    inferencer = Inferencer("checkpoints/best.pt")
    label, confidence = inferencer.predict(window)  # window: np.ndarray [16000]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from VoiceClassification.model.model import VoiceClassifier


LABEL_NAMES = {
    0: "기타",
    1: "일로와",
    2: "가",
    3: "멈춰",
}


class Inferencer:
    """
    Args:
        checkpoint_path : best.pt 경로
        device          : 'cpu' / 'cuda' / None (자동 선택)
    """

    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = VoiceClassifier().to(self.device)
        self._load(checkpoint_path)
        self.model.eval()

        print(f"[Inferencer] 모델 로드 완료 | device={self.device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, window: np.ndarray) -> tuple[int, float]:
        """
        Args:
            window : np.ndarray [16000] float32  (VoiceBuffer 출력)

        Returns:
            label      : int   (0 / 1 / 2)
            confidence : float (0.0 ~ 1.0)
        """
        x      = self._to_tensor(window)           # [1, 16000]
        logits = self.model(x)                     # [1, 3]
        probs  = torch.softmax(logits, dim=1)[0]   # [3]

        label      = int(probs.argmax().item())
        confidence = float(probs[label].item())
        return label, confidence

    def predict_batch(self, windows: list[np.ndarray]) -> list[tuple[int, float]]:
        """여러 window를 한 번에 추론 (파일 전체 처리용)"""
        x      = torch.stack([self._to_tensor(w).squeeze(0) for w in windows]).to(self.device)
        logits = self.model(x)                     # [N, 3]
        probs  = torch.softmax(logits, dim=1)      # [N, 3]

        results = []
        for p in probs:
            label      = int(p.argmax().item())
            confidence = float(p[label].item())
            results.append((label, confidence))
        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        epoch    = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", float("nan"))
        val_acc  = ckpt.get("val_acc", float("nan"))
        print(f"[Inferencer] epoch={epoch} | val_loss={val_loss:.4f} | val_acc={val_acc:.1%}")

    def _to_tensor(self, window: np.ndarray) -> torch.Tensor:
        return torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)