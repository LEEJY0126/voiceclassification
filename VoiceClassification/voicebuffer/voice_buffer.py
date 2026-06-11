import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BufferConfig:
    sample_rate: int = 16000
    window_sec: float = 1.0
    hop_sec: float = 0.5

    @property
    def window_size(self) -> int:
        return int(self.sample_rate * self.window_sec)

    @property
    def hop_size(self) -> int:
        return int(self.sample_rate * self.hop_sec)


class VoiceBuffer:
    """
    Sliding window voice buffer.

    마이크에서 들어오는 chunk를 받아서 window가 찰 때마다
    1초짜리 np.ndarray를 yield함.

    Timeline example (window=1s, hop=0.5s):
        [------- window 0 -------]
                    [------- window 1 -------]
                                [------- window 2 -------]
        0s   0.5s   1.0s  1.5s   2.0s
    """

    def __init__(self, config: Optional[BufferConfig] = None):
        self.cfg = config or BufferConfig()

        # 내부 circular buffer
        self._buf = np.zeros(self.cfg.window_size, dtype=np.float32)
        # 현재 채워진 샘플 수 (hop 이후 남은 샘플 포함)
        self._filled: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        """
        오디오 chunk를 버퍼에 넣고, 완성된 window 리스트를 반환.

        Args:
            chunk: shape (N,), dtype float32, 범위 [-1.0, 1.0]

        Returns:
            완성된 1초 window들의 리스트 (보통 0개 또는 1개)
        """
        chunk = self._validate(chunk)
        windows = []
        idx = 0

        while idx < len(chunk):
            # 버퍼의 남은 공간
            space = self.cfg.window_size - self._filled
            take = min(space, len(chunk) - idx)

            self._buf[self._filled: self._filled + take] = chunk[idx: idx + take]
            self._filled += take
            idx += take

            if self._filled == self.cfg.window_size:
                windows.append(self._buf.copy())
                self._slide()

        return windows

    def reset(self):
        """버퍼 초기화 (에러 복구 또는 재시작 시 사용)"""
        self._buf[:] = 0.0
        self._filled = 0

    @property
    def fill_ratio(self) -> float:
        """현재 버퍼가 얼마나 찼는지 (0.0 ~ 1.0)"""
        return self._filled / self.cfg.window_size

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _slide(self):
        """hop만큼 앞으로 밀기 (overlap 유지)"""
        keep = self.cfg.window_size - self.cfg.hop_size
        self._buf[:keep] = self._buf[self.cfg.hop_size:]
        self._buf[keep:] = 0.0
        self._filled = keep

    @staticmethod
    def _validate(chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 1:
            raise ValueError(f"chunk must be 1-D, got shape {chunk.shape}")
        return chunk