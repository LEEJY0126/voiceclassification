import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from pathlib import Path
from typing import Generator

from voicebuffer import VoiceBuffer, BufferConfig


class AudioFileReader:
    """
    오디오 파일을 읽어 VoiceBuffer에 chunk 단위로 push하는 파이프라인.

    실시간 마이크 입력을 시뮬레이션하기 위해 chunk 단위로 쪼개서 push함.
    마이크로 교체할 때도 VoiceBuffer 인터페이스는 그대로 유지됨.

    Usage:
        reader = AudioFileReader("sample.wav")
        for window in reader.stream():
            # window: np.ndarray shape (16000,), float32
            run_inference(window)
    """

    def __init__(
        self,
        path: str,
        buffer_config: BufferConfig | None = None,
        chunk_size: int = 512,  # pyaudio 기본값과 동일하게 시뮬레이션
    ):
        self.path = Path(path)
        self.cfg = buffer_config or BufferConfig()
        self.chunk_size = chunk_size

        if not self.path.exists():
            raise FileNotFoundError(f"오디오 파일 없음: {self.path}")

        self._audio = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        오디오를 chunk 단위로 VoiceBuffer에 push하고,
        완성된 window를 하나씩 yield함.

        Yields:
            window: np.ndarray shape (window_size,), float32
        """
        buf = VoiceBuffer(self.cfg)
        total = len(self._audio)
        idx = 0

        while idx < total:
            chunk = self._audio[idx: idx + self.chunk_size]
            idx += self.chunk_size

            for window in buf.push(chunk):
                yield window

    def windows(self) -> list[np.ndarray]:
        """stream()을 모두 소진해 리스트로 반환 (소규모 파일용)"""
        return list(self.stream())

    @property
    def duration_sec(self) -> float:
        return len(self._audio) / self.cfg.sample_rate

    @property
    def num_samples(self) -> int:
        return len(self._audio)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> np.ndarray:
        """
        soundfile로 파일 로드 + 필요시 리샘플링 + mono 변환
        반환: float32 1-D array, [-1.0, 1.0]
        """
        audio, orig_sr = sf.read(self.path, dtype="float32", always_2d=True)

        # stereo → mono (채널 평균)
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        else:
            audio = audio[:, 0]

        # 리샘플링 (원본 sr != 목표 sr일 때만)
        target_sr = self.cfg.sample_rate
        if orig_sr != target_sr:
            g = gcd(orig_sr, target_sr)
            audio = resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)

        print(
            f"[AudioFileReader] {self.path.name} | "
            f"원본 sr={orig_sr} → {target_sr}Hz | "
            f"{len(audio) / target_sr:.2f}s"
        )
        return audio.astype(np.float32)