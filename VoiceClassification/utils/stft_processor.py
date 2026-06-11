import numpy as np
from scipy.signal import stft


class STFTProcessor:
    """
    VoiceBuffer에서 나온 window에 STFT를 적용.

    모델의 STFTFrontend와 동일한 파라미터 사용:
        n_fft=1024, hop_length=256, sr=16000

    Input : np.ndarray [16000]          (VoiceBuffer 출력)
    Output: np.ndarray [63, 513] float32 (magnitude spectrogram)
    """

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256):
        self.sample_rate = sample_rate
        self.n_fft       = n_fft
        self.hop_length  = hop_length

        # 주파수 / 시간 축 크기
        self.n_freqs = n_fft // 2 + 1   # 513
        self.n_frames = (sample_rate - n_fft) // hop_length + 1  # 63

    def process(self, window: np.ndarray) -> np.ndarray:
        """
        Args:
            window: [16000] float32

        Returns:
            magnitude: [T, F] = [63, 513] float32
        """
        if window.ndim != 1:
            raise ValueError(f"1-D array 필요, got shape {window.shape}")

        _, _, Zxx = stft(
            window,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            window="hann",
            boundary=None,   # 모델 torch.stft와 동일하게 패딩 없음
            padded=False,
        )
        # Zxx: [F, T] complex → magnitude → [T, F]
        magnitude = np.abs(Zxx).T.astype(np.float32)
        return magnitude

    def to_db(self, magnitude: np.ndarray, ref=1.0, amin=1e-5) -> np.ndarray:
        """magnitude → dB 스케일 (시각화용)"""
        return 20 * np.log10(np.maximum(magnitude, amin) / ref)