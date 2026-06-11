"""
visualize_stft.py

오디오 파일 → VoiceBuffer → STFTProcessor → matplotlib 시각화

Usage:
    python3 visualize_stft.py sample.wav
    python3 visualize_stft.py sample.wav --max_windows 3
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 대응
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebuffer.voice_buffer import VoiceBuffer, BufferConfig
from stft_processor import STFTProcessor


# -----------------------------------------------------------------------

def load_audio(path: Path, target_sr=16000) -> np.ndarray:
    audio, orig_sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    if orig_sr != target_sr:
        g = gcd(orig_sr, target_sr)
        audio = resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
    return audio


def visualize(wav_path: str, max_windows: int = 4, output: str = "stft_visualization.png"):
    path = Path(wav_path)
    cfg  = BufferConfig(sample_rate=16000, window_sec=1.0, hop_sec=0.5)
    proc = STFTProcessor()

    # 오디오 로드
    audio = load_audio(path)
    print(f"파일: {path.name} | {len(audio)/16000:.2f}s | {len(audio)} samples")

    # 버퍼에 chunk 단위로 push → windows 수집
    buf     = VoiceBuffer(cfg)
    windows = []
    for i in range(0, len(audio), 512):
        chunk = audio[i:i+512]
        for w in buf.push(chunk):
            windows.append(w)
            if len(windows) >= max_windows:
                break
        if len(windows) >= max_windows:
            break

    if not windows:
        print("❌ window가 생성되지 않았습니다. 오디오가 1초 이상인지 확인하세요.")
        return

    print(f"생성된 window: {len(windows)}개")

    # -----------------------------------------------------------------------
    # 시각화: window마다 [waveform | spectrogram] 2열
    # -----------------------------------------------------------------------
    n   = len(windows)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
    if n == 1:
        axes = [axes]

    t_wave = np.linspace(0, 1.0, cfg.window_size)
    freqs  = np.linspace(0, 8000, proc.n_freqs)   # 0~8kHz (nyquist)
    times  = np.linspace(0, 1.0, proc.n_frames)

    for i, window in enumerate(windows):
        mag    = proc.process(window)              # [63, 513]
        mag_db = proc.to_db(mag)

        ax_wave, ax_spec = axes[i]

        # --- Waveform ---
        ax_wave.plot(t_wave, window, color="#4C9BE8", linewidth=0.7)
        ax_wave.set_title(f"Window {i}  |  Waveform", fontsize=11)
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_xlim(0, 1.0)
        ax_wave.set_ylim(-1.0, 1.0)
        ax_wave.grid(True, alpha=0.3)

        # --- Spectrogram ---
        im = ax_spec.imshow(
            mag_db.T,                              # [F, T] for imshow
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap="magma",
            vmin=-80, vmax=0,
        )
        ax_spec.set_title(f"Window {i}  |  Magnitude Spectrogram (dB)", fontsize=11)
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")
        plt.colorbar(im, ax=ax_spec, label="dB")

    fig.suptitle(f"{path.name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=120, bbox_inches="tight")
    print(f"✅ 저장: {output}")


# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav",          help="입력 wav 파일 경로")
    parser.add_argument("--max_windows", type=int, default=4)
    parser.add_argument("--output",      default="stft_visualization.png")
    args = parser.parse_args()

    visualize(args.wav, args.max_windows, args.output)