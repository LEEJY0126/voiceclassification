"""
inference.py

오디오 파일을 입력받아 window 단위로 추론 후 결과 출력.

Usage:
    python3 inference.py sample.wav --checkpoint ../checkpoints/best.pt
    python3 inference.py sample.wav --checkpoint ../checkpoints/best.pt --threshold 0.7
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from voicebuffer.voice_buffer import VoiceBuffer, BufferConfig
from model.inferencer import Inferencer, LABEL_NAMES


# ------------------------------------------------------------------

def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    audio, orig_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if orig_sr != target_sr:
        g = gcd(orig_sr, target_sr)
        audio = resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
    return audio


def run(wav_path: str, checkpoint: str, threshold: float = 0.0):
    path = Path(wav_path)
    if not path.exists():
        print(f"❌ 파일 없음: {path}")
        sys.exit(1)

    # 모델 로드
    inferencer = Inferencer(checkpoint)

    # 오디오 로드 → window 수집
    audio = load_audio(path)
    # 1초 미만이면 zero-pad
    window_size = 16000
    if len(audio) < window_size:
        pad   = np.zeros(window_size - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, pad])

    cfg   = BufferConfig()
    buf   = VoiceBuffer(cfg)

    windows = []
    for i in range(0, len(audio), 512):
        for w in buf.push(audio[i:i + 512]):
            windows.append(w)

    if not windows:
        print("❌ window 생성 실패 (오디오가 1초 이상인지 확인)")
        sys.exit(1)

    print(f"\n파일  : {path.name}  ({len(audio)/16000:.2f}s)")
    print(f"window: {len(windows)}개 (1초, hop 0.5s)\n")
    print(f"{'Window':>8} │ {'Label':>6} {'Name':>8} │ {'Confidence':>10} │ {'결과':>10}")
    print("─" * 55)

    # window 단위 추론
    results = inferencer.predict_batch(windows)

    detected = []
    for i, (label, conf) in enumerate(results):
        hop_start = i * cfg.hop_sec
        marker = "◀" if (conf >= threshold and label != 0) else ""
        print(
            f"{i:>8} │ {label:>6} {LABEL_NAMES[label]:>8} │ {conf:>9.1%} │ {marker}"
        )
        if conf >= threshold and label != 0:
            detected.append((i, label, conf, hop_start))

    # 감지 요약
    print("\n─" * 55)
    if detected:
        print("감지된 명령어:")
        for i, label, conf, t in detected:
            print(f"  window {i} ({t:.1f}s~{t+1.0:.1f}s) → [{label}] {LABEL_NAMES[label]}  ({conf:.1%})")
    else:
        print("감지된 명령어 없음")


# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav",          help="입력 wav 파일")
    parser.add_argument("--checkpoint", required=True, help="best.pt 경로")
    parser.add_argument("--threshold",  type=float, default=0.0,
                        help="이 confidence 이상일 때만 감지로 판단 (default: 0.0)")
    args = parser.parse_args()

    run(args.wav, args.checkpoint, args.threshold)