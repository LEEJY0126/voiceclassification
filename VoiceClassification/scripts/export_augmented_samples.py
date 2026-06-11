"""
export_augmented_samples.py

augmentation 결과를 wav로 저장해서 직접 들어볼 수 있게 함.

Usage:
    python3 export_augmented_samples.py --data_dir /path/to/data --n 3
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
from voicebuffer.voice_buffer import VoiceBuffer, BufferConfig
from utils.dataset import _load_audio, _file_to_windows, SUPPORTED_EXT


def export_samples(data_dir: str, n: int = 3, out_dir: str = "augmented_samples"):
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = BufferConfig()

    # noise pool: class 2 파일에서 구성
    noise_pool = []
    noise_dir  = data_dir / "0"  # class 0(기타)를 noise pool로 사용
    if noise_dir.exists():
        noise_files = sorted([f for f in noise_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXT])
        for f in noise_files:
            for w in _file_to_windows(f, cfg, max_windows=50):
                noise_pool.append(w)
        print(f"noise pool: {len(noise_pool)}개 window ({len(noise_files)}개 파일)")
    else:
        print("⚠️  2/ 디렉토리 없음 → noise mixing 생략")

    # class 0 파일에서 window 추출
    class1_dir = data_dir / "1"  # class 1(라붕아) 샘플로 augmentation 확인
    if not class1_dir.exists():
        print("❌ 1/ 디렉토리 없음")
        return

    files = sorted([f for f in class1_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXT])
    if not files:
        print("❌ 0/ 에 파일 없음")
        return

    windows = []
    for f in files:
        windows.extend(_file_to_windows(f, cfg, max_windows=None))

    if not windows:
        print("❌ window 생성 실패 (파일이 1초 미만이면 zero-pad됨)")
        return

    print(f"class 0: {len(windows)}개 window 추출")

    # n개 샘플에 대해 원본 / augmented 쌍으로 저장
    n = min(n, len(windows))

    # noise를 샘플마다 다르게 — 겹치지 않게 shuffle
    if noise_pool:
        noise_indices = np.random.choice(len(noise_pool), size=n, replace=len(noise_pool) < n)
    
    for i in range(n):
        window = windows[i].copy()

        # --- 원본 저장 ---
        orig_path = out_dir / f"sample_{i}_original.wav"
        sf.write(orig_path, window, cfg.sample_rate)

        # --- augmented 저장 ---
        aug = window.copy()

        # amplitude
        gain = np.random.uniform(0.5, 1.2)
        aug  = aug * gain

        # noise mixing
        noise_info = "없음"
        if noise_pool:
            noise       = noise_pool[noise_indices[i]].copy()
            noise_ratio = np.random.uniform(0.2, 0.5)
            aug         = aug + noise_ratio * noise
            noise_info  = f"{noise_ratio:.2f} (noise_pool[{noise_indices[i]}])"

        aug = np.clip(aug, -1.0, 1.0)

        aug_path = out_dir / f"sample_{i}_augmented_gain{gain:.2f}.wav"
        sf.write(aug_path, aug, cfg.sample_rate)

        print(f"  [{i}] gain={gain:.2f} | noise_ratio={noise_info}")
        print(f"       원본      → {orig_path.name}")
        print(f"       augmented → {aug_path.name}")

    print(f"\n✅ {out_dir}/ 에 {n*2}개 파일 저장 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="data/ 루트 경로")
    parser.add_argument("--n",        type=int, default=3, help="출력할 샘플 수")
    parser.add_argument("--out_dir",  default="augmented_samples")
    args = parser.parse_args()

    export_samples(args.data_dir, args.n, args.out_dir)