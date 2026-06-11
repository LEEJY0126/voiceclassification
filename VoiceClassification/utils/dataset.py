"""
dataset.py

data/
├── 0/  기타 (통화녹음, 노이즈)
├── 1/  라붕아, 일로와
├── 2/  가, stop
└── 3/  멈춰
처리:
    - 1초 미만 파일 → zero-pad → 1초 window 1개 생성
    - 긴 파일 → max_windows_per_file 개수 제한 (균등 샘플링)
    - 파일 단위 train/val 분리
    - WeightedRandomSampler로 클래스 불균형 보정
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from sklearn.model_selection import train_test_split

from voicebuffer.voice_buffer import VoiceBuffer, BufferConfig


SUPPORTED_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


class VoiceDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[np.ndarray, int]],
        is_train: bool = True,
        amp_range: tuple[float, float] = (0.5, 1.2),
        noise_pool: list[np.ndarray] | None = None,
        noise_ratio_range: tuple[float, float] = (0.2, 0.5),
    ):
        self.samples           = samples
        self.is_train          = is_train
        self.amp_min           = amp_range[0]
        self.amp_max           = amp_range[1]
        self.noise_pool        = noise_pool or []
        self.noise_ratio_min   = noise_ratio_range[0]
        self.noise_ratio_max   = noise_ratio_range[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, label = self.samples[idx]
        window = window.copy()

        if self.is_train:
            # 1. amplitude augmentation
            gain   = np.random.uniform(self.amp_min, self.amp_max)
            window = window * gain

            # 2. background noise mixing (class 2가 아닐 때만 적용)
            #    class 2는 이미 noise라서 섞을 필요 없음
            if self.noise_pool and label != 0:  # 기타(0)는 noise 자체라 mixing 제외
                noise_idx   = np.random.randint(len(self.noise_pool))
                noise       = self.noise_pool[noise_idx].copy()
                noise_ratio = np.random.uniform(self.noise_ratio_min, self.noise_ratio_max)
                window      = window + noise_ratio * noise

            window = np.clip(window, -1.0, 1.0)

        return torch.tensor(window, dtype=torch.float32), label

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        """
        클래스별 샘플 수에 반비례하는 가중치로 샘플러 생성.
        적은 클래스를 더 자주 뽑아 불균형 보정.
        """
        counts = self.class_counts()
        # 클래스별 가중치: 샘플 수가 적을수록 높은 가중치
        class_weights = {cls: 1.0 / cnt for cls, cnt in counts.items()}

        # 각 샘플에 가중치 부여
        sample_weights = [
            class_weights[label] for _, label in self.samples
        ]

        return WeightedRandomSampler(
            weights     =sample_weights,
            num_samples =len(sample_weights),
            replacement =True,
        )


# ------------------------------------------------------------------
# 오디오 로드 유틸
# ------------------------------------------------------------------

def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    """
    wav/flac/ogg  → soundfile로 로드
    m4a/mp3 등    → pydub으로 fallback (ffmpeg 필요)
    """
    try:
        audio, orig_sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    except Exception:
        # soundfile 미지원 포맷 (m4a 등) → pydub fallback
        from pydub import AudioSegment
        seg    = AudioSegment.from_file(path)
        seg    = seg.set_channels(1).set_frame_rate(target_sr)
        audio  = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
        return audio  # pydub이 이미 target_sr로 변환

    # soundfile 경로: 필요시 리샘플링
    if orig_sr != target_sr:
        g = gcd(orig_sr, target_sr)
        audio = resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
    return audio


def _file_to_windows(
    path: Path,
    cfg: BufferConfig,
    max_windows: int | None = None,
) -> list[np.ndarray]:
    """
    오디오 파일 → window 리스트

    - 1초 미만 파일: zero-pad 후 window 1개 생성
    - 긴 파일: max_windows 개수만큼 균등 샘플링
    """
    try:
        audio = _load_audio(path, cfg.sample_rate)
    except Exception as e:
        print(f"  ❌ {path.name} 로드 실패: {e}")
        return []

    # 1초 미만 → zero-pad
    if len(audio) < cfg.window_size:
        pad    = np.zeros(cfg.window_size - len(audio), dtype=np.float32)
        audio  = np.concatenate([audio, pad])

    # window 생성
    buf     = VoiceBuffer(cfg)
    windows = []
    for i in range(0, len(audio), 512):
        for w in buf.push(audio[i:i + 512]):
            windows.append(w)

    # 긴 파일: 균등 샘플링으로 max_windows 제한
    if max_windows and len(windows) > max_windows:
        indices = np.linspace(0, len(windows) - 1, max_windows, dtype=int)
        windows = [windows[i] for i in indices]

    return windows


# ------------------------------------------------------------------
# DataLoader 빌더
# ------------------------------------------------------------------

def _split_noise_files(
    files: list[Path],
    call_pattern: str,
) -> tuple[list[Path], list[Path]]:
    """파일명 패턴으로 통화녹음 / 기타 소음 분리"""
    call_files  = [f for f in files if call_pattern in f.name]
    other_files = [f for f in files if call_pattern not in f.name]
    return call_files, other_files


def _collect_windows_balanced(
    call_files:  list[Path],
    other_files: list[Path],
    cfg: BufferConfig,
    call_ratio: float,
    max_windows_per_file: int | None,
) -> list[np.ndarray]:
    """
    통화녹음과 기타 소음을 call_ratio 비율로 균형있게 수집.

    call_ratio=0.5  → 통화녹음 50%, 기타 50%
    call_ratio=0.3  → 통화녹음 30%, 기타 70%
    """
    # 각 그룹에서 window 수집
    call_windows  = []
    for f in call_files:
        call_windows.extend(_file_to_windows(f, cfg, max_windows=max_windows_per_file))

    other_windows = []
    for f in other_files:
        other_windows.extend(_file_to_windows(f, cfg, max_windows=max_windows_per_file))

    if not call_windows:
        return other_windows
    if not other_windows:
        return call_windows

    # call_ratio에 맞게 두 그룹 샘플링
    total        = len(call_windows) + len(other_windows)
    call_target  = int(total * call_ratio)
    other_target = total - call_target

    rng = np.random.default_rng()   # 시드 없음 → 매번 다른 window 선택

    call_sampled = (
        call_windows if len(call_windows) <= call_target
        else [call_windows[i] for i in rng.choice(len(call_windows), call_target, replace=False)]
    )
    other_sampled = (
        other_windows if len(other_windows) <= other_target
        else [other_windows[i] for i in rng.choice(len(other_windows), other_target, replace=False)]
    )

    print(f"    통화녹음: {len(call_windows)}개 → {len(call_sampled)}개 사용")
    print(f"    기타소음: {len(other_windows)}개 → {len(other_sampled)}개 사용")

    return call_sampled + other_sampled


def build_dataloaders(
    data_dir: str,
    val_ratio: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 0,
    cfg: BufferConfig | None = None,
    seed: int = 42,
    max_windows_per_file: int | None = 200,
    call_pattern: str = "통화 녹음",
    noise_call_ratio: float = 0.5,
) -> tuple[DataLoader, DataLoader]:
    """
    파일 단위 train/val 분리 + WeightedRandomSampler 적용

    Args:
        max_windows_per_file : 파일당 최대 window 수
        call_pattern         : 통화녹음 파일 식별 패턴 (파일명 포함 여부)
        noise_call_ratio     : class 2에서 통화녹음 비율 (0.0~1.0)

    Returns:
        train_loader, val_loader
    """
    cfg      = cfg or BufferConfig()
    data_dir = Path(data_dir)

    train_samples: list[tuple[np.ndarray, int]] = []
    val_samples:   list[tuple[np.ndarray, int]] = []

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"{data_dir} 안에 클래스 폴더가 없습니다.")

    print("=== 파일 단위 train/val 분리 ===")
    for class_dir in class_dirs:
        label = int(class_dir.name)
        files = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXT])

        if not files:
            print(f"  class {label}: 파일 없음 ⚠️")
            continue

        # 파일 단위 split
        if len(files) == 1:
            train_files, val_files = files, []
        else:
            train_files, val_files = train_test_split(
                files, test_size=val_ratio, random_state=seed
            )

        # class 2: 통화녹음/기타소음 비율 맞춰서 수집
        if label == 0:
            print(f"  class {label}: 통화녹음 비율 {noise_call_ratio:.0%} 적용")
            call_tr, other_tr = _split_noise_files(train_files, call_pattern)
            train_windows     = _collect_windows_balanced(
                call_tr, other_tr, cfg, noise_call_ratio, max_windows_per_file
            )
            for w in train_windows:
                train_samples.append((w, label))

            call_vl, other_vl = _split_noise_files(val_files, call_pattern)
            val_windows       = _collect_windows_balanced(
                call_vl, other_vl, cfg, noise_call_ratio, max_windows_per_file
            )
            for w in val_windows:
                val_samples.append((w, label))

            print(
                f"  class {label}: "
                f"train {len(train_files)}파일({len(train_windows)}window) | "
                f"val {len(val_files)}파일({len(val_windows)}window)"
            )

        # class 0, 1: 기존 방식 그대로
        else:
            train_w = 0
            for f in train_files:
                for w in _file_to_windows(f, cfg, max_windows=max_windows_per_file):
                    train_samples.append((w, label))
                    train_w += 1

            val_w = 0
            for f in val_files:
                for w in _file_to_windows(f, cfg, max_windows=None):
                    val_samples.append((w, label))
                    val_w += 1

            print(
                f"  class {label}: "
                f"train {len(train_files)}파일({train_w}window) | "
                f"val {len(val_files)}파일({val_w}window)"
            )

    # class 2 train window → noise pool 구성
    noise_pool = [w for w, lbl in train_samples if lbl == 0]  # class 0(기타)를 noise pool로 사용
    print(f"\n[NoisePool] class 0 train window {len(noise_pool)}개 → noise augmentation에 사용")

    train_dataset = VoiceDataset(train_samples, is_train=True,  noise_pool=noise_pool)
    val_dataset   = VoiceDataset(val_samples,   is_train=False)

    counts = train_dataset.class_counts()
    print(f"[train] 총 {len(train_dataset)}개 | 클래스별: {counts}")
    print(f"[val]   총 {len(val_dataset)}개   | 클래스별: {val_dataset.class_counts()}")

    # WeightedRandomSampler (train만 적용)
    sampler = train_dataset.make_weighted_sampler()
    print(f"\n[WeightedSampler] 클래스 가중치: { {k: f'{1/v:.1f}samples' for k,v in counts.items()} }")

    train_loader = DataLoader(
        train_dataset,
        batch_size  =batch_size,
        sampler     =sampler,        # shuffle=True 대신 sampler 사용
        num_workers =num_workers,
        pin_memory  =torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  =batch_size,
        shuffle     =False,
        num_workers =num_workers,
        pin_memory  =torch.cuda.is_available(),
    )

    return train_loader, val_loader