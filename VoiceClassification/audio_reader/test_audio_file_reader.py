"""
test_audio_file_reader.py

실제 wav 파일 없이 합성 오디오로 AudioFileReader 테스트.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path

from voicebuffer import VoiceBuffer, BufferConfig
from audio_file_reader import AudioFileReader


def make_wav(duration_sec: float, sr: int = 16000) -> Path:
    """테스트용 임시 wav 파일 생성 (440Hz 사인파)"""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return Path(tmp.name)


def test_stream():
    """3초 파일 → window 4개 나와야 함 (window=1s, hop=0.5s)"""
    wav = make_wav(3.0)
    cfg = BufferConfig(sample_rate=16000, window_sec=1.0, hop_sec=0.5)
    reader = AudioFileReader(wav, buffer_config=cfg)

    count = 0
    for window in reader.stream():
        assert window.shape == (16000,), f"Wrong shape: {window.shape}"
        assert window.dtype == np.float32
        count += 1

    # (total - window) / hop + 1 = (3.0 - 1.0) / 0.5 + 1 = 5개
    assert count == 5, f"Expected 5 windows, got {count}"
    print(f"✅ test_stream passed ({count} windows)")
    wav.unlink()


def test_windows_list():
    """windows()가 stream()과 동일한 결과 반환하는지 확인"""
    wav = make_wav(2.0)
    cfg = BufferConfig(sample_rate=16000, window_sec=1.0, hop_sec=0.5)
    reader = AudioFileReader(wav, buffer_config=cfg)

    wins = reader.windows()
    # (2.0 - 1.0) / 0.5 + 1 = 3개
    assert len(wins) == 3, f"Expected 3 windows, got {len(wins)}"
    print(f"✅ test_windows_list passed ({len(wins)} windows)")
    wav.unlink()


def test_duration():
    """duration_sec, num_samples 프로퍼티 확인"""
    wav = make_wav(5.0)
    reader = AudioFileReader(wav)

    assert abs(reader.duration_sec - 5.0) < 0.01
    assert reader.num_samples == 16000 * 5
    print(f"✅ test_duration passed ({reader.duration_sec:.2f}s)")
    wav.unlink()


def test_file_not_found():
    """없는 파일 넣으면 FileNotFoundError"""
    try:
        AudioFileReader("nonexistent.wav")
        assert False, "Should have raised"
    except FileNotFoundError:
        print("✅ test_file_not_found passed")


if __name__ == "__main__":
    test_stream()
    test_windows_list()
    test_duration()
    test_file_not_found()
    print("\n🎉 All tests passed!")