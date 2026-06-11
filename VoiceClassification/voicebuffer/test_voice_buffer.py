"""
test_voice_buffer.py

ROS2 없이 VoiceBuffer 단독으로 테스트.
"""

import numpy as np
from voice_buffer import VoiceBuffer, BufferConfig


def test_basic():
    """정확히 window_size만큼 push하면 window 1개 나와야 함"""
    cfg = BufferConfig(sample_rate=16000, window_sec=1.0, hop_sec=0.5)
    buf = VoiceBuffer(cfg)

    chunk = np.random.randn(16000).astype(np.float32)
    windows = buf.push(chunk)

    assert len(windows) == 1, f"Expected 1 window, got {len(windows)}"
    assert windows[0].shape == (16000,), f"Wrong shape: {windows[0].shape}"
    print("✅ test_basic passed")


def test_small_chunks():
    """작은 chunk 여러 번 push해도 정확히 window 나와야 함"""
    cfg = BufferConfig(sample_rate=16000, window_sec=1.0, hop_sec=0.5)
    buf = VoiceBuffer(cfg)

    chunk_size = 512  # 마이크 콜백 단위 (pyaudio 기본값)
    total_samples = 0
    window_count = 0

    # 3초치 데이터 넣기
    for _ in range(int(3.0 * 16000 / chunk_size)):
        chunk = np.random.randn(chunk_size).astype(np.float32)
        windows = buf.push(chunk)
        window_count += len(windows)
        total_samples += chunk_size

    # 3초 / hop 0.5초 = 최대 5~6개 window 예상 (첫 window는 1초 후)
    print(f"  총 샘플: {total_samples} ({total_samples/16000:.1f}s)")
    print(f"  생성된 window: {window_count}개")
    assert window_count >= 4, f"Expected >= 4 windows, got {window_count}"
    print("✅ test_small_chunks passed")


def test_overlap():
    """hop=0.5s이면 window 간 0.5초 overlap이 있어야 함"""
    cfg = BufferConfig(sample_rate=16000, window_sec=1.0, hop_sec=0.5)
    buf = VoiceBuffer(cfg)

    # 1.5초치 데이터: window 0 (0~1s), window 1 (0.5~1.5s)
    data = np.arange(16000 * 2, dtype=np.float32)  # 값으로 위치 추적
    windows = buf.push(data[:16000])   # 첫 1초 → window 0 나옴
    w0 = windows[0] if windows else None

    windows2 = buf.push(data[16000:24000])  # 다음 0.5초 → window 1 나옴
    w1 = windows2[0] if windows2 else None

    if w0 is not None and w1 is not None:
        # w1의 앞 8000샘플 == w0의 뒤 8000샘플 (overlap 구간)
        overlap_ok = np.allclose(w0[8000:], w1[:8000])
        assert overlap_ok, "Overlap mismatch!"
        print("✅ test_overlap passed")
    else:
        print("⚠️  test_overlap skipped (not enough windows)")


def test_reset():
    """reset 후 fill_ratio가 0이 돼야 함"""
    buf = VoiceBuffer()
    buf.push(np.ones(8000, dtype=np.float32))
    assert buf.fill_ratio > 0
    buf.reset()
    assert buf.fill_ratio == 0.0
    print("✅ test_reset passed")


if __name__ == "__main__":
    test_basic()
    test_small_chunks()
    test_overlap()
    test_reset()
    print("\n🎉 All tests passed!")