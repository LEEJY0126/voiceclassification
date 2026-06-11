#!/usr/bin/env python3
"""
mic_node.py

ReSpeaker USB 4 Mic Array (6ch firmware)에서 ch0(processed audio)를
읽어 /audio_raw 토픽으로 발행.

Topic I/O:
    Publish : /audio_raw (std_msgs/Float32MultiArray)  [chunk, float32 -1~1]

Parameters:
    device_index  (int)   : pyaudio 장치 인덱스 (-1이면 자동 탐색)
    sample_rate   (int)   : 샘플레이트 (default: 16000)
    chunk_size    (int)   : 한 번에 읽을 샘플 수 (default: 512)
    n_channels    (int)   : 전체 채널 수 (6ch 펌웨어 = 6)
    audio_channel (int)   : 사용할 채널 인덱스 (ch0 = processed)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

try:
    import pyaudio
except ImportError:
    print("pyaudio 설치 필요: pip install pyaudio")
    sys.exit(1)


class MicNode(Node):
    def __init__(self):
        super().__init__("mic_node")

        # ---------- 파라미터 ----------
        self.declare_parameter("device_index",  -1)
        self.declare_parameter("sample_rate",   16000)
        self.declare_parameter("chunk_size",    512)
        self.declare_parameter("n_channels",    6)
        self.declare_parameter("audio_channel", 0)   # ch0 = processed audio

        self.sample_rate   = self.get_parameter("sample_rate").value
        self.chunk_size    = self.get_parameter("chunk_size").value
        self.n_channels    = self.get_parameter("n_channels").value
        self.audio_channel = self.get_parameter("audio_channel").value
        device_index       = self.get_parameter("device_index").value

        # ---------- pyaudio 초기화 ----------
        self.pa = pyaudio.PyAudio()

        if device_index == -1:
            device_index = self._find_respeaker()

        self.get_logger().info(
            f"MicNode | device_index={device_index} | "
            f"sr={self.sample_rate} | chunk={self.chunk_size} | "
            f"ch={self.audio_channel}/{self.n_channels}"
        )

        self.stream = self.pa.open(
            rate=self.sample_rate,
            channels=self.n_channels,
            format=pyaudio.paInt16,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
        )

        # ---------- Publisher ----------
        self.pub = self.create_publisher(Float32MultiArray, "/audio_raw", 10)

        # ---------- Timer (chunk 단위로 읽기) ----------
        interval = self.chunk_size / self.sample_rate
        self.create_timer(interval, self._read_callback)

    # ------------------------------------------------------------------

    def _read_callback(self):
        try:
            raw = self.stream.read(self.chunk_size, exception_on_overflow=False)
        except Exception as e:
            self.get_logger().warn(f"read error: {e}")
            return

        # int16 interleaved → float32, ch0만 추출
        pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, self.n_channels)
        ch0 = pcm[:, self.audio_channel].astype(np.float32) / 32768.0

        msg      = Float32MultiArray()
        msg.data = ch0.tolist()
        self.pub.publish(msg)

    def _find_respeaker(self) -> int:
        """ReSpeaker 장치 자동 탐색"""
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if "ReSpeaker" in info["name"] and info["maxInputChannels"] >= self.n_channels:
                self.get_logger().info(f"ReSpeaker 발견: index={i} name={info['name']}")
                return i

        self.get_logger().warn("ReSpeaker를 찾지 못했습니다. 기본 장치(0) 사용")
        return 0

    def destroy_node(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        super().destroy_node()


# ------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = MicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
