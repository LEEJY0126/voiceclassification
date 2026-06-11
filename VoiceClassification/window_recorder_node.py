#!/usr/bin/env python3
"""
window_recorder_node.py

/voice_window 토픽을 구독해 window마다 wav 파일로 저장.
실시간 버퍼 출력 디버깅용.

Topic I/O:
    Subscribe : /voice_window (std_msgs/Float32MultiArray)

Parameters:
    save_dir    (str) : 저장 폴더 (default: ~/voice_windows)
    max_files   (int) : 최대 저장 파일 수, 초과 시 오래된 것 삭제 (default: 50)
    sample_rate (int) : 샘플레이트 (default: 16000)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class WindowRecorderNode(Node):
    def __init__(self):
        super().__init__("window_recorder_node")

        # ---------- 파라미터 ----------
        self.declare_parameter("save_dir",    str(Path.home() / "voice_windows"))
        self.declare_parameter("max_files",   50)
        self.declare_parameter("sample_rate", 16000)

        self.save_dir    = Path(self.get_parameter("save_dir").value)
        self.max_files   = self.get_parameter("max_files").value
        self.sample_rate = self.get_parameter("sample_rate").value
        self.count       = 0

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            f"WindowRecorder | 저장 위치: {self.save_dir} | "
            f"max_files={self.max_files}"
        )

        # ---------- Subscriber ----------
        self.create_subscription(
            Float32MultiArray,
            "/voice_window",
            self._callback,
            qos_profile=10,
        )

    # ------------------------------------------------------------------

    def _callback(self, msg: Float32MultiArray):
        window = np.array(msg.data, dtype=np.float32)

        # 파일명: window_0001_20250606_153022.wav
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename  = self.save_dir / f"window_{self.count:04d}_{timestamp}.wav"

        sf.write(str(filename), window, self.sample_rate)
        self.count += 1

        self.get_logger().info(
            f"[{self.count:04d}] 저장: {filename.name} "
            f"| max={window.max():.3f} min={window.min():.3f}"
        )

        # max_files 초과 시 오래된 파일 삭제
        self._cleanup()

    def _cleanup(self):
        files = sorted(self.save_dir.glob("window_*.wav"))
        if len(files) > self.max_files:
            for f in files[:len(files) - self.max_files]:
                f.unlink()
                self.get_logger().debug(f"삭제: {f.name}")


# ------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = WindowRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()