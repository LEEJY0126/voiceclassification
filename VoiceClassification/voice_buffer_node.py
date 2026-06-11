#!/usr/bin/env python3
"""
voice_buffer_node.py

마이크 → /audio_raw 토픽 수신 → VoiceBuffer → /voice_window 토픽 발행

Topic I/O:
    Subscribe : /audio_raw      (std_msgs/Float32MultiArray)  [-1.0 ~ 1.0 PCM]
    Publish   : /voice_window   (std_msgs/Float32MultiArray)  [16000 samples, 1초]
    Publish   : /buffer_status  (std_msgs/Float32)            [fill ratio 0.0~1.0]
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32

from VoiceClassification.voicebuffer import VoiceBuffer, BufferConfig


class VoiceBufferNode(Node):
    def __init__(self):
        super().__init__("voice_buffer_node")

        # ---------- 파라미터 선언 (ros2 run 시 --ros-args -p로 변경 가능) ----------
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("window_sec", 1.0)
        self.declare_parameter("hop_sec", 0.5)

        cfg = BufferConfig(
            sample_rate=self.get_parameter("sample_rate").value,
            window_sec=self.get_parameter("window_sec").value,
            hop_sec=self.get_parameter("hop_sec").value,
        )
        self.buffer = VoiceBuffer(cfg)
        self.get_logger().info(
            f"VoiceBuffer ready | "
            f"sr={cfg.sample_rate} window={cfg.window_sec}s hop={cfg.hop_sec}s "
            f"({cfg.window_size} / {cfg.hop_size} samples)"
        )

        # ---------- Subscriber ----------
        self.sub_audio = self.create_subscription(
            Float32MultiArray,
            "/audio_raw",
            self._audio_callback,
            qos_profile=10,
        )

        # ---------- Publisher ----------
        self.pub_window = self.create_publisher(
            Float32MultiArray,
            "/voice_window",
            qos_profile=10,
        )
        self.pub_status = self.create_publisher(
            Float32,
            "/buffer_status",
            qos_profile=10,
        )

    # ------------------------------------------------------------------

    def _audio_callback(self, msg: Float32MultiArray):
        """
        /audio_raw 수신 → 버퍼에 push → window 완성 시 /voice_window 발행
        """
        chunk = msg.data  # list[float]

        windows = self.buffer.push(list(chunk))

        for window in windows:
            out = Float32MultiArray()
            out.data = window.tolist()
            self.pub_window.publish(out)
            self.get_logger().debug("Published 1s voice window")

        # fill ratio 발행 (디버깅/모니터링용)
        status = Float32()
        status.data = self.buffer.fill_ratio
        self.pub_status.publish(status)


# ------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = VoiceBufferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()