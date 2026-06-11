#!/usr/bin/env python3
"""
inference_node.py

/voice_window 토픽을 구독해 추론 후 /voice_command 발행.

Topic I/O:
    Subscribe : /voice_window    (std_msgs/Float32MultiArray)  [16000 samples]
    Publish   : /voice_command   (std_msgs/Int32)              [0 / 1 / 2]
    Publish   : /voice_confidence(std_msgs/Float32)            [0.0 ~ 1.0]

Parameters:
    checkpoint_path (str)   : best.pt 경로
    threshold       (float) : 이 confidence 미만이면 class 0으로 처리 (default: 0.7)
    device          (str)   : 'cpu' / 'cuda' (default: 'cpu')
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32, Float32

from VoiceClassification.model.inferencer import Inferencer, LABEL_NAMES


class InferenceNode(Node):
    def __init__(self):
        super().__init__("inference_node")

        # ---------- 파라미터 ----------
        self.declare_parameter("checkpoint_path", "checkpoints/best.pt")
        self.declare_parameter("threshold",        0.7)
        self.declare_parameter("device",           "cpu")

        checkpoint = self.get_parameter("checkpoint_path").value
        threshold  = self.get_parameter("threshold").value
        device     = self.get_parameter("device").value

        self.threshold = threshold

        # ---------- 모델 로드 ----------
        try:
            self.inferencer = Inferencer(checkpoint, device=device)
        except Exception as e:
            self.get_logger().error(f"모델 로드 실패: {e}")
            raise

        self.get_logger().info(
            f"InferenceNode 준비 | threshold={threshold} | device={device}"
        )

        # ---------- Subscriber ----------
        self.sub = self.create_subscription(
            Float32MultiArray,
            "/voice_window",
            self._window_callback,
            qos_profile=10,
        )

        # ---------- Publisher ----------
        self.pub_cmd  = self.create_publisher(Int32,   "/voice_command",    10)
        self.pub_conf = self.create_publisher(Float32, "/voice_confidence",  10)

    # ------------------------------------------------------------------

    def _window_callback(self, msg: Float32MultiArray):
        window = np.array(msg.data, dtype=np.float32)   # [16000]

        label, confidence = self.inferencer.predict(window)

        # threshold 미만이면 기타(0)로 처리
        if confidence < self.threshold:
            label = 0

        # 발행
        cmd_msg       = Int32()
        cmd_msg.data  = label
        conf_msg      = Float32()
        conf_msg.data = confidence

        self.pub_cmd.publish(cmd_msg)
        self.pub_conf.publish(conf_msg)

        self.get_logger().info(
            f"[{label}] {LABEL_NAMES[label]}  confidence={confidence:.1%}"
        )


# ------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()