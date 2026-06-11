#!/usr/bin/env python3
"""
inference_onnx_node.py

ONNX 모델로 /voice_window 추론 후 /voice_command 발행.
PyTorch 불필요 → 라즈베리파이 배포용.

Topic I/O:
    Subscribe : /voice_window     (std_msgs/Float32MultiArray)  [16000 samples]
    Publish   : /voice_command    (std_msgs/Int32)              [0 / 1 / 2 / 3]
    Publish   : /voice_confidence (std_msgs/Float32)            [0.0 ~ 1.0]

Parameters:
    model_path  (str)   : voice_classifier.onnx 경로
    threshold   (float) : 이 confidence 미만이면 class 0으로 처리 (default: 0.7)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32, Float32

from VoiceClassification.model.inferencer_onnx import InferencerOnnx, LABEL_NAMES


class InferenceOnnxNode(Node):
    def __init__(self):
        super().__init__("inference_onnx_node")

        # ---------- 파라미터 ----------
        self.declare_parameter("model_path", "checkpoints/voice_classifier.onnx")
        self.declare_parameter("threshold",  0.7)

        model_path = self.get_parameter("model_path").value
        threshold  = self.get_parameter("threshold").value

        self.threshold = threshold

        # ---------- 모델 로드 ----------
        try:
            self.inferencer = InferencerOnnx(model_path)
        except Exception as e:
            self.get_logger().error(f"모델 로드 실패: {e}")
            raise

        self.get_logger().info(
            f"InferenceOnnxNode 준비 | threshold={threshold}"
        )

        # ---------- Subscriber ----------
        self.sub = self.create_subscription(
            Float32MultiArray,
            "/voice_window",
            self._window_callback,
            qos_profile=10,
        )

        # ---------- Publisher ----------
        self.pub_cmd  = self.create_publisher(Int32,   "/voice_command",   10)
        self.pub_conf = self.create_publisher(Float32, "/voice_confidence", 10)

    # ------------------------------------------------------------------

    def _window_callback(self, msg: Float32MultiArray):
        window = np.array(msg.data, dtype=np.float32)

        label, confidence = self.inferencer.predict(window)

        # threshold 미만이면 기타(0)로 처리
        if confidence < self.threshold:
            label = 0

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
    node = InferenceOnnxNode()
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