#!/usr/bin/env python3
"""
respeaker_doa_node.py

ReSpeaker 내장 DOA를 USB로 읽어 /sound_direction 발행.
ODAS 없이 동작 — ALSA 충돌 없음.

Topic I/O:
    Publish : /sound_direction (std_msgs/Float32)  [0~359도]

Parameters:
    rate_hz       (float) : DOA 읽기 주기 (default: 10Hz)
    tuning_path   (str)   : tuning.py 위치
"""

import sys
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class RespeakerDoaNode(Node):
    def __init__(self):
        super().__init__("respeaker_doa_node")

        # ---------- 파라미터 ----------
        self.declare_parameter("rate_hz",     10.0)
        self.declare_parameter("tuning_path", str(Path.home() / "workspace/usb_4_mic_array"))

        rate_hz      = self.get_parameter("rate_hz").value
        tuning_path  = self.get_parameter("tuning_path").value

        # ---------- tuning.py 동적 로드 ----------
        try:
            spec   = importlib.util.spec_from_file_location(
                "tuning", str(Path(tuning_path) / "tuning.py")
            )
            tuning_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tuning_mod)
        except Exception as e:
            self.get_logger().error(f"tuning.py 로드 실패: {e}")
            raise

        # ---------- ReSpeaker USB 연결 ----------
        try:
            import usb.core
            dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            if dev is None:
                raise RuntimeError("ReSpeaker를 찾을 수 없습니다. USB 연결 확인하세요.")
            self.tuning = tuning_mod.Tuning(dev)
            self.get_logger().info("ReSpeaker DOA 연결 완료")
        except Exception as e:
            self.get_logger().error(f"ReSpeaker 연결 실패: {e}")
            raise

        # ---------- Publisher ----------
        self.pub = self.create_publisher(Float32, "/sound_direction", 10)

        # ---------- Timer ----------
        self.create_timer(1.0 / rate_hz, self._timer_callback)
        self.get_logger().info(f"RespeakerDoaNode | rate={rate_hz}Hz")

    # ------------------------------------------------------------------

    def _timer_callback(self):
        try:
            direction = float(self.tuning.direction)
            msg       = Float32()
            msg.data  = direction
            self.pub.publish(msg)
            self.get_logger().debug(f"DOA: {direction:.0f}도")
        except Exception as e:
            self.get_logger().warn(f"DOA 읽기 실패: {e}")


# ------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = RespeakerDoaNode()
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