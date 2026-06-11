#!/usr/bin/env python3
"""
odas_bridge_node.py

ODAS tracked sources (TCP port 9000) → ROS2 토픽 발행

Topic I/O:
    Publish : /sound_direction  (std_msgs/Float32)         [0~360도]
    Publish : /sound_activity   (std_msgs/Float32)         [0.0~1.0]
    Publish : /sound_position   (geometry_msgs/Point)      [x, y, z=0 (m)]

Parameters:
    odas_host       (str)   : ODAS TCP host (default: 127.0.0.1)
    odas_port       (int)   : ODAS tracked port (default: 9000)
    activity_min    (float) : 최소 activity 임계값 (default: 0.5)
    source_height   (float) : 소리 발생 높이 고정값 (default: 1.5m)
    mic_height      (float) : ReSpeaker 설치 높이 (default: 0.3m)
"""

import sys
import json
import math
import socket
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Point


class OdasBridgeNode(Node):
    def __init__(self):
        super().__init__("odas_bridge_node")

        # ---------- 파라미터 ----------
        self.declare_parameter("odas_host",     "127.0.0.1")
        self.declare_parameter("odas_port",     9000)
        self.declare_parameter("activity_min",  0.5)
        self.declare_parameter("source_height", 1.5)   # 소리 발생 높이 (m)
        self.declare_parameter("mic_height",    0.3)   # ReSpeaker 설치 높이 (m)

        self.odas_host     = self.get_parameter("odas_host").value
        self.odas_port     = self.get_parameter("odas_port").value
        self.activity_min  = self.get_parameter("activity_min").value
        self.source_height = self.get_parameter("source_height").value
        self.mic_height    = self.get_parameter("mic_height").value

        # 마이크 기준 소리 높이차
        self.h = self.source_height - self.mic_height  # 예: 1.2m

        # ---------- Publisher ----------
        self.pub_dir  = self.create_publisher(Float32, "/sound_direction", 10)
        self.pub_act  = self.create_publisher(Float32, "/sound_activity",  10)
        self.pub_pos  = self.create_publisher(Point,   "/sound_position",  10)

        # ---------- TCP 서버 스레드 ----------
        self._running = True
        self._thread  = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f"OdasBridgeNode | port={self.odas_port} | "
            f"activity_min={self.activity_min} | "
            f"source_height={self.source_height}m | mic_height={self.mic_height}m"
        )

    # ------------------------------------------------------------------

    def _recv_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.odas_host, self.odas_port))
        server.listen(1)
        server.settimeout(1.0)
        self.get_logger().info(f"port {self.odas_port} 대기 중...")

        while self._running:
            try:
                conn, addr = server.accept()
                self.get_logger().info(f"ODAS 연결됨: {addr}")
                buf = ""
                while self._running:
                    data = conn.recv(4096).decode("utf-8", errors="ignore")
                    if not data:
                        break
                    buf += data
                    while True:
                        try:
                            obj, idx = json.JSONDecoder().raw_decode(buf)
                            buf = buf[idx:].lstrip()
                            self._process(obj)
                        except json.JSONDecodeError:
                            break
                conn.close()
                self.get_logger().warn("ODAS 연결 끊김")
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    self.get_logger().warn(f"에러: {e}")

        server.close()

    # ------------------------------------------------------------------

    def _process(self, obj: dict):
        sources = obj.get("src", [])

        # activity 가장 높은 소스 선택
        best = None
        for src in sources:
            activity = src.get("activity", 0.0)
            if activity < self.activity_min:
                continue
            if best is None or activity > best["activity"]:
                best = src

        if best is None:
            return

        x_u = best["x"]
        y_u = best["y"]
        z_u = best["z"]

        # ── 방향 (azimuth) ───────────────────────────────────────
        azimuth_rad = math.atan2(y_u, x_u)
        azimuth_deg = math.degrees(azimuth_rad) % 360.0

        dir_msg      = Float32()
        dir_msg.data = float(azimuth_deg)
        self.pub_dir.publish(dir_msg)

        act_msg      = Float32()
        act_msg.data = float(best["activity"])
        self.pub_act.publish(act_msg)

        # ── 위치 추정 (x, y) ─────────────────────────────────────
        # 단위벡터의 z 성분으로 elevation 추정
        # horizontal_u = sqrt(x_u² + y_u²) = cos(elevation)
        # z_u = sin(elevation)
        # d(수평거리) = h * cos(elevation) / sin(elevation)
        #             = h * horizontal_u / z_u
        horizontal_u = math.sqrt(x_u ** 2 + y_u ** 2)

        pos_msg = Point()
        if z_u > 0.05:  # elevation이 너무 작으면 거리 계산 불안정
            d = self.h * horizontal_u / z_u
            pos_msg.x = float(d * x_u / horizontal_u)
            pos_msg.y = float(d * y_u / horizontal_u)
            pos_msg.z = 0.0

            self.get_logger().info(
                f"id={best['id']} | "
                f"azimuth={azimuth_deg:.1f}° | "
                f"d={d:.2f}m | "
                f"pos=({pos_msg.x:.2f}, {pos_msg.y:.2f}) | "
                f"activity={best['activity']:.2f}"
            )
        else:
            # elevation 너무 낮으면 방향만 발행
            pos_msg.x = float("nan")
            pos_msg.y = float("nan")
            pos_msg.z = 0.0
            self.get_logger().debug(
                f"id={best['id']} | azimuth={azimuth_deg:.1f}° | "
                f"elevation too low for distance estimation"
            )

        self.pub_pos.publish(pos_msg)

    def destroy_node(self):
        self._running = False
        super().destroy_node()


# ------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = OdasBridgeNode()
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