#!/usr/bin/env python3
"""
odas_audio_bridge_node.py

ODAS sss.separated (TCP port 9002) → /audio_raw 토픽 발행
mic_node 대체 — ReSpeaker를 ODAS 하나만 점유

ODAS separated 출력 형식:
    raw PCM, int16, 16kHz, hopSize=128 샘플/패킷, n_sources 채널 interleaved

Topic I/O:
    Publish : /audio_raw (std_msgs/Float32MultiArray) [float32, -1~1]

Parameters:
    host       (str) : 바인드 주소 (default: 127.0.0.1)
    port       (int) : ODAS separated 포트 (default: 9002)
    n_sources  (int) : ODAS ssl.nPots 값 (default: 4)
"""

import sys
import socket
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class OdasAudioBridgeNode(Node):
    def __init__(self):
        super().__init__("odas_audio_bridge_node")

        # ---------- 파라미터 ----------
        self.declare_parameter("host",      "127.0.0.1")
        self.declare_parameter("port",      9002)
        self.declare_parameter("n_sources", 4)   # odas.cfg ssl.nPots 와 동일

        self.host      = self.get_parameter("host").value
        self.port      = self.get_parameter("port").value
        self.n_sources = self.get_parameter("n_sources").value

        # n_sources 채널 interleaved, hopSize=128, int16
        self.hop_size  = 128
        self.byte_size = self.hop_size * self.n_sources * 2

        # ---------- Publisher ----------
        self.pub = self.create_publisher(Float32MultiArray, "/audio_raw", 10)

        # ---------- TCP 서버 스레드 ----------
        self._running = True
        self._thread  = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f"OdasAudioBridgeNode | port={self.port} | "
            f"n_sources={self.n_sources} | byte_size={self.byte_size}"
        )

    # ------------------------------------------------------------------

    def _recv_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(1.0)
        self.get_logger().info(f"port {self.port} 대기 중...")

        while self._running:
            try:
                conn, addr = server.accept()
                self.get_logger().info(f"ODAS audio 연결됨: {addr}")

                buf = b""
                while self._running:
                    data = conn.recv(4096)
                    if not data:
                        break
                    buf += data

                    while len(buf) >= self.byte_size:
                        chunk_bytes = buf[:self.byte_size]
                        buf         = buf[self.byte_size:]

                        # interleaved int16 → source 0만 추출
                        pcm   = np.frombuffer(chunk_bytes, dtype=np.int16)
                        pcm   = pcm.reshape(-1, self.n_sources)          # [128, n_sources]
                        audio = pcm[:, 0].astype(np.float32) / 32768.0  # source 0

                        msg      = Float32MultiArray()
                        msg.data = [float(x) for x in audio]
                        self.pub.publish(msg)

                conn.close()
                self.get_logger().warn("ODAS audio 연결 끊김")

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    self.get_logger().warn(f"에러: {e}")

        server.close()

    def destroy_node(self):
        self._running = False
        super().destroy_node()


# ------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = OdasAudioBridgeNode()
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