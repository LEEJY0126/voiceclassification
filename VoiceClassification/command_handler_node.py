#!/usr/bin/env python3
"""
command_handler_node.py

voice_command + DOA → /goal_pose (PoseStamped) 변환.

Topic I/O:
    Subscribe : /voice_command    (std_msgs/Int32)           [0~3]
    Subscribe : /sound_direction  (std_msgs/Float32)         [0~359도]
    Subscribe : /odom             (nav_msgs/Odometry)        [현재 위치]
    Publish   : /goal_pose        (geometry_msgs/PoseStamped)

Command 동작:
    0 (기타)  → 무시
    1 (일로와) → DOA 방향으로 goal_distance 앞에 goal 발행
    2 (가)    → 현재 진행 방향으로 goal_distance 앞에 goal 발행
    3 (멈춰)  → 현재 위치를 goal로 발행 (제자리 정지)

Parameters:
    goal_distance  (float) : 목표 거리 (default: 1.0m)
    command_cooldown (float) : 같은 명령 재발행 쿨다운 (default: 2.0s)
"""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Int32, Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import time


LABEL_NAMES = {0: "기타", 1: "일로와", 2: "가", 3: "멈춰"}


class CommandHandlerNode(Node):
    def __init__(self):
        super().__init__("command_handler_node")

        # ---------- 파라미터 ----------
        self.declare_parameter("goal_distance",    1.0)
        self.declare_parameter("command_cooldown", 2.0)

        self.goal_distance    = self.get_parameter("goal_distance").value
        self.command_cooldown = self.get_parameter("command_cooldown").value

        # ---------- 상태 ----------
        self.latest_doa    : float       = 0.0   # 가장 최근 DOA (도)
        self.curr_x        : float | None = None
        self.curr_y        : float | None = None
        self.curr_heading  : float       = 0.0   # 로봇 heading (라디안)
        self.last_cmd_time : float       = 0.0

        # ---------- QoS ----------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ---------- Subscriber ----------
        self.create_subscription(Int32,       "/voice_command",   self._cmd_callback, 10)
        self.create_subscription(Float32,     "/sound_direction", self._doa_callback, 10)
        self.create_subscription(Odometry,    "/odom",            self._odom_callback, qos)

        # ---------- Publisher ----------
        self.pub_goal = self.create_publisher(PoseStamped, "/goal_pose", 10)

        self.get_logger().info(
            f"CommandHandlerNode | "
            f"goal_distance={self.goal_distance}m | "
            f"cooldown={self.command_cooldown}s"
        )

    # ------------------------------------------------------------------

    def _doa_callback(self, msg: Float32):
        self.latest_doa = msg.data

    def _odom_callback(self, msg: Odometry):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y

        # quaternion → yaw (heading)
        q  = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.curr_heading = math.atan2(siny_cosp, cosy_cosp)

    def _cmd_callback(self, msg: Int32):
        command = msg.data

        if command == 0:
            return

        # 쿨다운 체크
        now = time.time()
        if now - self.last_cmd_time < self.command_cooldown:
            return
        self.last_cmd_time = now

        if self.curr_x is None:
            self.get_logger().warn("odom 아직 없음, goal 발행 불가")
            return

        self.get_logger().info(
            f"명령 수신: [{command}] {LABEL_NAMES.get(command, '?')} | "
            f"DOA={self.latest_doa:.1f}° | "
            f"pos=({self.curr_x:.2f}, {self.curr_y:.2f}) | "
            f"heading={math.degrees(self.curr_heading):.1f}°"
        )

        if command == 1:    # 일로와 → DOA 방향으로 이동
            self._publish_goal_by_doa()

        elif command == 2:  # 가 → DOA 반대 방향으로 전진
            self._publish_goal_forward()

        elif command == 3:  # 멈춰 → 현재 위치 goal (정지)
            self._publish_goal_stop()

    # ------------------------------------------------------------------

    def _publish_goal_by_doa(self):
        """
        DOA 방향으로 goal_distance 앞에 goal 발행.

        ReSpeaker DOA 0° = 로봇 정면
        world frame 변환:
            world_angle = robot_heading + DOA_rad
            goal_x = curr_x + d * cos(world_angle)
            goal_y = curr_y + d * sin(world_angle)
        """
        doa_rad     = math.radians(self.latest_doa)
        world_angle = self.curr_heading + doa_rad

        goal_x = self.curr_x + self.goal_distance * math.cos(world_angle)
        goal_y = self.curr_y + self.goal_distance * math.sin(world_angle)

        self._publish_goal(goal_x, goal_y)
        self.get_logger().info(
            f"[일로와] DOA={self.latest_doa:.1f}° → "
            f"goal=({goal_x:.2f}, {goal_y:.2f})"
        )

    def _publish_goal_forward(self):
        """DOA 반대 방향으로 goal_distance 앞에 goal 발행"""
        doa_rad     = math.radians(self.latest_doa)
        world_angle = self.curr_heading + doa_rad + math.pi  # ← +180도

        goal_x = self.curr_x + self.goal_distance * math.cos(world_angle)
        goal_y = self.curr_y + self.goal_distance * math.sin(world_angle)

        self._publish_goal(goal_x, goal_y)
        self.get_logger().info(
            f"[가] DOA={self.latest_doa:.1f}° 반대 방향 → "
            f"goal=({goal_x:.2f}, {goal_y:.2f})"
        )

    def _publish_goal_stop(self):
        """현재 위치를 goal로 발행 → RL 노드가 제자리에서 멈춤"""
        self._publish_goal(self.curr_x, self.curr_y)
        self.get_logger().info(
            f"[멈춰] 현재 위치 goal=({self.curr_x:.2f}, {self.curr_y:.2f})"
        )

    def _publish_goal(self, x: float, y: float):
        msg                   = PoseStamped()
        msg.header.stamp      = self.get_clock().now().to_msg()
        msg.header.frame_id   = "odom"
        msg.pose.position.x   = x
        msg.pose.position.y   = y
        msg.pose.position.z   = 0.0
        msg.pose.orientation.w = 1.0  # 방향 무관 (RL이 알아서 회전)
        self.pub_goal.publish(msg)


# ------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = CommandHandlerNode()
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