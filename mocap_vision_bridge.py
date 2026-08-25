#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class MocapVisionBridge(Node):
    def __init__(self):
        super().__init__("mocap_vision_bridge")

        self.declare_parameter("input_topic", "/vrpn_mocap/xy1/pose")
        self.declare_parameter("output_topic", "/mavros/vision_pose/pose")
        self.declare_parameter("output_frame_id", "map")
        self.declare_parameter("timeout_s", 0.2)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.output_frame_id = self.get_parameter("output_frame_id").value
        self.timeout_s = float(self.get_parameter("timeout_s").value)

        self.publisher = self.create_publisher(
            PoseStamped, self.output_topic, qos_profile_sensor_data
        )
        self.subscription = self.create_subscription(
            PoseStamped, self.input_topic, self.pose_callback, qos_profile_sensor_data
        )
        self.last_pose_time = None
        self.stream_timed_out = False
        self.received_count = 0
        self.create_timer(0.1, self.check_timeout)

        self.get_logger().info(
            f"Relaying {self.input_topic} -> {self.output_topic}"
        )

    def pose_callback(self, msg: PoseStamped):
        values = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        if not all(math.isfinite(value) for value in values):
            self.get_logger().error("Rejected non-finite mocap pose")
            return

        quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
        if quaternion_norm < 1e-6:
            self.get_logger().error("Rejected mocap pose with invalid quaternion")
            return

        output = PoseStamped()
        output.header = msg.header
        output.header.frame_id = self.output_frame_id
        output.pose = msg.pose
        self.publisher.publish(output)

        self.last_pose_time = self.get_clock().now()
        self.stream_timed_out = False
        self.received_count += 1
        if self.received_count == 1:
            self.get_logger().info("Received and forwarded the first valid mocap pose")

    def check_timeout(self):
        if self.last_pose_time is None:
            return

        elapsed = (self.get_clock().now() - self.last_pose_time).nanoseconds / 1e9
        if elapsed > self.timeout_s and not self.stream_timed_out:
            self.stream_timed_out = True
            self.get_logger().error(
                f"Mocap stream timed out after {elapsed:.3f} s; no stale pose is published"
            )


def main():
    rclpy.init()
    node = MocapVisionBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
