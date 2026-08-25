#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class OffboardPolylineFollower(Node):
    def __init__(self):
        super().__init__("offboard_polyline_follower_py_ros2")

        # ============================================================
        # User parameters (hard-coded)
        # ============================================================

        self.rate_hz = 20.0
        self.dt = 1.0 / self.rate_hz

        # RViz markers + setpoints frame
        self.frame_id = "map"

        # ---- Added (Scheme A): publish static TF so "map" exists in RViz ----
        # RViz Fixed Frame can remain "map" after this.
        self.world_frame = "world"
        self.map_frame = "map"

        # Pose source (mocap/vision)
        # self.pose_topic = "/mavros/vision_pose/pose"
        self.pose_topic = "/mavros/local_position/pose"
        self.pose_timeout_s = 0.2

        # Waypoints [x, y, z, yaw_deg]
        self.waypoints = [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 45.0],
            [1.0, 1.0, 1.0, 0.0],
        ]

        # Virtual gate parameters (kept, not critical for landing)
        self.gate_diameter = 1.0
        self.gate_pre_distance = 0.1
        self.gate_post_distance = 0.1
        self.gate_circle_points = 64
        self.gate_axis_length = 0.8

        # Path-following parameters
        self.lookahead_distance = 0.5
        self.max_track_error = 1.5
        self.recover_track_error = 0.8
        self.takeoff_accept_radius = 0.3
        self.final_accept_radius = 0.4  # kept but NOT used for finish trigger anymore
        self.trajectory_max_len = 200
        self.search_back_segments = 1
        self.search_forward_segments = 2

        # ------------------------------------------------------------
        # Sequential pass gating (must pass each point in order)
        # ------------------------------------------------------------
        self.pass_radius = 0.5            # <==== pass radius you requested
        self.use_3d_pass_check = True     # True: XYZ distance; False: XY only

        # ------------------------------------------------------------
        # Finish -> OFFBOARD vertical descent -> AUTO.LAND -> Disarm
        # ------------------------------------------------------------
        self.finish_hold_s = 1.0

        # OFFBOARD descent rate (m/s)
        self.land_descent_rate = 0.30

        # Do not command z below this during OFFBOARD descent
        self.land_min_z_cmd = 0.05

        # Switch to AUTO.LAND when measured z below this (m)
        # (set to 0.25~0.35 typically)
        self.land_switch_to_auto_land_z = 0.25

        # AUTO.LAND mode name
        self.land_mode = "AUTO.LAND"

        # Landed confirm in AUTO.LAND (measured z)
        self.land_z_threshold = 0.15
        self.land_confirm_s = 2.0

        # After disarm request succeeds, wait 1s and ensure armed == False
        self.post_disarm_check_s = 1.0

        # Console print interval
        self.print_interval_s = 1.0

        # Service request interval
        self.request_interval = 1.0

        # RViz markers
        self.marker_lifetime = 0.5

        # PX4 offboard needs setpoint stream before OFFBOARD
        self.prestream_count = 100

        # ============================================================
        # Internal state
        # ============================================================

        self.current_state = State()

        self.has_pose = False
        self.current_position = None
        self.current_orientation = None

        self.last_pose_time = None
        self.pose_timed_out = False

        self.mission_state = "WAIT_FCU"   # WAIT_FCU -> WAIT_POSE -> PRESTREAM -> RUN

        # TAKEOFF -> FOLLOW -> FINISH_HOLD -> LAND_OFFBOARD -> LAND_REQUEST -> LANDING_CONFIRM -> DISARM_REQUEST -> DONE
        self.flight_phase = "TAKEOFF"

        self.recover_mode = False
        self.current_segment_idx = 0
        self.last_s_progress = 0.0

        self.projection_point = None
        self.lookahead_point = None
        self.target_point = None
        self.track_error = 0.0

        self.history_points = []

        # Sequential pass state: next waypoint index to pass (in expanded wp list)
        self.pass_idx = 1

        # Landing state
        self.land_xy_hold = None      # np.array([x, y]) locked once
        self.land_z_cmd = None        # float for OFFBOARD descent
        self._finish_enter_time = None
        self._land_low_enter_time = None
        self._disarm_success_time = None

        self._land_mode_sent = False

        # request throttling + prestream
        self._last_req_time = self.get_clock().now()
        self._prestream_sent = 0

        # logging throttle
        self._last_print_time = self.get_clock().now()

        # ============================================================
        # Path preprocessing
        # ============================================================

        self.original_wp = np.array(self.waypoints, dtype=float)
        if self.original_wp.shape[0] < 2:
            raise RuntimeError("At least two waypoints are required.")
        if self.original_wp.shape[1] != 4:
            raise RuntimeError("Each waypoint must be [x, y, z, yaw_deg].")

        self.build_expanded_path()
        self.wp = np.array(self.expanded_waypoints, dtype=float)

        if self.wp.shape[0] < 2:
            raise RuntimeError("Expanded path must contain at least two points.")

        self.num_waypoints = self.wp.shape[0]
        self.num_segments = self.num_waypoints - 1

        self.segment_lengths = []
        self.cumulative_s = [0.0]
        for i in range(self.num_segments):
            seg = self.wp[i + 1] - self.wp[i]
            length = float(np.linalg.norm(seg))
            if length < 1e-6:
                raise RuntimeError(f"Expanded waypoint {i} and {i+1} are too close/identical.")
            self.segment_lengths.append(length)
            self.cumulative_s.append(self.cumulative_s[-1] + length)

        self.segment_lengths = np.array(self.segment_lengths, dtype=float)
        self.cumulative_s = np.array(self.cumulative_s, dtype=float)
        self.total_length = float(self.cumulative_s[-1])

        # Clamp pass_idx valid
        self.pass_idx = max(1, min(self.pass_idx, self.num_waypoints - 1))

        # ============================================================
        # ROS 2 QoS
        # ============================================================
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # ============================================================
        # ROS 2 pub/sub
        # ============================================================

        self.state_sub = self.create_subscription(State, "/mavros/state", self.state_cb, 10)
        self.pose_sub = self.create_subscription(PoseStamped, self.pose_topic, self.pose_cb, sensor_qos)

        self.local_pos_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/offboard_polyline/markers", 10)

        # ---- Added: Static TF broadcaster (world -> map) ----
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_tf_world_to_map()

        # ============================================================
        # ROS 2 services
        # ============================================================

        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")

        self.get_logger().info("Waiting for /mavros/cmd/arming service...")
        self.arming_client.wait_for_service(timeout_sec=10.0)

        self.get_logger().info("Waiting for /mavros/set_mode service...")
        self.set_mode_client.wait_for_service(timeout_sec=10.0)

        # ============================================================
        # Timer
        # ============================================================

        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info("Initialized.")
        self.get_logger().info(
            f"Sequential pass: pass_radius={self.pass_radius:.2f}m, use_3d_pass_check={self.use_3d_pass_check}. "
            f"OFFBOARD descent rate={self.land_descent_rate:.2f} m/s, "
            f"switch to {self.land_mode} at z<{self.land_switch_to_auto_land_z:.2f} m, "
            f"land confirm z<{self.land_z_threshold:.2f} for {self.land_confirm_s:.2f}s"
        )
        self.get_logger().info(
            f"[TF static] Publishing identity transform: {self.world_frame} -> {self.map_frame}. "
            f"RViz Fixed Frame can stay '{self.map_frame}'."
        )

    # ============================================================
    # Added: publish static TF once
    # ============================================================

    def _publish_static_tf_world_to_map(self):
        """
        Publish a static identity transform world -> map so RViz has a valid 'map' frame
        even when no other /tf or /tf_static is being published.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.map_frame

        # Identity transform
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.static_tf_broadcaster.sendTransform(t)

    # ============================================================
    # Path expansion
    # ============================================================

    def build_expanded_path(self):
        self.expanded_waypoints = []
        self.gates = []

        n = self.original_wp.shape[0]
        self.expanded_waypoints.append(self.original_wp[0, 0:3].copy())

        for i in range(1, n - 1):
            center = self.original_wp[i, 0:3].copy()
            yaw_deg = float(self.original_wp[i, 3])
            yaw_rad = math.radians(yaw_deg)
            forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=float)

            pre_point = center - self.gate_pre_distance * forward
            post_point = center + self.gate_post_distance * forward

            self.expanded_waypoints.append(pre_point)
            self.expanded_waypoints.append(center)
            self.expanded_waypoints.append(post_point)

            self.gates.append({
                "index": i,
                "center": center,
                "yaw_deg": yaw_deg,
                "yaw_rad": yaw_rad,
                "forward": forward,
                "pre_point": pre_point,
                "post_point": post_point,
                "diameter": self.gate_diameter
            })

        self.expanded_waypoints.append(self.original_wp[-1, 0:3].copy())

    # ============================================================
    # Callbacks
    # ============================================================

    def state_cb(self, msg: State):
        self.current_state = msg

    def pose_cb(self, msg: PoseStamped):
        self.current_position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        self.current_orientation = msg.pose.orientation
        self.has_pose = True
        self.last_pose_time = self.get_clock().now()
        self.pose_timed_out = False

        p = Point()
        p.x = float(msg.pose.position.x)
        p.y = float(msg.pose.position.y)
        p.z = float(msg.pose.position.z)
        self.history_points.append(p)
        if len(self.history_points) > self.trajectory_max_len:
            self.history_points = self.history_points[-self.trajectory_max_len:]

    # ============================================================
    # Main loop
    # ============================================================

    def loop(self):
        now = self.get_clock().now()
        initial_point = np.array([0.0, 0.0, 0.0], dtype=float)

        # pose timeout
        if self.last_pose_time is None:
            self.pose_timed_out = True
        else:
            elapsed = (now - self.last_pose_time).nanoseconds / 1e9
            self.pose_timed_out = elapsed > self.pose_timeout_s

        # WAIT stages
        if self.mission_state == "WAIT_FCU":
            self.publish_setpoint(initial_point)
            self.publish_markers()
            if self.current_state.connected:
                self.get_logger().info("FCU connected.")
                self.mission_state = "WAIT_POSE"
            return

        if self.mission_state == "WAIT_POSE":
            self.publish_setpoint(initial_point)
            self.publish_markers()
            if self.has_pose and (not self.pose_timed_out):
                self.get_logger().info("Pose received.")
                self.mission_state = "PRESTREAM"
            return

        if self.mission_state == "PRESTREAM":
            self.publish_setpoint(initial_point)
            self.publish_markers()
            self._prestream_sent += 1
            if self._prestream_sent >= self.prestream_count:
                self.get_logger().info("Prestream done. RUN.")
                self.mission_state = "RUN"
                self._last_req_time = now
            return

        # RUN: keep OFFBOARD+armed during flight and during OFFBOARD landing phase
        if self.flight_phase in ["TAKEOFF", "FOLLOW", "FINISH_HOLD", "LAND_OFFBOARD"]:
            self.request_offboard_and_arm(now)

        # pose missing
        if self.pose_timed_out or (not self.has_pose) or (self.current_position is None):
            target = self.current_position.copy() if self.current_position is not None else initial_point
            self.target_point = target.copy()
            self.publish_setpoint(target)
            self.publish_markers()
            self.throttled_print(now)
            return

        z_meas = float(self.current_position[2])

        if self.flight_phase == "TAKEOFF":
            target = self.wp[0].copy()
            if float(np.linalg.norm(self.current_position - self.wp[0])) < self.takeoff_accept_radius:
                self.flight_phase = "FOLLOW"
                self.last_s_progress = 0.0
                self.current_segment_idx = 0
                self.pass_idx = 1  # reset sequential pass
                self.get_logger().info("Reached first waypoint. FOLLOW.")
            self.target_point = target.copy()
            self.publish_setpoint(target)

        elif self.flight_phase == "FOLLOW":
            target = self.update_guidance()

            # Sequential pass update
            if self.pass_idx < self.num_waypoints:
                wp_next = self.wp[self.pass_idx]
                if self.use_3d_pass_check:
                    d = float(np.linalg.norm(self.current_position - wp_next))
                else:
                    d = float(np.linalg.norm(self.current_position[0:2] - wp_next[0:2]))

                if d < self.pass_radius:
                    self.get_logger().info(
                        f"Passed wp[{self.pass_idx}] (d={d:.2f} < {self.pass_radius:.2f})."
                    )
                    self.pass_idx += 1

            # Finish only after all points passed
            if self.pass_idx >= self.num_waypoints:
                self.flight_phase = "FINISH_HOLD"
                self._finish_enter_time = now
                self.get_logger().info(f"All waypoints passed. Enter FINISH_HOLD {self.finish_hold_s:.2f}s.")
                target = self.wp[-1].copy()

            self.target_point = target.copy()
            self.publish_setpoint(target)

        elif self.flight_phase == "FINISH_HOLD":
            target = self.wp[-1].copy()
            self.target_point = target.copy()
            self.publish_setpoint(target)

            if self._finish_enter_time is None:
                self._finish_enter_time = now
            if (now - self._finish_enter_time).nanoseconds / 1e9 >= self.finish_hold_s:
                self.flight_phase = "LAND_OFFBOARD"
                self.land_xy_hold = self.current_position[0:2].copy()
                self.land_z_cmd = float(self.current_position[2])
                self._land_low_enter_time = None
                self.get_logger().info(
                    f"Enter LAND_OFFBOARD: lock XY=({self.land_xy_hold[0]:.2f},{self.land_xy_hold[1]:.2f}), "
                    f"start z_cmd={self.land_z_cmd:.2f}"
                )

        elif self.flight_phase == "LAND_OFFBOARD":
            # OFFBOARD vertical descent until close to ground, then switch to AUTO.LAND
            if z_meas <= self.land_switch_to_auto_land_z:
                self.flight_phase = "LAND_REQUEST"
                self._land_mode_sent = False
                self._last_req_time = now - Duration(seconds=self.request_interval)  # immediate
                self.get_logger().info(
                    f"z_meas={z_meas:.2f} <= {self.land_switch_to_auto_land_z:.2f}. Switch to {self.land_mode}."
                )
            else:
                dz = self.land_descent_rate / self.rate_hz
                self.land_z_cmd = max(self.land_min_z_cmd, float(self.land_z_cmd) - float(dz))

                sp = np.array([self.land_xy_hold[0], self.land_xy_hold[1], self.land_z_cmd], dtype=float)
                self.target_point = sp.copy()
                self.publish_setpoint(sp)

        elif self.flight_phase == "LAND_REQUEST":
            # Request AUTO.LAND; still publish "hold XY" setpoint to avoid lateral pull.
            self.request_land_mode(now)
            sp = self.make_xy_hold_setpoint()
            self.target_point = sp.copy()
            self.publish_setpoint(sp)

            if self.current_state.mode == self.land_mode:
                self.flight_phase = "LANDING_CONFIRM"
                self._land_low_enter_time = None
                self.get_logger().info(f"{self.land_mode} active. LANDING_CONFIRM.")

        elif self.flight_phase == "LANDING_CONFIRM":
            # Keep publishing XY-hold setpoint; PX4 handles descent in AUTO.LAND
            sp = self.make_xy_hold_setpoint()
            self.target_point = sp.copy()
            self.publish_setpoint(sp)

            if z_meas < self.land_z_threshold:
                if self._land_low_enter_time is None:
                    self._land_low_enter_time = now
                low_elapsed = (now - self._land_low_enter_time).nanoseconds / 1e9
                if low_elapsed >= self.land_confirm_s:
                    self.flight_phase = "DISARM_REQUEST"
                    self._disarm_success_time = None
                    self._last_req_time = now - Duration(seconds=self.request_interval)
                    self.get_logger().info("Landed confirmed. DISARM_REQUEST.")
            else:
                self._land_low_enter_time = None

        elif self.flight_phase == "DISARM_REQUEST":
            sp = self.make_xy_hold_setpoint()
            self.target_point = sp.copy()
            self.publish_setpoint(sp)

            self.request_disarm(now)

            if self._disarm_success_time is not None:
                elapsed = (now - self._disarm_success_time).nanoseconds / 1e9
                if elapsed >= self.post_disarm_check_s:
                    if not self.current_state.armed:
                        self.get_logger().info("Disarm confirmed (armed=False). DONE.")
                        self.flight_phase = "DONE"
                    else:
                        self.get_logger().warn("Still armed=True after 1s, retry disarm.")
                        self._disarm_success_time = None

        elif self.flight_phase == "DONE":
            self.publish_markers()
            self.throttled_print(now)
            self.shutdown_requested()
            return

        self.publish_markers()
        self.throttled_print(now)

    def make_xy_hold_setpoint(self):
        # Hold XY at locked landing XY (or current if not locked); set z as current measured z (neutral)
        if self.land_xy_hold is None:
            xy = self.current_position[0:2].copy()
        else:
            xy = self.land_xy_hold
        z = float(self.current_position[2]) if self.current_position is not None else 0.0
        return np.array([float(xy[0]), float(xy[1]), z], dtype=float)

    # ============================================================
    # Throttled print
    # ============================================================

    def throttled_print(self, now):
        if (now - self._last_print_time) < Duration(seconds=self.print_interval_s):
            return
        self._last_print_time = now

        z = float(self.current_position[2]) if self.current_position is not None else float("nan")
        mode = self.current_state.mode if self.current_state is not None else ""
        armed = bool(self.current_state.armed) if self.current_state is not None else False
        land_xy = "None" if self.land_xy_hold is None else f"({self.land_xy_hold[0]:.2f},{self.land_xy_hold[1]:.2f})"
        land_z_cmd = "None" if self.land_z_cmd is None else f"{self.land_z_cmd:.2f}"

        self.get_logger().info(
            f"phase={self.flight_phase} mode={mode} armed={armed} z={z:.2f} "
            f"pass_idx={self.pass_idx}/{self.num_waypoints - 1} pass_r={self.pass_radius:.2f} "
            f"land_xy={land_xy} land_z_cmd={land_z_cmd} pose_timeout={self.pose_timed_out}"
        )

    # ============================================================
    # Service helpers
    # ============================================================

    def request_offboard_and_arm(self, now):
        if (now - self._last_req_time) < Duration(seconds=self.request_interval):
            return

        if not self.set_mode_client.service_is_ready() or not self.arming_client.service_is_ready():
            self._last_req_time = now
            return

        if self.current_state.mode != "OFFBOARD":
            req = SetMode.Request()
            req.custom_mode = "OFFBOARD"
            fut = self.set_mode_client.call_async(req)

            def _cb(f):
                try:
                    r = f.result()
                    if r is not None and r.mode_sent:
                        self.get_logger().info("OFFBOARD enabled.")
                except Exception as e:
                    self.get_logger().warn(f"SetMode OFFBOARD failed: {e}")

            fut.add_done_callback(_cb)
            self._last_req_time = now
            return

        if not self.current_state.armed:
            req = CommandBool.Request()
            req.value = True
            fut = self.arming_client.call_async(req)

            def _cb(f):
                try:
                    r = f.result()
                    if r is not None and r.success:
                        self.get_logger().info("Armed.")
                except Exception as e:
                    self.get_logger().warn(f"Arming failed: {e}")

            fut.add_done_callback(_cb)
            self._last_req_time = now
            return

        self._last_req_time = now

    def request_land_mode(self, now):
        if (now - self._last_req_time) < Duration(seconds=self.request_interval):
            return

        if not self.set_mode_client.service_is_ready():
            self._last_req_time = now
            return

        if self.current_state.mode == self.land_mode:
            self._last_req_time = now
            return

        req = SetMode.Request()
        req.custom_mode = self.land_mode
        fut = self.set_mode_client.call_async(req)

        def _cb(f):
            try:
                r = f.result()
                if r is not None and r.mode_sent:
                    self._land_mode_sent = True
                    self.get_logger().info(f"{self.land_mode} requested (mode_sent=True).")
            except Exception as e:
                self.get_logger().warn(f"SetMode {self.land_mode} failed: {e}")

        fut.add_done_callback(_cb)
        self._last_req_time = now

    def request_disarm(self, now):
        if (now - self._last_req_time) < Duration(seconds=self.request_interval):
            return

        if not self.arming_client.service_is_ready():
            self._last_req_time = now
            return

        if not self.current_state.armed:
            if self._disarm_success_time is None:
                self._disarm_success_time = now
                self.get_logger().info("Already disarmed, start 1s confirm timer.")
            self._last_req_time = now
            return

        req = CommandBool.Request()
        req.value = False
        fut = self.arming_client.call_async(req)

        def _cb(f):
            try:
                r = f.result()
                if r is not None and r.success:
                    self._disarm_success_time = self.get_clock().now()
                    self.get_logger().info("Disarm request success=True, wait 1s to confirm armed=False.")
            except Exception as e:
                self.get_logger().warn(f"Disarm failed: {e}")

        fut.add_done_callback(_cb)
        self._last_req_time = now

    # ============================================================
    # Guidance (unchanged)
    # ============================================================

    def update_guidance(self):
        _, s_candidate, _, _ = self.compute_projection(self.current_position)

        s_used = max(float(s_candidate), float(self.last_s_progress))
        s_used = min(s_used, self.total_length)
        self.last_s_progress = s_used

        self.current_segment_idx = self.segment_index_from_s(s_used)

        projection_point_used = self.point_at_s(s_used)
        self.projection_point = projection_point_used
        self.track_error = float(np.linalg.norm(self.current_position - projection_point_used))

        if not self.recover_mode:
            if self.track_error > self.max_track_error:
                self.recover_mode = True
        else:
            if self.track_error < self.recover_track_error:
                self.recover_mode = False

        s_lookahead = s_used + self.lookahead_distance
        self.lookahead_point = self.wp[-1].copy() if s_lookahead >= self.total_length else self.point_at_s(s_lookahead)
        target = projection_point_used.copy() if self.recover_mode else self.lookahead_point.copy()
        self.target_point = target.copy()
        return target

    def compute_projection(self, position):
        start_idx = max(0, self.current_segment_idx - self.search_back_segments)
        end_idx = min(self.num_segments - 1, self.current_segment_idx + self.search_forward_segments)

        best_dist = float("inf")
        best_s = float(self.cumulative_s[self.current_segment_idx])

        for i in range(start_idx, end_idx + 1):
            A = self.wp[i]
            B = self.wp[i + 1]
            v = B - A
            vv = float(np.dot(v, v))
            if vv < 1e-12:
                continue

            t = float(np.dot(position - A, v) / vv)
            t = max(0.0, min(1.0, t))
            proj = A + t * v
            dist = float(np.linalg.norm(position - proj))
            s_proj = float(self.cumulative_s[i] + t * self.segment_lengths[i])

            if dist < best_dist:
                best_dist = dist
                best_s = s_proj

        return None, best_s, best_dist, None

    def point_at_s(self, s_target):
        s_target = max(0.0, min(float(s_target), self.total_length))
        if s_target <= 0.0:
            return self.wp[0].copy()
        if s_target >= self.total_length:
            return self.wp[-1].copy()

        idx = int(np.searchsorted(self.cumulative_s, s_target, side="right") - 1)
        idx = max(0, min(idx, self.num_segments - 1))

        s0 = float(self.cumulative_s[idx])
        seg_len = float(self.segment_lengths[idx])
        ratio = (s_target - s0) / seg_len
        ratio = max(0.0, min(1.0, ratio))
        return self.wp[idx] + ratio * (self.wp[idx + 1] - self.wp[idx])

    def segment_index_from_s(self, s_value):
        s_value = max(0.0, min(float(s_value), self.total_length))
        if s_value >= self.total_length:
            return self.num_segments - 1
        idx = int(np.searchsorted(self.cumulative_s, s_value, side="right") - 1)
        return max(0, min(idx, self.num_segments - 1))

    # ============================================================
    # Publishers / markers
    # ============================================================

    def publish_setpoint(self, point_np):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(point_np[0])
        msg.pose.position.y = float(point_np[1])
        msg.pose.position.z = float(point_np[2])
        msg.pose.orientation.w = 1.0
        self.local_pos_pub.publish(msg)

    def publish_markers(self):
        now_msg = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        # history
        m = Marker()
        m.header.stamp = now_msg
        m.header.frame_id = self.frame_id
        m.ns = "history"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.lifetime = Duration(seconds=self.marker_lifetime).to_msg()
        m.scale.x = 0.04
        m.color.r, m.color.g, m.color.b, m.color.a = (0.0, 1.0, 0.0, 1.0)
        m.points = list(self.history_points)
        marker_array.markers.append(m)

        # target
        if self.target_point is not None:
            s = Marker()
            s.header.stamp = now_msg
            s.header.frame_id = self.frame_id
            s.ns = "target"
            s.id = 0
            s.type = Marker.SPHERE
            s.action = Marker.ADD
            s.lifetime = Duration(seconds=self.marker_lifetime).to_msg()
            s.pose.position.x = float(self.target_point[0])
            s.pose.position.y = float(self.target_point[1])
            s.pose.position.z = float(self.target_point[2])
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 0.18
            s.color.r, s.color.g, s.color.b, s.color.a = (0.7, 0.0, 1.0, 1.0)
            marker_array.markers.append(s)

        self.marker_pub.publish(marker_array)

    # ============================================================
    # Shutdown
    # ============================================================

    def shutdown_requested(self):
        if rclpy.ok():
            rclpy.shutdown()


def main():
    rclpy.init()
    node = OffboardPolylineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    try:
        node.destroy_node()
    except Exception:
        pass
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
