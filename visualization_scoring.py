#!/usr/bin/env python3
# Pose-based RViz visualization and competition scoring
# ROS 2 version
# -*- coding: utf-8 -*-

import math
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray


def quat_to_yaw(q):
    # geometry_msgs/Quaternion -> yaw(rad)
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw):
    # yaw(rad) -> geometry_msgs/Quaternion (z,w only)
    # roll=pitch=0
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class OffboardVizOnly(Node):
    def __init__(self):
        super().__init__("offboard_viz_only")

        # Frames，RViz Fixed Frame
        self.frame_id = "map"

        # PoseStamped topic. It can be MAVROS local pose or a Vicon/VRPN pose.
        self.pose_topic = "/mavros/local_position/pose"
        # Example Vicon topic:
        # self.pose_topic = "/mavros/vision_pose/pose"
        self.pose_timeout_s = 0.3

        # Publish rate
        self.rate_hz = 30.0
        self.dt = 1.0 / self.rate_hz

        # Waypoints [x, y, z, yaw_deg]
        # IMPORTANT
        self.waypoints = [
            [-1.67, -1.01, 1.00, 0.0],
            [-0.51, -0.11, 1.27, 0.0],
            [1.7, -0.60, 2.11, 45.0],
            [1.80, 1.58, 1.61, 135.0],
            [-0.96, 1.75, 2.21, -135.0],
            [-1.78, -0.44, 1.78, 0.0],
        ]
        # Gate visualization parameters
        self.gate_diameter = 1.0
        self.gate_pre_distance = 0.8
        self.gate_post_distance = 0.9
        self.gate_circle_points = 64
        self.gate_axis_length = 0.8

        # Expanded path lookahead
        self.lookahead_distance = 0.5

        # Local projection search
        self.search_back_segments = 1
        self.search_forward_segments = 2

        # History
        self.history_max_len = 1500

        # Evaluation interval defined by 3D distance to the first/last waypoint.
        # Timing and RMS tracking-error accumulation use the same interval.
        self.start_radius = 0.30
        self.finish_radius = 0.30
        self.start_confirm_s = 0.20
        self.finish_confirm_s = 0.20
        self.finish_progress_ratio = 0.80

        # Competition scoring
        self.error_full_score_m = 0.0
        self.error_zero_score_m = 0.50
        self.time_full_score_s = 5.0
        self.time_zero_score_s = 60.0
        self.accuracy_weight = 0.50
        self.time_weight = 0.50

        # Tracking error computed as distance to polyline 3D
        self.use_3d_error = True

        # Marker
        self.marker_lifetime = 0.3
        self.line_width = 0.05
        self.traj_width = 0.04

        # ============================================================
        # Internal state
        # ============================================================

        self.original_wp = np.array(self.waypoints, dtype=float)
        if self.original_wp.shape[0] < 2 or self.original_wp.shape[1] != 4:
            raise RuntimeError("waypoints must be Nx4 [x,y,z,yaw_deg].")

        self._build_expanded_path()
        self.wp = np.array(self.expanded_waypoints, dtype=float)
        self._precompute_path_arclength()

        self.has_pose = False
        self.pose_timed_out = True
        self.last_pose_time = None

        self.pos = None
        self.q = None

        self.history_points = deque(maxlen=self.history_max_len)

        # READY -> TRACKING -> FINISHED
        self.stage = "READY"
        self._start_candidate_time = None
        self._finish_candidate_time = None
        self._start_candidate_samples = []
        self._t_start = None
        self._t_finish = None

        # Running metric state
        self.error_sq_sum = 0.0
        self.error_count = 0
        self.error_rms = float("nan")

        # Sequential ring traversal state
        self.next_gate_idx = 0
        self.gate_passed = [False] * len(self.gates)
        self.previous_flight_pos = None
        self.run_valid = False
        self._result_printed = False

        # computed points for viz
        self.projection_point = None
        self.lookahead_point = None
        self.track_error = float("nan")
        self.s_progress = 0.0
        self.current_segment_idx = 0

        # ============================================================
        # ROS 2 interfaces
        # ============================================================

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_cb,
            sensor_qos,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/offboard_viz/markers",
            10,
        )

        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info("Viz-only node initialized.")
        self.get_logger().info(f"Pose topic: {self.pose_topic}")
        self.get_logger().info(f"RViz Fixed Frame: {self.frame_id}")
        self.get_logger().info(f"Expanded path points: {self.wp.shape[0]}")
        self.get_logger().info(
            "Evaluation rule: first/last waypoint radii = "
            f"{self.start_radius:.2f}/{self.finish_radius:.2f} m."
        )

    # ============================================================
    # Path build
    # ============================================================

    def _build_expanded_path(self):
        self.expanded_waypoints = []
        self.gates = []

        n = self.original_wp.shape[0]
        self.expanded_waypoints.append(self.original_wp[0, 0:3].copy())

        for i in range(1, n - 1):
            center = self.original_wp[i, 0:3].copy()
            yaw_deg = float(self.original_wp[i, 3])
            yaw_rad = math.radians(yaw_deg)
            forward = np.array(
                [math.cos(yaw_rad), math.sin(yaw_rad), 0.0],
                dtype=float
            )

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

    def _precompute_path_arclength(self):
        self.num_waypoints = self.wp.shape[0]
        self.num_segments = self.num_waypoints - 1
        self.segment_lengths = []
        self.cumulative_s = [0.0]

        for i in range(self.num_segments):
            v = self.wp[i + 1] - self.wp[i]
            L = float(np.linalg.norm(v))
            if L < 1e-6:
                raise RuntimeError(
                    "Expanded waypoint {}->{} too close.".format(i, i + 1)
                )
            self.segment_lengths.append(L)
            self.cumulative_s.append(self.cumulative_s[-1] + L)

        self.segment_lengths = np.array(self.segment_lengths, dtype=float)
        self.cumulative_s = np.array(self.cumulative_s, dtype=float)
        self.total_length = float(self.cumulative_s[-1])

    # ============================================================
    # Sub callback
    # ============================================================

    def pose_cb(self, msg):
        now = self.get_clock().now()
        self.last_pose_time = now
        self.pose_timed_out = False
        self.has_pose = True

        p = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ],
            dtype=float
        )
        self.pos = p
        self.q = msg.pose.orientation

        # Process every new pose message exactly once.
        self._update_stage_from_pose(now)

    # ============================================================
    # Core math: projection to polyline + point_at_s
    # ============================================================

    def compute_projection(self, position):
        best_dist = float("inf")
        best_s = float(self.cumulative_s[self.current_segment_idx])

        start_idx = max(0, self.current_segment_idx - self.search_back_segments)
        end_idx = min(
            self.num_segments - 1,
            self.current_segment_idx + self.search_forward_segments
        )

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

            if self.use_3d_error:
                dist = float(np.linalg.norm(position - proj))
            else:
                dist = float(np.linalg.norm(position[0:2] - proj[0:2]))

            s_proj = float(self.cumulative_s[i] + t * self.segment_lengths[i])
            if dist < best_dist:
                best_dist = dist
                best_s = s_proj

        # Never allow the path progress to jump backwards
        best_s = max(best_s, self.s_progress)
        best_s = min(best_s, self.total_length)

        best_proj = self.point_at_s(best_s)

        if self.use_3d_error:
            best_dist = float(np.linalg.norm(position - best_proj))
        else:
            best_dist = float(np.linalg.norm(position[0:2] - best_proj[0:2]))

        self.current_segment_idx = self.segment_index_from_s(best_s)
        return best_s, best_proj, best_dist

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
    # Stage logic
    # ============================================================

    def distance_to_waypoint(self, waypoint):
        if self.pos is None:
            return float("nan")
        return float(np.linalg.norm(self.pos - waypoint))

    def start_distance(self):
        return self.distance_to_waypoint(self.original_wp[0, 0:3])

    def finish_distance(self):
        return self.distance_to_waypoint(self.original_wp[-1, 0:3])

    def _reset_run(self):
        self.error_sq_sum = 0.0
        self.error_count = 0
        self.error_rms = float("nan")
        self.history_points.clear()
        self.next_gate_idx = 0
        self.gate_passed = [False] * len(self.gates)
        self.previous_flight_pos = None
        self.run_valid = False
        self._result_printed = False
        self.s_progress = 0.0
        self.current_segment_idx = 0

    def _update_projection(self, position):
        s, proj, err = self.compute_projection(position)
        self.s_progress = s
        self.projection_point = proj
        self.track_error = err
        self.lookahead_point = self.point_at_s(
            min(self.total_length, s + self.lookahead_distance)
        )

    def _append_flight_sample(self, position):
        self._update_projection(position)

        self.error_sq_sum += self.track_error * self.track_error
        self.error_count += 1
        self.error_rms = math.sqrt(self.error_sq_sum / self.error_count)

        pt = Point()
        pt.x = float(position[0])
        pt.y = float(position[1])
        pt.z = float(position[2])
        self.history_points.append(pt)

        self._update_gate_pass(position)

    def _update_gate_pass(self, position):
        """
        Detect ordered crossings of the physical ring planes using pose data.
        """
        if self.previous_flight_pos is None:
            self.previous_flight_pos = position.copy()
            return

        if self.next_gate_idx < len(self.gates):
            gate = self.gates[self.next_gate_idx]
            center = gate["center"]
            forward = gate["forward"]

            d_prev = float(np.dot(self.previous_flight_pos - center, forward))
            d_now = float(np.dot(position - center, forward))

            if d_prev < 0.0 <= d_now:
                denom = d_now - d_prev
                alpha = 1.0 if abs(denom) < 1e-9 else -d_prev / denom

                crossing = self.previous_flight_pos + alpha * (
                    position - self.previous_flight_pos
                )
                offset = crossing - center
                in_plane = offset - float(np.dot(offset, forward)) * forward
                radial_error = float(np.linalg.norm(in_plane))
                radius = 0.5 * float(gate["diameter"])

                if radial_error <= radius:
                    self.gate_passed[self.next_gate_idx] = True
                    self.next_gate_idx += 1
                    self.get_logger().info(
                        f"Ring {self.next_gate_idx}/{len(self.gates)} passed, "
                        f"radial error={radial_error:.3f} m."
                    )
                else:
                    self.get_logger().warn(
                        f"Missed ring {self.next_gate_idx + 1}: "
                        f"radial error={radial_error:.3f} m > {radius:.3f} m."
                    )

        self.previous_flight_pos = position.copy()

    def _finish_region_enabled(self):
        # Progress gating prevents an early finish if the final waypoint is
        # geometrically close to the start. Missing a ring does not block the
        # finish event; it makes the completed run INVALID instead.
        all_rings = self.next_gate_idx >= len(self.gates)
        near_end = self.s_progress >= self.finish_progress_ratio * self.total_length
        return all_rings or near_end

    def _update_stage_from_pose(self, now):
        if self.stage == "READY":
            # Projection is updated for RViz, but no error is accumulated yet.
            self._update_projection(self.pos)
            d_start = self.start_distance()

            if d_start <= self.start_radius:
                if self._start_candidate_time is None:
                    self._start_candidate_time = now
                    self._start_candidate_samples = []

                self._start_candidate_samples.append(self.pos.copy())
                elapsed = (now - self._start_candidate_time).nanoseconds / 1e9

                if elapsed >= self.start_confirm_s:
                    start_time = self._start_candidate_time
                    start_samples = list(self._start_candidate_samples)
                    self._reset_run()
                    self.stage = "TRACKING"

                    # Backdate timing to the first entry into the start sphere.
                    self._t_start = start_time
                    self._t_finish = None
                    self._finish_candidate_time = None
                    self._start_candidate_time = None
                    self._start_candidate_samples = []

                    # Include all samples collected since the first entry, so
                    # the RMS interval and timing interval start together.
                    for sample in start_samples:
                        self._append_flight_sample(sample)

                    self.get_logger().info(
                        "Start waypoint reached. Evaluation started."
                    )
            else:
                self._start_candidate_time = None
                self._start_candidate_samples = []

        elif self.stage == "TRACKING":
            # Update progress before checking the finish region.
            self._update_projection(self.pos)
            d_finish = self.finish_distance()

            finish_condition = (
                d_finish <= self.finish_radius
                and self._finish_region_enabled()
            )

            if finish_condition:
                if self._finish_candidate_time is None:
                    self._finish_candidate_time = now

                elapsed = (now - self._finish_candidate_time).nanoseconds / 1e9

                if elapsed >= self.finish_confirm_s:
                    self.stage = "FINISHED"

                    # Stop both timing and RMS accumulation at the first entry
                    # into the finish sphere, excluding the confirmation delay.
                    self._t_finish = self._finish_candidate_time
                    self.run_valid = self.next_gate_idx >= len(self.gates)
                    self._print_final_result()
            else:
                self._finish_candidate_time = None
                self._append_flight_sample(self.pos)

    def flight_time_s(self, now):
        if self._t_start is None:
            return 0.0

        if self._t_finish is not None:
            return (self._t_finish - self._t_start).nanoseconds / 1e9

        end_time = self._finish_candidate_time or now
        return max(0.0, (end_time - self._t_start).nanoseconds / 1e9)

    def scores(self, now):
        if self.error_count <= 0:
            return float("nan"), float("nan"), float("nan")

        error_span = self.error_zero_score_m - self.error_full_score_m
        accuracy = 100.0 * (self.error_zero_score_m - self.error_rms) / error_span
        accuracy = max(0.0, min(100.0, accuracy))

        duration = self.flight_time_s(now)
        time_span = self.time_zero_score_s - self.time_full_score_s
        time_score = 100.0 * (self.time_zero_score_s - duration) / time_span
        time_score = max(0.0, min(100.0, time_score))

        final = self.accuracy_weight * accuracy + self.time_weight * time_score

        if self.stage == "FINISHED" and not self.run_valid:
            final = 0.0

        return accuracy, time_score, final

    def _print_final_result(self):
        if self._result_printed:
            return

        self._result_printed = True
        now = self.get_clock().now()
        accuracy, time_score, final = self.scores(now)
        validity = "VALID" if self.run_valid else "INVALID"

        self.get_logger().info("========== FINAL RESULT ==========")
        self.get_logger().info(
            f"Run: {validity} | Rings: "
            f"{self.next_gate_idx}/{len(self.gates)}"
        )
        self.get_logger().info(f"Flight time: {self.flight_time_s(now):.3f} s")
        self.get_logger().info(f"RMS tracking error: {self.error_rms:.3f} m")
        self.get_logger().info(f"Accuracy score: {accuracy:.1f}/100")
        self.get_logger().info(f"Time score: {time_score:.1f}/100")
        self.get_logger().info(f"Final score: {final:.1f}/100")
        self.get_logger().info("==================================")

    # ============================================================
    # Main loop
    # ============================================================

    def loop(self):
        now = self.get_clock().now()

        # timeout
        if self.last_pose_time is None:
            self.pose_timed_out = True
        else:
            self.pose_timed_out = (
                (now - self.last_pose_time).nanoseconds / 1e9
                > self.pose_timeout_s
            )

        self.publish_markers(now)

    # ============================================================
    # Marker helpers
    # ============================================================

    def _mk(self, ns, mid, mtype):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame_id
        m.ns = ns
        m.id = int(mid)
        m.type = mtype
        m.action = Marker.ADD
        m.lifetime = Duration(seconds=self.marker_lifetime).to_msg()
        return m

    def publish_markers(self, now):
        ma = MarkerArray()

        # ----------------------------
        # 1) Reference polyline
        # ----------------------------
        m_path = self._mk("ref_path", 0, Marker.LINE_STRIP)
        m_path.scale.x = self.line_width
        m_path.color.r = 0.1
        m_path.color.g = 0.7
        m_path.color.b = 1.0
        m_path.color.a = 1.0
        m_path.points = []

        for p in self.wp:
            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = float(p[2])
            m_path.points.append(pt)

        ma.markers.append(m_path)

        # ----------------------------
        # 2) Gate visuals
        # ----------------------------
        base_id = 1000

        for gi, g in enumerate(self.gates):
            center = g["center"]
            yaw = g["yaw_rad"]
            forward = g["forward"]
            pre_p = g["pre_point"]
            post_p = g["post_point"]
            radius = 0.5 * float(g["diameter"])

            # Circle
            m_circle = self._mk(
                "gate_circle",
                base_id + gi * 10 + 0,
                Marker.LINE_STRIP
            )
            m_circle.scale.x = 0.035

            if self.gate_passed[gi]:
                m_circle.color.r = 0.1
                m_circle.color.g = 1.0
                m_circle.color.b = 0.2
                m_circle.color.a = 1.0
            else:
                m_circle.color.r = 1.0
                m_circle.color.g = 0.6
                m_circle.color.b = 0.0
                m_circle.color.a = 1.0

            pts = []

            # Circle plane: vertical ring whose normal is forward
            left = np.array(
                [-math.sin(yaw), math.cos(yaw), 0.0],
                dtype=float
            )
            up = np.array([0.0, 0.0, 1.0], dtype=float)

            for k in range(self.gate_circle_points + 1):
                ang = 2.0 * math.pi * k / self.gate_circle_points
                p = (
                    center
                    + radius * math.cos(ang) * left
                    + radius * math.sin(ang) * up
                )

                pt = Point()
                pt.x = float(p[0])
                pt.y = float(p[1])
                pt.z = float(p[2])
                pts.append(pt)

            m_circle.points = pts
            ma.markers.append(m_circle)

            # Axis
            m_axis = self._mk(
                "gate_axis",
                base_id + gi * 10 + 1,
                Marker.ARROW
            )
            m_axis.scale.x = 0.08
            m_axis.scale.y = 0.14
            m_axis.scale.z = 0.18
            m_axis.color.r = 1.0
            m_axis.color.g = 0.2
            m_axis.color.b = 0.2
            m_axis.color.a = 1.0

            p0 = Point()
            p0.x = float(center[0])
            p0.y = float(center[1])
            p0.z = float(center[2])

            p1 = center + float(self.gate_axis_length) * forward

            p1m = Point()
            p1m.x = float(p1[0])
            p1m.y = float(p1[1])
            p1m.z = float(p1[2])

            m_axis.points = [p0, p1m]
            ma.markers.append(m_axis)

            # Pre/post segment line
            m_ext = self._mk(
                "gate_ext",
                base_id + gi * 10 + 2,
                Marker.LINE_STRIP
            )
            m_ext.scale.x = 0.03
            m_ext.color.r = 1.0
            m_ext.color.g = 1.0
            m_ext.color.b = 0.0
            m_ext.color.a = 1.0
            m_ext.points = []

            for p in [pre_p, center, post_p]:
                pt = Point()
                pt.x = float(p[0])
                pt.y = float(p[1])
                pt.z = float(p[2])
                m_ext.points.append(pt)

            ma.markers.append(m_ext)

        # ----------------------------
        # 3) History trajectory
        # ----------------------------
        m_hist = self._mk("history", 0, Marker.LINE_STRIP)
        m_hist.scale.x = self.traj_width
        m_hist.color.r = 0.0
        m_hist.color.g = 1.0
        m_hist.color.b = 0.2
        m_hist.color.a = 1.0
        m_hist.points = list(self.history_points)
        ma.markers.append(m_hist)

        # ----------------------------
        # 4) UAV current pose arrow
        # ----------------------------
        if (
            self.has_pose
            and (not self.pose_timed_out)
            and self.pos is not None
            and self.q is not None
        ):
            m_uav = self._mk("uav", 0, Marker.ARROW)
            m_uav.scale.x = 0.7
            m_uav.scale.y = 0.12
            m_uav.scale.z = 0.12
            m_uav.color.r = 0.9
            m_uav.color.g = 0.9
            m_uav.color.b = 0.9
            m_uav.color.a = 1.0
            m_uav.pose.position.x = float(self.pos[0])
            m_uav.pose.position.y = float(self.pos[1])
            m_uav.pose.position.z = float(self.pos[2])
            m_uav.pose.orientation = self.q
            ma.markers.append(m_uav)

        # ----------------------------
        # 5) Projection / lookahead points
        # ----------------------------
        if self.projection_point is not None:
            mp = self._mk("projection", 0, Marker.SPHERE)
            mp.scale.x = 0.16
            mp.scale.y = 0.16
            mp.scale.z = 0.16
            mp.color.r = 0.2
            mp.color.g = 0.9
            mp.color.b = 1.0
            mp.color.a = 1.0
            mp.pose.position.x = float(self.projection_point[0])
            mp.pose.position.y = float(self.projection_point[1])
            mp.pose.position.z = float(self.projection_point[2])
            mp.pose.orientation.w = 1.0
            ma.markers.append(mp)

        if self.lookahead_point is not None:
            mt = self._mk("lookahead", 0, Marker.SPHERE)
            mt.scale.x = 0.18
            mt.scale.y = 0.18
            mt.scale.z = 0.18
            mt.color.r = 0.8
            mt.color.g = 0.0
            mt.color.b = 1.0
            mt.color.a = 1.0
            mt.pose.position.x = float(self.lookahead_point[0])
            mt.pose.position.y = float(self.lookahead_point[1])
            mt.pose.position.z = float(self.lookahead_point[2])
            mt.pose.orientation.w = 1.0
            ma.markers.append(mt)

        # ----------------------------
        # 6) Text HUD
        # ----------------------------
        hud = self._mk("hud", 0, Marker.TEXT_VIEW_FACING)
        hud.scale.z = 0.28
        hud.color.r = 1.0
        hud.color.g = 1.0
        hud.color.b = 1.0
        hud.color.a = 1.0

        if self.pos is not None:
            hud.pose.position.x = float(self.pos[0])
            hud.pose.position.y = float(self.pos[1])
            hud.pose.position.z = float(self.pos[2] + 0.6)
        else:
            hud.pose.position.x = 0.0
            hud.pose.position.y = 0.0
            hud.pose.position.z = 1.0

        hud.pose.orientation.w = 1.0

        t_flight = self.flight_time_s(now)

        if self.track_error is None or math.isnan(self.track_error):
            err_txt = "nan"
        else:
            err_txt = "{:.2f}".format(self.track_error)

        if math.isnan(self.error_rms):
            rms_txt = "nan"
        else:
            rms_txt = "{:.2f}".format(self.error_rms)

        d_start = self.start_distance()
        if math.isnan(d_start):
            start_txt = "nan"
        else:
            start_txt = "{:.2f}".format(d_start)

        d_finish = self.finish_distance()
        if math.isnan(d_finish):
            finish_txt = "nan"
        else:
            finish_txt = "{:.2f}".format(d_finish)

        accuracy, time_score, final_score = self.scores(now)

        if math.isnan(final_score):
            final_txt = "--"
        else:
            final_txt = "{:.1f}".format(final_score)

        valid_txt = ""
        if self.stage == "FINISHED":
            valid_txt = "VALID" if self.run_valid else "INVALID"

        hud.text = (
            "stage: {}\n"
            "start_distance: {} m\n"
            "finish_distance: {} m\n"
            "flight_time: {:.1f} s\n"
            "track_error: {} m\n"
            "e_RMS: {} m\n"
            "rings: {}/{}\n"
            "score: {}/100 {}\n"
            "pose_timeout: {}"
        ).format(
            self.stage,
            start_txt,
            finish_txt,
            t_flight,
            err_txt,
            rms_txt,
            self.next_gate_idx,
            len(self.gates),
            final_txt,
            valid_txt,
            self.pose_timed_out
        )

        ma.markers.append(hud)

        self.marker_pub.publish(ma)


def main():
    rclpy.init()
    node = OffboardVizOnly()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
