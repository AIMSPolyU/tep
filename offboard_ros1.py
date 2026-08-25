#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math
import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.srv import SetMode, SetModeRequest
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


class OffboardPolylineFollower(object):
    def __init__(self):
        rospy.init_node("offboard_polyline_follower_py", anonymous=False)

        # Parameters
        self.rate_hz = 20.0
        self.dt = 1.0 / self.rate_hz

        self.frame_id = "map"
        self.world_frame = "world"
        self.map_frame = "map"

        self.pose_topic = "/mavros/local_position/pose"
        self.pose_timeout_s = 0.2

        self.waypoints = [
            [0.0, 0.0, 2.0, 0.0],
            [2.0, 0.0, 2.0, 45.0],
            [2.0, 2.0, 2.0, 135.0],
            [0.0, 2.0, 2.0, -135.0],
            [-1.0, 0.0, 2.0, 0.0],
        ]

        self.gate_diameter = 1.0
        self.gate_pre_distance = 0.8
        self.gate_post_distance = 0.9
        self.gate_circle_points = 64
        self.gate_axis_length = 0.8

        self.lookahead_distance = 0.5
        self.max_track_error = 1.5
        self.recover_track_error = 0.8
        self.takeoff_accept_radius = 0.3
        self.final_accept_radius = 0.4

        self.trajectory_max_len = 200
        self.search_back_segments = 1
        self.search_forward_segments = 2

        self.pass_radius = 0.5
        self.use_3d_pass_check = True

        self.finish_hold_s = 1.0
        self.land_descent_rate = 0.30
        self.land_min_z_cmd = 0.05
        self.land_switch_to_auto_land_z = 0.25
        self.land_mode = "AUTO.LAND"
        self.land_z_threshold = 0.15
        self.land_confirm_s = 2.0
        self.post_disarm_check_s = 1.0

        self.print_interval_s = 1.0
        self.request_interval = 1.0
        self.marker_lifetime = 0.5
        self.prestream_count = 100
        self.axis_length = 0.6

        # State
        self.current_state = State()

        self.current_pose = None
        self.current_position = None
        self.current_orientation = None
        self.has_pose = False

        self.last_pose_time = None
        self.pose_timed_out = False

        self.mission_state = "WAIT_FCU"
        self.flight_phase = "TAKEOFF"

        self.recover_mode = False
        self.current_segment_idx = 0
        self.last_s_progress = 0.0

        self.projection_point = None
        self.lookahead_point = None
        self.target_point = None
        self.track_error = 0.0

        self.history_points = []
        self.pass_idx = 1

        self.land_xy_hold = None
        self.land_z_cmd = None
        self._finish_enter_time = None
        self._land_low_enter_time = None
        self._disarm_success_time = None
        self._land_mode_sent = False

        self._last_req_time = rospy.Time.now()
        self._prestream_sent = 0
        self._last_print_time = rospy.Time.now()

        self.expanded_waypoints = []
        self.gates = []

        # Path setup
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
                raise RuntimeError(
                    "Expanded waypoint {} and waypoint {} are too close or identical.".format(i, i + 1)
                )

            self.segment_lengths.append(length)
            self.cumulative_s.append(self.cumulative_s[-1] + length)

        self.segment_lengths = np.array(self.segment_lengths, dtype=float)
        self.cumulative_s = np.array(self.cumulative_s, dtype=float)
        self.total_length = float(self.cumulative_s[-1])

        self.pass_idx = max(1, min(self.pass_idx, self.num_waypoints - 1))

        # ROS interfaces
        self.state_sub = rospy.Subscriber(
            "/mavros/state",
            State,
            callback=self.state_cb,
            queue_size=10
        )

        self.local_pose_sub = rospy.Subscriber(
            self.pose_topic,
            PoseStamped,
            callback=self.pose_cb,
            queue_size=10
        )

        self.local_pos_pub = rospy.Publisher(
            "/mavros/setpoint_position/local",
            PoseStamped,
            queue_size=10
        )

        self.marker_pub = rospy.Publisher(
            "/offboard_polyline/markers",
            MarkerArray,
            queue_size=10
        )

        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.publish_static_tf_world_to_map()

        rospy.loginfo("Waiting for /mavros/cmd/arming service...")
        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy(
            "/mavros/cmd/arming",
            CommandBool
        )

        rospy.loginfo("Waiting for /mavros/set_mode service...")
        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy(
            "/mavros/set_mode",
            SetMode
        )

        rospy.loginfo("OffboardPolylineFollower ROS1 initialized.")
        rospy.loginfo("Pose topic: %s", self.pose_topic)
        rospy.loginfo("Original waypoint number: %d", self.original_wp.shape[0])
        rospy.loginfo("Expanded waypoint number: %d", self.wp.shape[0])
        rospy.loginfo("Virtual gate number: %d", len(self.gates))
        rospy.loginfo("Total expanded path length: %.3f m", self.total_length)
        rospy.loginfo(
            "Sequential pass: pass_radius=%.2fm, use_3d_pass_check=%s. "
            "OFFBOARD descent rate=%.2f m/s, switch to %s at z<%.2f m, "
            "land confirm z<%.2f for %.2fs",
            self.pass_radius,
            str(self.use_3d_pass_check),
            self.land_descent_rate,
            self.land_mode,
            self.land_switch_to_auto_land_z,
            self.land_z_threshold,
            self.land_confirm_s
        )
        rospy.loginfo(
            "Static TF: %s -> %s.",
            self.world_frame,
            self.map_frame
        )

    def publish_static_tf_world_to_map(self):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.map_frame

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.static_tf_broadcaster.sendTransform(t)

    def build_expanded_path(self):
        self.expanded_waypoints = []
        self.gates = []

        n = self.original_wp.shape[0]

        self.expanded_waypoints.append(self.original_wp[0, 0:3].copy())

        for i in range(1, n - 1):
            center = self.original_wp[i, 0:3].copy()
            yaw_deg = float(self.original_wp[i, 3])
            yaw_rad = math.radians(yaw_deg)

            forward = np.array(
                [
                    math.cos(yaw_rad),
                    math.sin(yaw_rad),
                    0.0
                ],
                dtype=float
            )

            pre_point = center - self.gate_pre_distance * forward
            post_point = center + self.gate_post_distance * forward

            self.expanded_waypoints.append(pre_point)
            self.expanded_waypoints.append(center)
            self.expanded_waypoints.append(post_point)

            gate_info = {
                "index": i,
                "center": center,
                "yaw_deg": yaw_deg,
                "yaw_rad": yaw_rad,
                "forward": forward,
                "pre_point": pre_point,
                "post_point": post_point,
                "diameter": self.gate_diameter
            }

            self.gates.append(gate_info)

        self.expanded_waypoints.append(self.original_wp[-1, 0:3].copy())

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.current_pose = msg
        self.current_position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ],
            dtype=float
        )

        self.current_orientation = msg.pose.orientation
        self.has_pose = True
        self.last_pose_time = rospy.Time.now()
        self.pose_timed_out = False

        p = Point()
        p.x = float(msg.pose.position.x)
        p.y = float(msg.pose.position.y)
        p.z = float(msg.pose.position.z)

        self.history_points.append(p)

        if len(self.history_points) > self.trajectory_max_len:
            self.history_points = self.history_points[-self.trajectory_max_len:]

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("Starting main loop.")

        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()

    def loop(self):
        now = rospy.Time.now()
        initial_point = np.array([0.0, 0.0, 0.0], dtype=float)

        if self.last_pose_time is None:
            self.pose_timed_out = True
        else:
            elapsed = (now - self.last_pose_time).to_sec()
            self.pose_timed_out = elapsed > self.pose_timeout_s

        if self.mission_state == "WAIT_FCU":
            self.publish_setpoint(initial_point)
            self.publish_markers()

            if self.current_state.connected:
                rospy.loginfo("FCU connected.")
                self.mission_state = "WAIT_POSE"

            return

        if self.mission_state == "WAIT_POSE":
            self.publish_setpoint(initial_point)
            self.publish_markers()

            if self.has_pose and (not self.pose_timed_out):
                rospy.loginfo("Pose received.")
                self.mission_state = "PRESTREAM"

            return

        if self.mission_state == "PRESTREAM":
            self.publish_setpoint(initial_point)
            self.publish_markers()

            self._prestream_sent += 1

            if self._prestream_sent >= self.prestream_count:
                rospy.loginfo("Prestream done. RUN.")
                self.mission_state = "RUN"
                self._last_req_time = now

            return

        if self.flight_phase in ["TAKEOFF", "FOLLOW", "FINISH_HOLD", "LAND_OFFBOARD"]:
            self.request_offboard_and_arm(now)

        if self.pose_timed_out or (not self.has_pose) or (self.current_position is None):
            if self.current_position is not None:
                target = self.current_position.copy()
            else:
                target = initial_point.copy()

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
                self.pass_idx = 1
                rospy.loginfo("Reached first waypoint. FOLLOW.")

            self.target_point = target.copy()
            self.publish_setpoint(target)

        elif self.flight_phase == "FOLLOW":
            target = self.update_guidance()

            if self.pass_idx < self.num_waypoints:
                wp_next = self.wp[self.pass_idx]

                if self.use_3d_pass_check:
                    d = float(np.linalg.norm(self.current_position - wp_next))
                else:
                    d = float(np.linalg.norm(self.current_position[0:2] - wp_next[0:2]))

                if d < self.pass_radius:
                    rospy.loginfo(
                        "Passed wp[%d] d=%.2f < %.2f.",
                        self.pass_idx,
                        d,
                        self.pass_radius
                    )
                    self.pass_idx += 1

            if self.pass_idx >= self.num_waypoints:
                self.flight_phase = "FINISH_HOLD"
                self._finish_enter_time = now
                rospy.loginfo(
                    "All waypoints passed. Enter FINISH_HOLD %.2fs.",
                    self.finish_hold_s
                )
                target = self.wp[-1].copy()

            self.target_point = target.copy()
            self.publish_setpoint(target)

        elif self.flight_phase == "FINISH_HOLD":
            target = self.wp[-1].copy()
            self.target_point = target.copy()
            self.publish_setpoint(target)

            if self._finish_enter_time is None:
                self._finish_enter_time = now

            if (now - self._finish_enter_time).to_sec() >= self.finish_hold_s:
                self.flight_phase = "LAND_OFFBOARD"
                self.land_xy_hold = self.current_position[0:2].copy()
                self.land_z_cmd = float(self.current_position[2])
                self._land_low_enter_time = None

                rospy.loginfo(
                    "Enter LAND_OFFBOARD: lock XY=(%.2f, %.2f), start z_cmd=%.2f",
                    self.land_xy_hold[0],
                    self.land_xy_hold[1],
                    self.land_z_cmd
                )

        elif self.flight_phase == "LAND_OFFBOARD":
            if z_meas <= self.land_switch_to_auto_land_z:
                self.flight_phase = "LAND_REQUEST"
                self._land_mode_sent = False
                self._last_req_time = now - rospy.Duration(self.request_interval)

                rospy.loginfo(
                    "z_meas=%.2f <= %.2f. Switch to %s.",
                    z_meas,
                    self.land_switch_to_auto_land_z,
                    self.land_mode
                )

            else:
                dz = self.land_descent_rate / self.rate_hz

                if self.land_z_cmd is None:
                    self.land_z_cmd = float(self.current_position[2])

                if self.land_xy_hold is None:
                    self.land_xy_hold = self.current_position[0:2].copy()

                self.land_z_cmd = max(
                    self.land_min_z_cmd,
                    float(self.land_z_cmd) - float(dz)
                )

                sp = np.array(
                    [
                        self.land_xy_hold[0],
                        self.land_xy_hold[1],
                        self.land_z_cmd
                    ],
                    dtype=float
                )

                self.target_point = sp.copy()
                self.publish_setpoint(sp)

        elif self.flight_phase == "LAND_REQUEST":
            self.request_land_mode(now)

            sp = self.make_xy_hold_setpoint()
            self.target_point = sp.copy()
            self.publish_setpoint(sp)

            if self.current_state.mode == self.land_mode:
                self.flight_phase = "LANDING_CONFIRM"
                self._land_low_enter_time = None
                rospy.loginfo("%s active. LANDING_CONFIRM.", self.land_mode)

        elif self.flight_phase == "LANDING_CONFIRM":
            sp = self.make_xy_hold_setpoint()
            self.target_point = sp.copy()
            self.publish_setpoint(sp)

            if z_meas < self.land_z_threshold:
                if self._land_low_enter_time is None:
                    self._land_low_enter_time = now

                low_elapsed = (now - self._land_low_enter_time).to_sec()

                if low_elapsed >= self.land_confirm_s:
                    self.flight_phase = "DISARM_REQUEST"
                    self._disarm_success_time = None
                    self._last_req_time = now - rospy.Duration(self.request_interval)
                    rospy.loginfo("Landed confirmed. DISARM_REQUEST.")
            else:
                self._land_low_enter_time = None

        elif self.flight_phase == "DISARM_REQUEST":
            sp = self.make_xy_hold_setpoint()
            self.target_point = sp.copy()
            self.publish_setpoint(sp)

            self.request_disarm(now)

            if self._disarm_success_time is not None:
                elapsed = (now - self._disarm_success_time).to_sec()

                if elapsed >= self.post_disarm_check_s:
                    if not self.current_state.armed:
                        rospy.loginfo("Disarm confirmed armed=False. DONE.")
                        self.flight_phase = "DONE"
                    else:
                        rospy.logwarn("Still armed=True after 1s, retry disarm.")
                        self._disarm_success_time = None

        elif self.flight_phase == "DONE":
            self.publish_markers()
            self.throttled_print(now)

            rospy.signal_shutdown("Mission DONE.")
            return

        else:
            rospy.logwarn_throttle(
                2.0,
                "Unknown flight_phase: %s",
                self.flight_phase
            )

        self.publish_markers()
        self.throttled_print(now)

    def make_xy_hold_setpoint(self):
        if self.current_position is None:
            return np.array([0.0, 0.0, 0.0], dtype=float)

        if self.land_xy_hold is None:
            xy = self.current_position[0:2].copy()
        else:
            xy = self.land_xy_hold

        z = float(self.current_position[2])

        return np.array(
            [
                float(xy[0]),
                float(xy[1]),
                z
            ],
            dtype=float
        )

    def throttled_print(self, now):
        if (now - self._last_print_time) < rospy.Duration(self.print_interval_s):
            return

        self._last_print_time = now

        z = float(self.current_position[2]) if self.current_position is not None else float("nan")
        mode = self.current_state.mode if self.current_state is not None else ""
        armed = bool(self.current_state.armed) if self.current_state is not None else False

        if self.land_xy_hold is None:
            land_xy = "None"
        else:
            land_xy = "({:.2f},{:.2f})".format(
                self.land_xy_hold[0],
                self.land_xy_hold[1]
            )

        if self.land_z_cmd is None:
            land_z_cmd = "None"
        else:
            land_z_cmd = "{:.2f}".format(self.land_z_cmd)

        rospy.loginfo(
            "mission=%s phase=%s mode=%s armed=%s z=%.2f "
            "pass_idx=%d/%d pass_r=%.2f "
            "land_xy=%s land_z_cmd=%s pose_timeout=%s",
            self.mission_state,
            self.flight_phase,
            mode,
            str(armed),
            z,
            self.pass_idx,
            self.num_waypoints - 1,
            self.pass_radius,
            land_xy,
            land_z_cmd,
            str(self.pose_timed_out)
        )

    def request_offboard_and_arm(self, now):
        if (now - self._last_req_time) < rospy.Duration(self.request_interval):
            return

        if self.current_state.mode != "OFFBOARD":
            req = SetModeRequest()
            req.custom_mode = "OFFBOARD"

            try:
                response = self.set_mode_client.call(req)
                if response.mode_sent:
                    rospy.loginfo("OFFBOARD enabled.")
            except rospy.ServiceException as e:
                rospy.logwarn("SetMode OFFBOARD failed: %s", str(e))

            self._last_req_time = now
            return

        if not self.current_state.armed:
            req = CommandBoolRequest()
            req.value = True

            try:
                response = self.arming_client.call(req)
                if response.success:
                    rospy.loginfo("Armed.")
            except rospy.ServiceException as e:
                rospy.logwarn("Arming failed: %s", str(e))

            self._last_req_time = now
            return

        self._last_req_time = now

    def request_land_mode(self, now):
        if (now - self._last_req_time) < rospy.Duration(self.request_interval):
            return

        if self.current_state.mode == self.land_mode:
            self._last_req_time = now
            return

        req = SetModeRequest()
        req.custom_mode = self.land_mode

        try:
            response = self.set_mode_client.call(req)
            if response.mode_sent:
                self._land_mode_sent = True
                rospy.loginfo("%s requested mode_sent=True.", self.land_mode)
        except rospy.ServiceException as e:
            rospy.logwarn("SetMode %s failed: %s", self.land_mode, str(e))

        self._last_req_time = now

    def request_disarm(self, now):
        if (now - self._last_req_time) < rospy.Duration(self.request_interval):
            return

        if not self.current_state.armed:
            if self._disarm_success_time is None:
                self._disarm_success_time = now
                rospy.loginfo("Already disarmed, start 1s confirm timer.")

            self._last_req_time = now
            return

        req = CommandBoolRequest()
        req.value = False

        try:
            response = self.arming_client.call(req)

            if response.success:
                self._disarm_success_time = rospy.Time.now()
                rospy.loginfo("Disarm request success=True, wait 1s to confirm armed=False.")

        except rospy.ServiceException as e:
            rospy.logwarn("Disarm failed: %s", str(e))

        self._last_req_time = now

    def update_guidance(self):
        _, s_candidate, _, _ = self.compute_projection(self.current_position)

        s_used = max(float(s_candidate), float(self.last_s_progress))
        s_used = min(s_used, self.total_length)
        self.last_s_progress = s_used

        self.current_segment_idx = self.segment_index_from_s(s_used)

        projection_point_used = self.point_at_s(s_used)
        self.projection_point = projection_point_used.copy()

        self.track_error = float(np.linalg.norm(self.current_position - projection_point_used))

        if not self.recover_mode:
            if self.track_error > self.max_track_error:
                self.recover_mode = True
                rospy.logwarn(
                    "Enter RECOVER mode. track_error=%.3f m",
                    self.track_error
                )
        else:
            if self.track_error < self.recover_track_error:
                self.recover_mode = False
                rospy.loginfo(
                    "Exit RECOVER mode. track_error=%.3f m",
                    self.track_error
                )

        s_lookahead = s_used + self.lookahead_distance

        if s_lookahead >= self.total_length:
            self.lookahead_point = self.wp[-1].copy()
        else:
            self.lookahead_point = self.point_at_s(s_lookahead)

        if self.recover_mode:
            target = projection_point_used.copy()
        else:
            target = self.lookahead_point.copy()

        self.target_point = target.copy()
        return target

    def compute_projection(self, position):
        start_idx = max(
            0,
            self.current_segment_idx - self.search_back_segments
        )
        end_idx = min(
            self.num_segments - 1,
            self.current_segment_idx + self.search_forward_segments
        )

        best_dist = float("inf")
        best_s = float(self.cumulative_s[self.current_segment_idx])
        best_projection = self.wp[self.current_segment_idx].copy()
        best_idx = self.current_segment_idx

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
                best_projection = proj
                best_idx = i

        return best_projection, best_s, best_dist, best_idx

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

    def publish_setpoint(self, point_np):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.frame_id

        pose.pose.position.x = float(point_np[0])
        pose.pose.position.y = float(point_np[1])
        pose.pose.position.z = float(point_np[2])

        pose.pose.orientation.w = 1.0

        self.local_pos_pub.publish(pose)

    def publish_markers(self):
        now = rospy.Time.now()
        marker_array = MarkerArray()

        marker_array.markers.append(
            self.make_reference_path_marker(now)
        )

        marker_array.markers.append(
            self.make_original_waypoints_marker(now)
        )

        marker_array.markers.append(
            self.make_history_marker(now)
        )

        for gate_id, gate in enumerate(self.gates):
            marker_array.markers.append(
                self.make_gate_circle_marker(now, gate_id, gate)
            )
            marker_array.markers.append(
                self.make_gate_direction_marker(now, gate_id, gate)
            )
            marker_array.markers.append(
                self.make_gate_text_marker(now, gate_id, gate)
            )
            marker_array.markers.append(
                self.make_gate_ext_marker(now, gate_id, gate)
            )

        if self.projection_point is not None:
            marker_array.markers.append(
                self.make_sphere_marker(
                    now=now,
                    ns="projection_point",
                    marker_id=0,
                    point_np=self.projection_point,
                    scale=0.18,
                    color=(1.0, 1.0, 0.0, 1.0)
                )
            )

        if self.lookahead_point is not None:
            marker_array.markers.append(
                self.make_sphere_marker(
                    now=now,
                    ns="lookahead_point",
                    marker_id=0,
                    point_np=self.lookahead_point,
                    scale=0.22,
                    color=(1.0, 0.0, 0.0, 1.0)
                )
            )

        if self.target_point is not None:
            marker_array.markers.append(
                self.make_sphere_marker(
                    now=now,
                    ns="target_point",
                    marker_id=0,
                    point_np=self.target_point,
                    scale=0.16,
                    color=(0.7, 0.0, 1.0, 1.0)
                )
            )

        if self.has_pose and self.current_orientation is not None and self.current_position is not None:
            axis_markers = self.make_body_axis_markers(now)
            marker_array.markers.extend(axis_markers)

        if self.has_pose and self.current_position is not None:
            marker_array.markers.append(
                self.make_status_text_marker(now)
            )

        self.marker_pub.publish(marker_array)

    def init_marker(self, now, ns, marker_id, marker_type):
        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = self.frame_id
        marker.ns = ns
        marker.id = int(marker_id)
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(self.marker_lifetime)
        marker.pose.orientation.w = 1.0
        return marker

    def make_reference_path_marker(self, now):
        marker = self.init_marker(
            now=now,
            ns="expanded_reference_path",
            marker_id=0,
            marker_type=Marker.LINE_STRIP
        )

        marker.scale.x = 0.05

        marker.color.r = 0.0
        marker.color.g = 0.3
        marker.color.b = 1.0
        marker.color.a = 1.0

        for p in self.wp:
            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = float(p[2])
            marker.points.append(pt)

        return marker

    def make_original_waypoints_marker(self, now):
        marker = self.init_marker(
            now=now,
            ns="original_waypoints",
            marker_id=0,
            marker_type=Marker.SPHERE_LIST
        )

        marker.scale.x = 0.16
        marker.scale.y = 0.16
        marker.scale.z = 0.16

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        for p in self.original_wp:
            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = float(p[2])
            marker.points.append(pt)

        return marker

    def make_history_marker(self, now):
        marker = self.init_marker(
            now=now,
            ns="history_traj",
            marker_id=0,
            marker_type=Marker.LINE_STRIP
        )

        marker.scale.x = 0.04

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.points = list(self.history_points)

        return marker

    def make_gate_circle_marker(self, now, gate_id, gate):
        marker = self.init_marker(
            now=now,
            ns="virtual_gate_circle",
            marker_id=gate_id,
            marker_type=Marker.LINE_STRIP
        )

        marker.scale.x = 0.04

        gate_post_expanded_idx = gate_id * 3 + 3

        if self.pass_idx > gate_post_expanded_idx:
            marker.color.r = 0.1
            marker.color.g = 1.0
            marker.color.b = 0.2
            marker.color.a = 1.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.55
            marker.color.b = 0.0
            marker.color.a = 1.0

        center = gate["center"]
        yaw_rad = gate["yaw_rad"]
        radius = 0.5 * float(gate["diameter"])

        horizontal_axis = np.array(
            [
                -math.sin(yaw_rad),
                math.cos(yaw_rad),
                0.0
            ],
            dtype=float
        )

        vertical_axis = np.array(
            [
                0.0,
                0.0,
                1.0
            ],
            dtype=float
        )

        for k in range(self.gate_circle_points + 1):
            theta = 2.0 * math.pi * float(k) / float(self.gate_circle_points)

            p = (
                center
                + radius * math.cos(theta) * horizontal_axis
                + radius * math.sin(theta) * vertical_axis
            )

            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = float(p[2])
            marker.points.append(pt)

        return marker

    def make_gate_direction_marker(self, now, gate_id, gate):
        marker = self.init_marker(
            now=now,
            ns="virtual_gate_direction",
            marker_id=gate_id,
            marker_type=Marker.ARROW
        )

        center = gate["center"]
        forward = gate["forward"]

        start_np = center
        end_np = center + self.gate_axis_length * forward

        start = Point()
        start.x = float(start_np[0])
        start.y = float(start_np[1])
        start.z = float(start_np[2])

        end = Point()
        end.x = float(end_np[0])
        end.y = float(end_np[1])
        end.z = float(end_np[2])

        marker.points.append(start)
        marker.points.append(end)

        marker.scale.x = 0.04
        marker.scale.y = 0.10
        marker.scale.z = 0.16

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        return marker

    def make_gate_ext_marker(self, now, gate_id, gate):
        marker = self.init_marker(
            now=now,
            ns="virtual_gate_pre_post",
            marker_id=gate_id,
            marker_type=Marker.LINE_STRIP
        )

        marker.scale.x = 0.03

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for p in [gate["pre_point"], gate["center"], gate["post_point"]]:
            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = float(p[2])
            marker.points.append(pt)

        return marker

    def make_gate_text_marker(self, now, gate_id, gate):
        marker = self.init_marker(
            now=now,
            ns="virtual_gate_text",
            marker_id=gate_id,
            marker_type=Marker.TEXT_VIEW_FACING
        )

        center = gate["center"]

        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2] + 0.8)

        marker.scale.z = 0.22

        marker.color.r = 1.0
        marker.color.g = 0.8
        marker.color.b = 0.2
        marker.color.a = 1.0

        marker.text = "Gate {}\nyaw: {:.1f} deg\nD: {:.2f} m".format(
            gate["index"],
            gate["yaw_deg"],
            gate["diameter"]
        )

        return marker

    def make_sphere_marker(self, now, ns, marker_id, point_np, scale, color):
        marker = self.init_marker(
            now=now,
            ns=ns,
            marker_id=marker_id,
            marker_type=Marker.SPHERE
        )

        marker.pose.position.x = float(point_np[0])
        marker.pose.position.y = float(point_np[1])
        marker.pose.position.z = float(point_np[2])

        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        return marker

    def make_body_axis_markers(self, now):
        markers = []

        q = self.current_orientation
        R = self.quat_to_rot_matrix(q.x, q.y, q.z, q.w)

        origin = self.current_position.copy()

        axes = [
            ("body_axis", 0, R[:, 0], (1.0, 0.0, 0.0, 1.0)),
            ("body_axis", 1, R[:, 1], (0.0, 1.0, 0.0, 1.0)),
            ("body_axis", 2, R[:, 2], (0.0, 0.2, 1.0, 1.0)),
        ]

        for ns, marker_id, direction, color in axes:
            marker = self.init_marker(
                now=now,
                ns=ns,
                marker_id=marker_id,
                marker_type=Marker.ARROW
            )

            start = Point()
            start.x = float(origin[0])
            start.y = float(origin[1])
            start.z = float(origin[2])

            end_np = origin + self.axis_length * direction

            end = Point()
            end.x = float(end_np[0])
            end.y = float(end_np[1])
            end.z = float(end_np[2])

            marker.points.append(start)
            marker.points.append(end)

            marker.scale.x = 0.04
            marker.scale.y = 0.08
            marker.scale.z = 0.12

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            markers.append(marker)

        return markers

    def make_status_text_marker(self, now):
        marker = self.init_marker(
            now=now,
            ns="status_text",
            marker_id=0,
            marker_type=Marker.TEXT_VIEW_FACING
        )

        marker.pose.position.x = float(self.current_position[0])
        marker.pose.position.y = float(self.current_position[1])
        marker.pose.position.z = float(self.current_position[2] + 0.8)

        marker.scale.z = 0.25

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = (
            "mission: {}\n"
            "phase: {}\n"
            "mode: {}\n"
            "armed: {}\n"
            "recover: {}\n"
            "track_error: {:.2f} m\n"
            "s: {:.2f} / {:.2f} m\n"
            "segment: {}\n"
            "pass_idx: {} / {}\n"
            "pose_timeout: {}"
        ).format(
            self.mission_state,
            self.flight_phase,
            self.current_state.mode,
            self.current_state.armed,
            self.recover_mode,
            self.track_error,
            self.last_s_progress,
            self.total_length,
            self.current_segment_idx,
            self.pass_idx,
            self.num_waypoints - 1,
            self.pose_timed_out
        )

        return marker

    def quat_to_rot_matrix(self, x, y, z, w):
        norm_q = math.sqrt(x * x + y * y + z * z + w * w)

        if norm_q < 1e-9:
            return np.eye(3)

        x /= norm_q
        y /= norm_q
        z /= norm_q
        w /= norm_q

        xx = x * x
        yy = y * y
        zz = z * z

        xy = x * y
        xz = x * z
        yz = y * z

        wx = w * x
        wy = w * y
        wz = w * z

        R = np.array(
            [
                [
                    1.0 - 2.0 * (yy + zz),
                    2.0 * (xy - wz),
                    2.0 * (xz + wy)
                ],
                [
                    2.0 * (xy + wz),
                    1.0 - 2.0 * (xx + zz),
                    2.0 * (yz - wx)
                ],
                [
                    2.0 * (xz - wy),
                    2.0 * (yz + wx),
                    1.0 - 2.0 * (xx + yy)
                ]
            ],
            dtype=float
        )

        return R


if __name__ == "__main__":
    try:
        follower = OffboardPolylineFollower()
        follower.run()
    except rospy.ROSInterruptException:
        pass
