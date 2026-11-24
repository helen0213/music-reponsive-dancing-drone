#!/usr/bin/env python3
"""primitives_node: executes movement primitives and patterns as smooth setpoint streams.

Interface (all names are OUR OWN internal topics, not Duckietown's):

  subscribes  show/cmd_safe     std_msgs/String   JSON command, already vetted by safety_node
                                {"primitive": "bounce", "params": {"dz": 0.15},
                                 "duration": 0.9, "emergency": false, "abort": false}
  publishes   show/motion_event std_msgs/String   JSON {"event": "motion_start"|"motion_end",
                                "primitive": ..., "wall_time_s": ..., "duration_s": ...}

Commands are queued and executed one after the other. ``emergency`` or ``land``
flushes the queue and descends; ``abort`` flushes the queue and freezes in
place before running the command (used by the safety watchdog's hover).

The only place that talks to the real drone is ``send_setpoint`` (plus the
arm/disarm stubs) below: both are TODOs until the setpoint interface is known.
"""

import collections
import json
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .primitives import Motion, build_motion
from .trajectory import PEAK_VEL_FACTOR, Limits, time_scale_for_limits


class PrimitivesNode(Node):
    def __init__(self):
        super().__init__("primitives_node")
        self.declare_parameter("config_path", "")
        self.declare_parameter("hover_altitude", 1.0)  # m, takeoff target
        self.declare_parameter("control_rate", 20.0)  # Hz, setpoint stream rate (PX4 needs > 2 Hz)
        self.declare_parameter("land_speed", 0.25)  # m/s, nominal descent speed

        cfg = self.get_parameter("config_path").value
        if cfg:
            self.limits = Limits.from_yaml(cfg)
        else:
            self.limits = Limits()
            self.get_logger().warn("No config_path given: using default limits.")
        self.hover_altitude = float(self.get_parameter("hover_altitude").value)
        self.land_speed = float(self.get_parameter("land_speed").value)
        rate = float(self.get_parameter("control_rate").value)

        # Current commanded state (relative to the takeoff point).
        self.anchor = np.zeros(3)
        self.anchor_yaw = 0.0
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.queue = collections.deque()
        self.current: Optional[Motion] = None
        self.current_start = 0.0
        self.landing = False

        self.create_subscription(String, "show/cmd_safe", self._on_cmd, 10)
        self.pub_event = self.create_publisher(String, "show/motion_event", 10)
        self.create_timer(1.0 / rate, self._step)
        self.get_logger().info("primitives_node ready.")

    # ------------------------------------------------------------------ #
    # TODO: the ONLY drone-facing functions. Fill in from the Duckietown tutorial.
    # ------------------------------------------------------------------ #
    def send_setpoint(self, position, yaw, velocity, yaw_rate):
        """TODO(tutorial): publish one setpoint to the drone.

        position: np.ndarray (x, y, z) m relative to the takeoff point.
        yaw: rad. velocity: np.ndarray (vx, vy, vz) m/s. yaw_rate: rad/s.

        What the tutorials show so far (lx-dd-altitude-pid-control): only the
        LOW-LEVEL interface, i.e. /mavros/setpoint_attitude/attitude
        (geometry_msgs/PoseStamped) + /mavros/setpoint_attitude/thrust
        (mavros_msgs/Thrust), streamed at ~20 Hz, with the altitude loop closed
        on /<VEHICLE_NAME>/bottom_tof_driver_node/range. No position or velocity
        setpoint topic is documented there, so none is invented here.
        """
        self.get_logger().debug(
            f"[no-op send_setpoint] pos={np.round(position, 3)} yaw={yaw:.3f}",
            throttle_duration_sec=1.0,
        )

    def request_arm(self):
        """TODO(tutorial): arm + enter the offboard/guided mode before takeoff.
        (Tutorials use /mavros/set_mode and /mavros/cmd/arming; confirm the
        intended services for this project before wiring them.)"""
        self.get_logger().warn("TODO request_arm(): not implemented (no-op).")

    def request_disarm(self):
        """TODO(tutorial): disarm after landing."""
        self.get_logger().warn("TODO request_disarm(): not implemented (no-op).")

    # ------------------------------------------------------------------ #
    def _on_cmd(self, msg: String):
        try:
            cmd = json.loads(msg.data)
            name = cmd["primitive"]
            params = dict(cmd.get("params") or {})
            duration = float(cmd["duration"]) if "duration" in cmd else None
            emergency = bool(cmd.get("emergency", False))
        except (ValueError, KeyError, TypeError) as exc:
            self.get_logger().error(f"Bad command {msg.data!r}: {exc}")
            return

        if emergency or name == "land":
            self._begin_landing(emergency)
            return
        if cmd.get("abort"):  # e.g. watchdog hover: drop everything, freeze in place
            self.queue.clear()
            if self.current is not None:
                self.anchor = self.position.copy()
                self.anchor_yaw = self.yaw
                self.current = None
        if self.landing:
            self.get_logger().warn(f"Ignoring '{name}': landing in progress.")
            return
        if name == "takeoff":
            params.setdefault("altitude", self.hover_altitude)
            duration = duration or max(2.0, PEAK_VEL_FACTOR * params["altitude"] / self.land_speed)
            self.request_arm()
        try:
            motion = build_motion(name, params, duration)
        except (KeyError, ValueError) as exc:
            self.get_logger().error(f"Rejected '{name}': {exc}")
            return
        self.queue.append(motion)

    def _begin_landing(self, emergency: bool):
        """Flush everything and descend from the current commanded height."""
        self.queue.clear()
        if self.current is not None:  # abort in place
            self.anchor = self.position.copy()
            self.anchor_yaw = self.yaw
            self.current = None
        self.landing = True
        z = max(float(self.position[2]), 0.0)
        duration = max(2.0, PEAK_VEL_FACTOR * z / self.land_speed)
        self.queue.append(build_motion("land", {"altitude": z}, duration))
        self.get_logger().warn(f"{'EMERGENCY ' if emergency else ''}landing from z={z:.2f} m over {duration:.1f} s")

    def _limit_motion(self, motion: Motion) -> Motion:
        """Stretch a motion in time so it respects the configured limits."""
        pk = motion.peaks()
        f = time_scale_for_limits(pk.velocity, pk.acceleration, pk.yaw_rate_dps, self.limits)
        if f > 1.0 + 1e-6 and motion.name not in ("land",):
            self.get_logger().warn(f"'{motion.name}' exceeds limits: slowing x{f:.2f}.")
            return motion.with_duration(motion.duration * f)
        return motion

    def _publish_event(self, event: str, motion: Motion):
        m = String()
        m.data = json.dumps(
            {"event": event, "primitive": motion.name, "wall_time_s": time.time(), "duration_s": motion.duration}
        )
        self.pub_event.publish(m)

    def _step(self):
        now = time.monotonic()
        if self.current is None and self.queue:
            self.current = self._limit_motion(self.queue.popleft())
            self.current_start = now
            self._publish_event("motion_start", self.current)

        velocity = np.zeros(3)
        yaw_rate = 0.0
        if self.current is not None:
            t = now - self.current_start
            off, yaw_off = self.current.sample(t)
            self.position = self.anchor + off
            self.yaw = self.anchor_yaw + yaw_off
            eps = 0.01
            o2, y2 = self.current.sample(t + eps)
            o1, y1 = self.current.sample(max(t - eps, 0.0))
            span = (t + eps) - max(t - eps, 0.0)
            velocity = (o2 - o1) / span
            yaw_rate = (y2 - y1) / span
            if t >= self.current.duration:
                off_f, yaw_f = self.current.final_offset
                self.anchor = self.anchor + off_f
                self.anchor_yaw += yaw_f
                self._publish_event("motion_end", self.current)
                finished = self.current
                self.current = None
                if finished.name == "land":
                    self.anchor[2] = 0.0
                    self.position = self.anchor.copy()
                    self.request_disarm()
                    self.landing = False
        else:
            self.position = self.anchor.copy()
            self.yaw = self.anchor_yaw

        self.send_setpoint(self.position, self.yaw, velocity, yaw_rate)


def main(args=None):
    rclpy.init(args=args)
    node = PrimitivesNode()
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
