#!/usr/bin/env python3
"""player_node: runs a precomputed choreography schedule against one monotonic clock.

The music is played elsewhere (a phone); this node never touches audio. It
waits for a start signal that says "song time 0 is N seconds from now".

  subscribes  show/takeoff       std_msgs/Empty    take off and start the heartbeat
              show/start_in      std_msgs/Float64  start the schedule, song t=0 in N seconds
                                                   (negative = the song started that long ago)
              show/motion_event  std_msgs/String    (from primitives_node, logged)
              show/safety_event  std_msgs/String    (from safety_node, logged)
  publishes   show/cmd_in        std_msgs/String   JSON commands + 10 Hz heartbeat -> safety_node

Each command is published early by the latency offsets in latency.yaml. Every
event is appended to a CSV in the logs folder:
    wall_time_s, song_time_s, event, primitive, beat_s, detail
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Float64, String

from .player_logic import Latency, build_send_plan, load_schedule, schedule_end_time

LATE_WARN_S = 0.05
STALE_S = 0.25  # a command later than this is skipped, not played in a burst


class PlayerNode(Node):
    def __init__(self):
        super().__init__("player_node")
        self.declare_parameter("schedule_path", "")
        self.declare_parameter("config_path", "")  # directory holding limits.yaml and latency.yaml
        self.declare_parameter("log_dir", "")
        self.declare_parameter("heartbeat_rate", 10.0)
        self.declare_parameter("tick_rate", 100.0)
        self.declare_parameter("end_pad_s", 1.0)  # wait after the last move before landing
        self.declare_parameter("auto_land", True)

        cfg_dir = self.get_parameter("config_path").value
        latency_file = Path(cfg_dir) / "latency.yaml" if cfg_dir else None
        if latency_file and latency_file.is_file():
            self.latency = Latency.from_yaml(latency_file)
        else:
            self.latency = Latency()
            self.get_logger().warn("No latency.yaml found: using zero latency offsets.")

        self.plan, self.end_time = [], 0.0
        sched = self.get_parameter("schedule_path").value
        if sched:
            schedule = load_schedule(sched)
            self.plan = build_send_plan(schedule, self.latency)
            self.end_time = schedule_end_time(schedule) + float(self.get_parameter("end_pad_s").value)
            self.get_logger().info(f"Loaded {len(self.plan)} commands from {sched}")
        else:
            self.get_logger().warn("No schedule_path given: nothing to play.")

        self._open_log()

        self.t0 = None  # monotonic time of song t=0
        self.index = 0
        self.running = False
        self.heartbeat_on = False
        self.last_heartbeat = 0.0

        self.pub_cmd = self.create_publisher(String, "show/cmd_in", 50)
        self.create_subscription(Empty, "show/takeoff", self._on_takeoff, 10)
        self.create_subscription(Float64, "show/start_in", self._on_start, 10)
        self.create_subscription(String, "show/motion_event", lambda m: self._log_external("motion", m), 50)
        self.create_subscription(String, "show/safety_event", lambda m: self._log_external("safety", m), 50)
        self.create_timer(1.0 / float(self.get_parameter("tick_rate").value), self._tick)
        self.get_logger().info("player_node ready: publish show/takeoff, then show/start_in.")

    # ---------------- logging ---------------- #
    def _open_log(self):
        log_dir = self.get_parameter("log_dir").value
        if not log_dir:
            log_dir = os.path.join(os.environ.get("DT_PROJECT_PATH", "."), "logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / f"show_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._log_file = open(self.log_path, "w", newline="")
        self._csv = csv.writer(self._log_file)
        self._csv.writerow(["wall_time_s", "song_time_s", "event", "primitive", "beat_s", "detail"])
        self._log("node_start", detail=str(self.log_path))
        self.get_logger().info(f"Logging to {self.log_path}")

    def _song_time(self):
        return "" if self.t0 is None else f"{time.monotonic() - self.t0:.4f}"

    def _log(self, event, primitive="", beat_s="", detail=""):
        self._csv.writerow([f"{time.time():.4f}", self._song_time(), event, primitive, beat_s, detail])
        self._log_file.flush()

    def _log_external(self, kind, msg: String):
        try:
            d = json.loads(msg.data)
            self._log(f"{kind}:{d.get('event', '')}", d.get("primitive", ""), "", msg.data)
        except ValueError:
            self._log(f"{kind}:raw", detail=msg.data)

    # ---------------- inputs ---------------- #
    def _publish(self, payload: dict):
        m = String()
        m.data = json.dumps(payload)
        self.pub_cmd.publish(m)

    def _on_takeoff(self, _msg):
        self._publish({"type": "move", "primitive": "takeoff"})
        self.heartbeat_on = True
        self._log("takeoff", "takeoff")

    def _on_start(self, msg: Float64):
        if self.running:
            self.get_logger().warn("start_in ignored: show already running.")
            return
        delay = float(msg.data)
        self.t0 = time.monotonic() + delay
        self.index = 0
        self.running = True
        self._log("start", detail=f"song t=0 in {delay:.3f}s (wall {time.time() + delay:.4f})")
        self.get_logger().info(f"Show starts in {delay:.2f}s")

    # ---------------- main loop ---------------- #
    def _tick(self):
        now = time.monotonic()
        if self.heartbeat_on and now - self.last_heartbeat >= 1.0 / float(self.get_parameter("heartbeat_rate").value):
            self._publish({"type": "heartbeat"})
            self.last_heartbeat = now
        if not self.running:
            return

        song_t = now - self.t0
        while self.index < len(self.plan) and song_t >= self.plan[self.index].send_time:
            c = self.plan[self.index]
            late = song_t - c.send_time
            if late > STALE_S:  # joined mid-song or fell behind: drop rather than burst
                self._log("skipped_stale", c.primitive, f"{c.t_peak:.4f}", f"late_ms={late * 1000:.1f}")
                self.index += 1
                continue
            self._publish({"type": "move", "primitive": c.primitive, "params": c.params, "duration": c.duration})
            self._log(
                "cmd_sent", c.primitive, f"{c.t_peak:.4f}",
                f"planned_send={c.send_time:.4f} t_start={c.t_start:.4f} late_ms={late * 1000:.1f}",
            )
            if late > LATE_WARN_S:
                self.get_logger().warn(f"{c.primitive} sent {late * 1000:.0f} ms late")
            self.index += 1

        if self.index >= len(self.plan) and song_t >= self.end_time:
            self.running = False
            self._log("show_end")
            if self.get_parameter("auto_land").value:
                self._publish({"type": "move", "primitive": "land"})
                self._log("land", "land")
            self.heartbeat_on = False

    def destroy_node(self):
        try:
            self._log("node_stop")
            self._log_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PlayerNode()
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
