#!/usr/bin/env python3
"""safety_node: sits between the player (choreography) and the primitives node.

  subscribes  show/cmd_in      std_msgs/String  JSON command or heartbeat from player_node
                               {"type": "move"|"heartbeat", "primitive": ..., "params": ..., "duration": ...}
              show/kill        std_msgs/Empty   publish anything here to land immediately
  service     show/kill        std_srvs/Trigger call to land immediately
  publishes   show/cmd_safe    std_msgs/String  vetted command for primitives_node
              show/safety_event std_msgs/String JSON {"event", "detail", "wall_time_s"}

All the rules live in safety_logic.SafetyLogic (unit-testable without ROS).
"""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from drone_primitives.trajectory import Limits

from .safety_logic import SafetyLogic


class SafetyNode(Node):
    def __init__(self):
        super().__init__("safety_node")
        self.declare_parameter("config_path", "")
        self.declare_parameter("hover_altitude", 1.0)
        self.declare_parameter("hover_before_land_s", 1.0)
        self.declare_parameter("watchdog_rate", 20.0)

        cfg = self.get_parameter("config_path").value
        limits = Limits.from_yaml(cfg) if cfg else Limits()
        if not cfg:
            self.get_logger().warn("No config_path given: using default limits.")
        self.logic = SafetyLogic(
            limits,
            hover_altitude=float(self.get_parameter("hover_altitude").value),
            hover_before_land=float(self.get_parameter("hover_before_land_s").value),
        )

        self.create_subscription(String, "show/cmd_in", self._on_cmd, 50)
        self.create_subscription(Empty, "show/kill", lambda _m: self._do_kill("topic"), 10)
        self.create_service(Trigger, "show/kill", self._on_kill_srv)
        self.pub_safe = self.create_publisher(String, "show/cmd_safe", 50)
        self.pub_event = self.create_publisher(String, "show/safety_event", 50)
        self.create_timer(1.0 / float(self.get_parameter("watchdog_rate").value), self._on_watchdog)
        self.get_logger().info("safety_node ready.")

    def _forward(self, cmd: dict):
        m = String()
        m.data = json.dumps(cmd)
        self.pub_safe.publish(m)

    def _event(self, event: str, detail: str = ""):
        m = String()
        m.data = json.dumps({"event": event, "detail": detail, "wall_time_s": time.time()})
        self.pub_event.publish(m)

    def _on_cmd(self, msg: String):
        now = time.monotonic()
        try:
            cmd = json.loads(msg.data)
        except ValueError:
            self.get_logger().error(f"Unparseable command: {msg.data!r}")
            return
        if cmd.get("type") == "heartbeat":
            self.logic.heartbeat(now)
            return
        decision = self.logic.check_command(cmd, now)
        if decision.accepted:
            if decision.stretch > 1.0:
                self._event("slowed", f"{cmd.get('primitive')} x{decision.stretch:.2f}")
            self._forward(decision.command)
        else:
            self.get_logger().warn(f"Rejected {cmd.get('primitive')}: {decision.reason}")
            self._event("rejected", f"{cmd.get('primitive')}: {decision.reason}")

    def _do_kill(self, source: str):
        self.get_logger().error(f"KILL requested via {source}: landing now.")
        for cmd in self.logic.kill():
            self._forward(cmd)
        self._event("kill", source)
        # TODO(tutorial): also trigger the flight controller's own emergency stop
        # if wanted (the altitude LX force-disarms through /mavros/cmd/command).

    def _on_kill_srv(self, _req, resp):
        self._do_kill("service")
        resp.success = True
        resp.message = "kill latched; landing."
        return resp

    def _on_watchdog(self):
        for cmd in self.logic.watchdog(time.monotonic()):
            self.get_logger().warn(f"No command for > {self.logic.limits.link_timeout}s: {cmd['primitive']}")
            self._event("watchdog", cmd["primitive"])
            self._forward(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
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
