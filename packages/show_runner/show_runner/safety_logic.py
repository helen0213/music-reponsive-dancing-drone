"""Safety rules as a pure class (no ROS imports), so they can be unit-tested.

The safety node feeds every command from the player through
:meth:`SafetyLogic.check_command` and only forwards what it returns.

Rules
-----
* Geofence: the estimated position (takeoff point + the net offsets of every
  accepted command) plus the swept extent of a motion must stay inside the
  box ``+-half_x, +-half_y`` around the takeoff point, else the command is rejected.
* Altitude: absolute z must stay in ``[altitude_min, altitude_max]`` during
  every motion (takeoff and land may cross the lower bound on the way
  to/from the floor); otherwise the command is rejected.
* Speed clamp: a motion exceeding the velocity / acceleration / yaw-rate limits
  is *slowed down* (duration stretched), never sped up.
* Kill: :meth:`kill` latches; it emits an emergency land and every later
  command is refused.
* Watchdog: if no command (heartbeats count) arrives for ``link_timeout``
  seconds, emit a hover; if still nothing ``hover_before_land`` seconds later,
  land. Fresh commands after the hover cancel the pending land.

Position tracking is dead reckoning from commands, not from sensors. TODO:
replace/augment with a real position estimate once one is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from drone_primitives.primitives import build_motion
from drone_primitives.trajectory import Limits, time_scale_for_limits

EPS = 1e-6


@dataclass
class Decision:
    accepted: bool
    command: Optional[dict] = None
    reason: str = ""
    stretch: float = 1.0  # >1 means the motion was slowed down


class SafetyLogic:
    def __init__(self, limits: Limits, hover_altitude: float = 1.0, hover_before_land: float = 1.0):
        self.limits = limits
        self.hover_altitude = hover_altitude
        self.hover_before_land = hover_before_land
        self.position = np.zeros(3)  # estimated, relative to takeoff point
        self.flying = False
        self.killed = False
        self.landing = False
        self.last_command_time: Optional[float] = None
        self.hover_sent_time: Optional[float] = None

    # ------------------------------------------------------------------ #
    def heartbeat(self, now: float) -> None:
        self._touch(now)

    def _touch(self, now: float) -> None:
        self.last_command_time = now
        if not self.landing:
            self.hover_sent_time = None  # link recovered before the land fired

    def kill(self) -> List[dict]:
        if self.killed:
            return []
        self.killed = True
        self.landing = True
        return [{"primitive": "land", "emergency": True}]

    def watchdog(self, now: float) -> List[dict]:
        """Commands to emit if the command stream has gone silent."""
        if self.killed or self.landing or not self.flying or self.last_command_time is None:
            return []
        if self.hover_sent_time is None:
            if now - self.last_command_time > self.limits.link_timeout:
                self.hover_sent_time = now
                return [{"primitive": "hover", "duration": self.hover_before_land, "abort": True}]
            return []
        if now - self.hover_sent_time >= self.hover_before_land:
            self.landing = True
            return [{"primitive": "land", "emergency": False}]
        return []

    # ------------------------------------------------------------------ #
    def check_command(self, cmd: dict, now: float) -> Decision:
        self._touch(now)
        if self.killed:
            return Decision(False, reason="kill latched")
        if self.landing:
            return Decision(False, reason="landing in progress")

        name = cmd.get("primitive")
        params = dict(cmd.get("params") or {})
        duration = cmd.get("duration")
        lim = self.limits

        if name == "land":
            self.landing = True
            return Decision(True, {"primitive": "land", "emergency": False})

        if name == "takeoff":
            if self.flying:
                return Decision(False, reason="already flying")
            alt = float(params.get("altitude", self.hover_altitude))
            if not lim.altitude_min <= alt <= lim.altitude_max:
                return Decision(False, reason=f"takeoff altitude {alt} outside [{lim.altitude_min}, {lim.altitude_max}]")
            params["altitude"] = alt
            duration = duration or 3.0
        elif not self.flying:
            return Decision(False, reason=f"'{name}' before takeoff")

        try:
            motion = build_motion(name, params, duration)
        except (KeyError, ValueError, TypeError) as exc:
            return Decision(False, reason=f"bad command: {exc}")

        pk = motion.peaks()
        stretch = time_scale_for_limits(pk.velocity, pk.acceleration, pk.yaw_rate_dps, lim)
        if stretch > 1.0 + EPS:
            motion = motion.with_duration(motion.duration * stretch)
            pk = motion.peaks()

        lo = self.position + pk.offset_min
        hi = self.position + pk.offset_max
        if hi[0] > lim.geofence_half_x + EPS or lo[0] < -lim.geofence_half_x - EPS:
            return Decision(False, reason="geofence x")
        if hi[1] > lim.geofence_half_y + EPS or lo[1] < -lim.geofence_half_y - EPS:
            return Decision(False, reason="geofence y")
        if name == "takeoff":
            pass  # target altitude already checked; the climb starts on the floor
        elif hi[2] > lim.altitude_max + EPS or lo[2] < lim.altitude_min - EPS:
            return Decision(False, reason="altitude")

        off, _ = motion.final_offset
        self.position = self.position + off
        if name == "takeoff":
            self.flying = True
        out = {"primitive": name, "params": params, "duration": motion.duration}
        return Decision(True, out, stretch=stretch)
