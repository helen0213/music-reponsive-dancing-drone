"""Movement primitives and patterns as pure functions (no ROS imports).

Every primitive is described by a name, a params dict and a duration, and is
turned into a :class:`Motion`: a smooth (min-jerk) path of *offsets* relative
to wherever the drone was when the motion started.

Frames / units: offsets are in the fixed takeoff frame (x forward at takeoff,
y left, z up), in metres; they do NOT rotate with the drone's yaw. Yaw is in
radians internally, ``angle_deg`` in params. TODO: confirm the axis convention
against the drone's setpoint frame once the setpoint interface is known (the
MAVROS tutorials use ENU for setpoints).

Primitives : takeoff, land, hover, ascend, descend, translate, yaw
Patterns   : bounce, side_step, circle, spin
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .trajectory import min_jerk_s

Shape = Callable[[float], Tuple[np.ndarray, float]]  # tau in [0,1] -> (xyz offset, yaw offset rad)

N_PEAK_SAMPLES = 400


@dataclass
class Peaks:
    """Kinematic extremes of a motion (relative to its start)."""

    velocity: float  # m/s, norm of the xyz velocity
    acceleration: float  # m/s^2
    yaw_rate_dps: float  # deg/s
    offset_min: np.ndarray  # per-axis minimum xyz offset (m)
    offset_max: np.ndarray  # per-axis maximum xyz offset (m)


@dataclass
class Motion:
    name: str
    duration: float
    shape: Shape
    heading: np.ndarray  # unit vector of the outbound direction (zeros if none)

    def sample(self, t: float) -> Tuple[np.ndarray, float]:
        """(xyz offset, yaw offset) at time ``t`` seconds after the start."""
        tau = min(max(t / self.duration, 0.0), 1.0)
        xyz, yaw = self.shape(tau)
        return np.asarray(xyz, dtype=float), float(yaw)

    @property
    def final_offset(self) -> Tuple[np.ndarray, float]:
        return self.sample(self.duration)

    def with_duration(self, duration: float) -> "Motion":
        return Motion(self.name, duration, self.shape, self.heading)

    def peaks(self, n: int = N_PEAK_SAMPLES) -> Peaks:
        ts = np.linspace(0.0, self.duration, n + 1)
        pos = np.array([self.sample(t)[0] for t in ts])
        yaw = np.array([self.sample(t)[1] for t in ts])
        dt = ts[1] - ts[0]
        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        yaw_rate = np.gradient(yaw, dt)
        return Peaks(
            velocity=float(np.max(np.linalg.norm(vel, axis=1))),
            acceleration=float(np.max(np.linalg.norm(acc, axis=1))),
            yaw_rate_dps=float(np.max(np.abs(np.degrees(yaw_rate)))),
            offset_min=pos.min(axis=0),
            offset_max=pos.max(axis=0),
        )


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
_Leg = Tuple[Sequence[float], float, float]  # (delta xyz, delta yaw rad, weight)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(3)


def _from_legs(name: str, duration: float, legs: List[_Leg], heading=None) -> Motion:
    """Chain of min-jerk legs, each starting and ending at rest."""
    weights = np.array([w for _, _, w in legs], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(weights) / weights.sum()])
    deltas = [np.asarray(d, dtype=float) for d, _, _ in legs]
    dyaws = [float(y) for _, y, _ in legs]
    if heading is None:
        heading = _unit(deltas[0])

    def shape(tau: float):
        pos = np.zeros(3)
        yaw = 0.0
        for i, (d, dy) in enumerate(zip(deltas, dyaws)):
            a, b = edges[i], edges[i + 1]
            if tau >= b:
                pos = pos + d
                yaw += dy
            else:
                s = float(min_jerk_s((tau - a) / (b - a))) if tau > a else 0.0
                return pos + d * s, yaw + dy * s
        return pos, yaw

    return Motion(name, duration, shape, np.asarray(heading, dtype=float))


def _sign(x: float) -> float:
    return -1.0 if x < 0 else 1.0


# --------------------------------------------------------------------------- #
# Primitives (net displacement allowed)
# --------------------------------------------------------------------------- #
def takeoff(params: dict, duration: float) -> Motion:
    h = float(params.get("altitude", 1.0))
    return _from_legs("takeoff", duration, [((0, 0, h), 0.0, 1.0)])


def land(params: dict, duration: float) -> Motion:
    """Descend to the floor from ``params['altitude']`` (current height, m)."""
    h = float(params.get("altitude", 1.0))
    return _from_legs("land", duration, [((0, 0, -h), 0.0, 1.0)])


def hover(params: dict, duration: float) -> Motion:
    return Motion("hover", duration, lambda tau: (np.zeros(3), 0.0), np.zeros(3))


def ascend(params: dict, duration: float) -> Motion:
    dz = float(params.get("dz", 0.2))
    return _from_legs("ascend", duration, [((0, 0, dz), 0.0, 1.0)])


def descend(params: dict, duration: float) -> Motion:
    dz = float(params.get("dz", 0.2))
    return _from_legs("descend", duration, [((0, 0, -dz), 0.0, 1.0)])


def translate(params: dict, duration: float) -> Motion:
    dx, dy = float(params.get("dx", 0.0)), float(params.get("dy", 0.0))
    return _from_legs("translate", duration, [((dx, dy, 0.0), 0.0, 1.0)])


def yaw(params: dict, duration: float) -> Motion:
    ang = math.radians(float(params.get("angle_deg", 90.0)))
    return _from_legs("yaw", duration, [((0, 0, 0), ang, 1.0)], heading=np.zeros(3))


# --------------------------------------------------------------------------- #
# Patterns (out-and-back, net displacement ~ 0)
# --------------------------------------------------------------------------- #
def bounce(params: dict, duration: float) -> Motion:
    """Up by dz and back down; the apex is the accent (half way through)."""
    dz = float(params.get("dz", 0.15))
    return _from_legs("bounce", duration, [((0, 0, dz), 0.0, 1.0), ((0, 0, -dz), 0.0, 1.0)])


def side_step(params: dict, duration: float) -> Motion:
    """Step sideways by dy (+ = left) and return."""
    dy = float(params.get("dy", 0.2))
    return _from_legs("side_step", duration, [((0, dy, 0), 0.0, 1.0), ((0, -dy, 0), 0.0, 1.0)])


def circle(params: dict, duration: float) -> Motion:
    """One full horizontal circle of ``radius`` through the start point.

    The angle follows a min-jerk profile so velocity is zero at start and end.
    ``direction`` +1 = initially forward then to the left (counter-clockwise
    seen from above)."""
    r = float(params.get("radius", 0.2))
    d = _sign(float(params.get("direction", 1)))

    def shape(tau: float):
        th = 2.0 * math.pi * float(min_jerk_s(tau))
        # centre at (0, d*r); start at origin heading along +x
        return np.array([r * math.sin(th), d * r * (1.0 - math.cos(th)), 0.0]), 0.0

    return Motion("circle", duration, shape, np.array([1.0, 0.0, 0.0]))


def spin(params: dict, duration: float) -> Motion:
    """Rotate in place by ``angle_deg`` (default one full turn)."""
    ang = math.radians(float(params.get("angle_deg", 360.0)))
    return _from_legs("spin", duration, [((0, 0, 0), ang, 1.0)], heading=np.zeros(3))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
PRIMITIVES: Dict[str, Callable[[dict, float], Motion]] = {
    "takeoff": takeoff,
    "land": land,
    "hover": hover,
    "ascend": ascend,
    "descend": descend,
    "translate": translate,
    "yaw": yaw,
    "bounce": bounce,
    "side_step": side_step,
    "circle": circle,
    "spin": spin,
}

# Dance patterns the choreographer may pick, with the param that scales with
# music energy and its (min, max) range (m, or degrees for spin).
PATTERNS = ("bounce", "side_step", "circle", "spin")
AMPLITUDE_PARAM = {"bounce": "dz", "side_step": "dy", "circle": "radius", "spin": "angle_deg"}
AMPLITUDE_RANGE = {
    "bounce": (0.05, 0.25),
    "side_step": (0.10, 0.40),
    "circle": (0.10, 0.35),
    "spin": (90.0, 360.0),
}


def build_motion(name: str, params: dict | None, duration: float) -> Motion:
    """Build the :class:`Motion` for a primitive; raises KeyError if unknown."""
    if name not in PRIMITIVES:
        raise KeyError(f"unknown primitive '{name}'")
    if duration is None or duration <= 0:
        raise ValueError("duration must be > 0")
    return PRIMITIVES[name](dict(params or {}), float(duration))
