"""Minimum-jerk trajectories and limit clamping.

Pure Python + numpy (no ROS imports) so it can be unit-tested and reused by the
choreographer, the safety node and the primitives node.

Minimum-jerk point-to-point motion over duration T (tau = t / T):

    s(tau)  = 10 tau^3 - 15 tau^4 + 6 tau^5          (0 -> 1, zero vel/acc at both ends)
    s'(tau) = 30 tau^2 - 60 tau^3 + 30 tau^4
    s''(tau)= 60 tau - 180 tau^2 + 120 tau^3

Peak velocity     = 1.875  * d / T       (at tau = 0.5)
Peak acceleration = 5.7735 * d / T^2     (at tau = 0.5 +/- 0.2113)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

PEAK_VEL_FACTOR = 1.875
PEAK_ACC_FACTOR = 10.0 / math.sqrt(3.0)  # ~5.7735


# --------------------------------------------------------------------------- #
# Limits (read from assets/config/limits.yaml)
# --------------------------------------------------------------------------- #
@dataclass
class Limits:
    geofence_half_x: float = 1.0  # m, box half-size around the takeoff point
    geofence_half_y: float = 1.0
    altitude_min: float = 0.5  # m
    altitude_max: float = 1.5
    max_velocity: float = 0.5  # m/s
    max_accel: float = 1.0  # m/s^2
    max_yaw_rate: float = 90.0  # deg/s
    link_timeout: float = 0.5  # s
    min_duration: Dict[str, float] = field(default_factory=dict)  # per primitive, s

    @classmethod
    def from_dict(cls, d: dict) -> "Limits":
        g = d.get("geofence", {})
        a = d.get("altitude", {})
        return cls(
            geofence_half_x=float(g.get("half_x_m", 1.0)),
            geofence_half_y=float(g.get("half_y_m", 1.0)),
            altitude_min=float(a.get("min_m", 0.5)),
            altitude_max=float(a.get("max_m", 1.5)),
            max_velocity=float(d.get("max_velocity_mps", 0.5)),
            max_accel=float(d.get("max_accel_mps2", 1.0)),
            max_yaw_rate=float(d.get("max_yaw_rate_dps", 90.0)),
            link_timeout=float(d.get("link_timeout_s", 0.5)),
            min_duration={k: float(v) for k, v in (d.get("min_duration_s") or {}).items()},
        )

    @classmethod
    def from_yaml(cls, path) -> "Limits":
        with open(Path(path), "r") as f:
            return cls.from_dict(yaml.safe_load(f) or {})


# --------------------------------------------------------------------------- #
# Minimum-jerk interpolation
# --------------------------------------------------------------------------- #
def min_jerk_s(tau):
    """Normalized min-jerk position profile s(tau); tau is clipped to [0, 1]."""
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def min_jerk_ds(tau):
    """ds/dtau."""
    tau = np.clip(tau, 0.0, 1.0)
    return 30 * tau**2 - 60 * tau**3 + 30 * tau**4


def min_jerk_dds(tau):
    """d2s/dtau2."""
    tau = np.clip(tau, 0.0, 1.0)
    return 60 * tau - 180 * tau**2 + 120 * tau**3


def min_jerk(p0, p1, duration: float, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-jerk interpolation from p0 to p1 over ``duration``.

    Works for scalars and vectors. Returns (position, velocity, acceleration)
    at time ``t`` (clipped to [0, duration]; velocity/acc are 0 outside).
    """
    if duration <= 0:
        raise ValueError("duration must be > 0")
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    tau = t / duration
    inside = 0.0 <= tau <= 1.0
    pos = p0 + d * min_jerk_s(tau)
    vel = d * min_jerk_ds(tau) / duration if inside else np.zeros_like(d)
    acc = d * min_jerk_dds(tau) / duration**2 if inside else np.zeros_like(d)
    return pos, vel, acc


@dataclass
class MinJerkSegment:
    """A min-jerk move between two setpoints (positions and/or yaw)."""

    p0: np.ndarray
    p1: np.ndarray
    duration: float

    def __post_init__(self):
        self.p0 = np.asarray(self.p0, dtype=float)
        self.p1 = np.asarray(self.p1, dtype=float)

    def sample(self, t: float):
        return min_jerk(self.p0, self.p1, self.duration, t)

    @property
    def peak_velocity(self) -> float:
        return PEAK_VEL_FACTOR * float(np.linalg.norm(self.p1 - self.p0)) / self.duration

    @property
    def peak_acceleration(self) -> float:
        return PEAK_ACC_FACTOR * float(np.linalg.norm(self.p1 - self.p0)) / self.duration**2


# --------------------------------------------------------------------------- #
# Duration / clamping helpers
# --------------------------------------------------------------------------- #
def min_duration_for_limits(distance: float, max_vel: float, max_acc: float) -> float:
    """Shortest min-jerk duration covering ``distance`` within both limits."""
    distance = abs(distance)
    if distance == 0:
        return 0.0
    return max(PEAK_VEL_FACTOR * distance / max_vel, math.sqrt(PEAK_ACC_FACTOR * distance / max_acc))


def plan_segment(p0, p1, duration: float, limits: Limits) -> MinJerkSegment:
    """Min-jerk segment whose duration is stretched, if needed, to respect the
    velocity and acceleration limits (never shortened)."""
    dist = float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float)))
    return MinJerkSegment(p0, p1, max(duration, min_duration_for_limits(dist, limits.max_velocity, limits.max_accel)))


def clamp_norm(v, max_norm: float) -> np.ndarray:
    """Scale vector ``v`` down so its Euclidean norm is <= max_norm."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v if n <= max_norm or n == 0 else v * (max_norm / n)


def clamp_velocity(v, limits: Limits) -> np.ndarray:
    return clamp_norm(v, limits.max_velocity)


def clamp_acceleration(a, limits: Limits) -> np.ndarray:
    return clamp_norm(a, limits.max_accel)


def clamp_yaw_rate(rate_dps: float, limits: Limits) -> float:
    return float(np.clip(rate_dps, -limits.max_yaw_rate, limits.max_yaw_rate))


def slew_limit(v_prev, v_desired, dt: float, limits: Limits) -> np.ndarray:
    """One control step of velocity command shaping: limit the change in
    velocity to max_accel * dt, then clamp the result to max_velocity."""
    v_prev = np.asarray(v_prev, dtype=float)
    dv = clamp_norm(np.asarray(v_desired, dtype=float) - v_prev, limits.max_accel * dt)
    return clamp_velocity(v_prev + dv, limits)


def time_scale_for_limits(peak_vel: float, peak_acc: float, peak_yaw_rate_dps: float, limits: Limits) -> float:
    """Factor f >= 1 by which a motion's duration must be multiplied so that it
    respects all limits. Velocity scales as 1/f and acceleration as 1/f^2."""
    f_v = peak_vel / limits.max_velocity if limits.max_velocity > 0 else 1.0
    f_a = math.sqrt(peak_acc / limits.max_accel) if limits.max_accel > 0 else 1.0
    f_w = peak_yaw_rate_dps / limits.max_yaw_rate if limits.max_yaw_rate > 0 else 1.0
    return max(1.0, f_v, f_a, f_w)
