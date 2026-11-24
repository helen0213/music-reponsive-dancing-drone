#!/usr/bin/env python3
"""features.json -> schedule.json.

Plain script and importable module (no ROS). Usage:

    python generate.py features.json --limits assets/config/limits.yaml -o schedule.json

Mapping (see the constants below for the exact numbers)
-------------------------------------------------------
* beats            -> move timing. Each move is centred on a beat: its accent
                      (``t_peak``) lands on the beat and it spans +-duration/2.
* beat gap         -> move duration. If one beat gap is shorter than the
                      primitive's ``min_duration`` the move is stretched over
                      2 or 4 beats; if even 4 beats are too short it is rejected.
* RMS energy       -> amplitude, between the primitive's min and max amplitude.
* spectral centroid-> move type. Below the track's median centroid (bass-heavy)
                      pick the section palette's first pattern, above it
                      (treble-heavy) the second.
* sections         -> palette. Each distinct section label gets one pair of
                      patterns (PALETTES), in order of first appearance.
* reversals        -> if a move's outbound direction is opposite to the previous
                      move's, a ``hover`` of at least HOVER_S is inserted between
                      them; if there is no room the move is skipped.
* limits.yaml      -> amplitude is scaled down until velocity, acceleration and
                      yaw-rate limits hold; below the minimum amplitude, or if the
                      motion would leave the geofence / altitude band, the move is
                      rejected. Rejections are reported, never silently emitted.

Schedule entries: {"t_start", "t_peak", "primitive", "params", "duration"}
(seconds; amplitudes in metres, spin in degrees). Takeoff and landing are not
in the schedule; the player handles them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import drone_primitives  # noqa: F401  (installed / on PYTHONPATH)
except ImportError:  # running from a source checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "drone_primitives"))

from drone_primitives.primitives import AMPLITUDE_PARAM, AMPLITUDE_RANGE, build_motion
from drone_primitives.trajectory import Limits

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
PALETTES: List[Tuple[str, str]] = [  # (bass-heavy pick, treble-heavy pick) per section
    ("bounce", "side_step"),
    ("bounce", "circle"),
    ("side_step", "spin"),
    ("circle", "spin"),
]
STRETCH_BEATS = (1, 2, 4)  # allowed move lengths in beats
HOVER_S = 0.15  # minimum hover between moves that reverse direction
DEFAULT_MIN_DURATION_S = 0.5  # used if limits.yaml has no entry for a primitive
AMPLITUDE_SAFETY = 0.98  # scale-to-fit margin under the limits
REVERSAL_DOT = -0.5  # headings with cosine below this count as a reversal
TOL = 1e-6


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _curve(points, default: float):
    """Callable t -> value from a [[t, v], ...] list (linear interpolation)."""
    if not points:
        return lambda t: default
    arr = np.asarray(points, dtype=float)
    return lambda t: float(np.interp(t, arr[:, 0], arr[:, 1]))


def _section_index(sections, t: float, labels: Dict[str, int]) -> int:
    for start, end, label in sections:
        if start <= t < end:
            return labels[label]
    return labels[sections[-1][2]] if sections else 0


def _beat_gaps(beats: np.ndarray) -> np.ndarray:
    gaps = np.diff(beats)
    return np.append(gaps, gaps[-1])


def _amplitude(prim: str, rms: float) -> float:
    lo, hi = AMPLITUDE_RANGE[prim]
    return lo + float(np.clip(rms, 0.0, 1.0)) * (hi - lo)


def _make_params(prim: str, amplitude: float, sign: float) -> dict:
    key = AMPLITUDE_PARAM[prim]
    if prim == "circle":
        return {"radius": round(amplitude, 4), "direction": int(sign)}
    if prim == "spin":
        return {"angle_deg": round(sign * amplitude, 2)}
    return {key: round(sign * amplitude, 4)}


def _min_duration(limits: Limits, prim: str) -> float:
    return limits.min_duration.get(prim, DEFAULT_MIN_DURATION_S)


def _hits_envelope(peaks, base_altitude: float, limits: Limits) -> Optional[str]:
    """Reason string if a motion's swept extent leaves the geofence / altitude band."""
    lo, hi = peaks.offset_min, peaks.offset_max
    if hi[0] > limits.geofence_half_x + TOL or lo[0] < -limits.geofence_half_x - TOL:
        return "geofence x"
    if hi[1] > limits.geofence_half_y + TOL or lo[1] < -limits.geofence_half_y - TOL:
        return "geofence y"
    if base_altitude + hi[2] > limits.altitude_max + TOL or base_altitude + lo[2] < limits.altitude_min - TOL:
        return "altitude"
    return None


def _limit_factor(peaks, limits: Limits) -> float:
    """Amplitude divisor needed to satisfy the limits (velocity, acceleration and
    yaw rate all scale linearly with amplitude). <= 1 means already fine."""
    return max(
        peaks.velocity / limits.max_velocity,
        peaks.acceleration / limits.max_accel,
        peaks.yaw_rate_dps / limits.max_yaw_rate,
    )


# --------------------------------------------------------------------------- #
# Validation (independent of how the schedule was produced)
# --------------------------------------------------------------------------- #
def validate_schedule(schedule: List[dict], limits: Limits, base_altitude: Optional[float] = None) -> List[str]:
    """Return a list of human-readable violations (empty = valid)."""
    if base_altitude is None:
        base_altitude = 0.5 * (limits.altitude_min + limits.altitude_max)
    problems: List[str] = []
    prev_end = -np.inf
    for i, m in enumerate(schedule):
        name, dur = m["primitive"], float(m["duration"])
        where = f"#{i} {name}@{m['t_start']}"
        if m["t_start"] < -TOL:
            problems.append(f"{where}: starts before t=0")
        if m["t_start"] < prev_end - 1e-3:
            problems.append(f"{where}: overlaps the previous move")
        prev_end = m["t_start"] + dur
        if name != "hover" and dur < _min_duration(limits, name) - 1e-3:
            problems.append(f"{where}: duration {dur:.3f}s below min {_min_duration(limits, name):.3f}s")
        try:
            motion = build_motion(name, m.get("params"), dur)
        except (KeyError, ValueError) as exc:
            problems.append(f"{where}: {exc}")
            continue
        pk = motion.peaks()
        if pk.velocity > limits.max_velocity * 1.01:
            problems.append(f"{where}: velocity {pk.velocity:.2f} > {limits.max_velocity}")
        if pk.acceleration > limits.max_accel * 1.01:
            problems.append(f"{where}: acceleration {pk.acceleration:.2f} > {limits.max_accel}")
        if pk.yaw_rate_dps > limits.max_yaw_rate * 1.01:
            problems.append(f"{where}: yaw rate {pk.yaw_rate_dps:.0f} > {limits.max_yaw_rate}")
        why = _hits_envelope(pk, base_altitude, limits)
        if why:
            problems.append(f"{where}: {why}")
    return problems


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_schedule(
    features: dict, limits: Limits, base_altitude: Optional[float] = None
) -> Tuple[List[dict], List[dict]]:
    """Returns (schedule, rejected). ``rejected`` lists skipped moves with a reason."""
    if base_altitude is None:
        base_altitude = 0.5 * (limits.altitude_min + limits.altitude_max)
    beats = np.asarray(features.get("beats", []), dtype=float)
    if len(beats) < 2:
        return [], []
    gaps = _beat_gaps(beats)
    rms_at = _curve(features.get("rms"), 0.5)
    cent = features.get("centroid") or []
    cent_median = float(np.median([c[1] for c in cent])) if cent else 0.0
    cent_at = _curve(cent, cent_median)
    sections = features.get("sections") or []
    labels: Dict[str, int] = {}
    for _, _, lab in sections:
        labels.setdefault(lab, len(labels))

    schedule: List[dict] = []
    rejected: List[dict] = []
    prev_end = 0.0
    prev_heading: Optional[np.ndarray] = None
    flip: Dict[str, float] = {"side_step": 1.0, "circle": 1.0, "spin": 1.0, "bounce": 1.0}

    def reject(t, prim, why):
        rejected.append({"t": round(float(t), 4), "primitive": prim, "reason": why})

    for k, b in enumerate(beats):
        # --- what to do ---
        pal = PALETTES[_section_index(sections, b, labels) % len(PALETTES)]
        prim = pal[1] if cent_at(b) > cent_median else pal[0]

        # --- how long and how big ---
        # Try 1, 2 then 4 beats (never shorter than the primitive's min_duration):
        # amplitude follows the energy, scaled down to satisfy the limits; if the
        # amplitude would fall below its minimum, a longer stretch is tried.
        min_dur = _min_duration(limits, prim)
        amp_lo, _ = AMPLITUDE_RANGE[prim]
        sign = flip[prim]
        chosen = None
        why_not = f"beat gap {gaps[k]:.3f}s too short even over {STRETCH_BEATS[-1]} beats (min {min_dur}s)"
        for n_beats in STRETCH_BEATS:
            if n_beats * gaps[k] < min_dur - TOL:
                continue
            dur = n_beats * float(gaps[k])
            if float(b) - dur / 2.0 < 0.0:
                why_not = "would start before t=0"
                break
            amp = _amplitude(prim, rms_at(b))
            motion = build_motion(prim, _make_params(prim, amp, sign), dur)
            pk = motion.peaks()
            factor = _limit_factor(pk, limits)
            if factor > AMPLITUDE_SAFETY:
                amp = amp * AMPLITUDE_SAFETY / factor
                if amp < amp_lo:
                    why_not = f"limits require amplitude {amp:.3f} below minimum {amp_lo} (even over {n_beats} beats)"
                    continue
                motion = build_motion(prim, _make_params(prim, amp, sign), dur)
                pk = motion.peaks()
            envelope = _hits_envelope(pk, base_altitude, limits)
            if envelope:
                why_not = envelope
                continue
            chosen = (dur, amp, motion)
            break
        if chosen is None:
            reject(b, prim, why_not)
            continue
        dur, amp, motion = chosen
        t_start = float(b) - dur / 2.0

        # --- spacing: no overlap, hover between direction reversals ---
        heading = motion.heading
        reversal = (
            prev_heading is not None
            and np.linalg.norm(prev_heading) > 0
            and np.linalg.norm(heading) > 0
            and float(np.dot(prev_heading, heading)) < REVERSAL_DOT
        )
        earliest = prev_end + (HOVER_S if reversal else 0.0)
        if t_start < earliest - TOL:
            reject(b, prim, "overlaps previous move" if not reversal else "no room for reversal hover")
            continue
        if reversal and t_start - prev_end > TOL:
            hov = t_start - prev_end
            schedule.append(
                {
                    "t_start": round(prev_end, 4),
                    "t_peak": round(prev_end + hov / 2.0, 4),
                    "primitive": "hover",
                    "params": {},
                    "duration": round(hov, 4),
                }
            )

        schedule.append(
            {
                "t_start": round(t_start, 4),
                "t_peak": round(float(b), 4),
                "primitive": prim,
                "params": _make_params(prim, amp, sign),
                "duration": round(dur, 4),
            }
        )
        flip[prim] = -sign if prim in ("side_step", "circle", "spin") else sign
        prev_end = t_start + dur
        prev_heading = heading

    return schedule, rejected


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default_limits_path() -> Path:
    return Path(__file__).resolve().parents[3] / "assets" / "config" / "limits.yaml"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate schedule.json from features.json")
    ap.add_argument("features", help="features.json written by music_tools.analyze")
    ap.add_argument("--limits", default=str(_default_limits_path()), help="limits.yaml")
    ap.add_argument("-o", "--out", default="schedule.json", help="output schedule.json")
    ap.add_argument("--base-altitude", type=float, default=None, help="hover altitude the moves are relative to, m")
    args = ap.parse_args(argv)

    with open(args.features) as f:
        features = json.load(f)
    limits = Limits.from_yaml(args.limits)
    schedule, rejected = generate_schedule(features, limits, args.base_altitude)
    problems = validate_schedule(schedule, limits, args.base_altitude)
    if problems:  # should never happen; refuse to write an unsafe schedule
        print("Generated schedule violates limits:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    with open(args.out, "w") as f:
        json.dump(schedule, f, indent=2)
    n_moves = sum(1 for m in schedule if m["primitive"] != "hover")
    print(f"wrote {args.out}: {n_moves} moves, {len(rejected)} rejected", file=sys.stderr)
    for r in rejected:
        print(f"  rejected t={r['t']}s {r['primitive']}: {r['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
