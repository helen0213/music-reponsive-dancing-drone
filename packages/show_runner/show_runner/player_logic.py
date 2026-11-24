"""Pure helpers for the player (no ROS): schedule loading and send-time planning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class Latency:
    """Values from assets/config/latency.yaml (all milliseconds, measured later)."""

    audio_output_ms: float = 0.0  # delay between song clock t=0 and what the audience hears (+ = later)
    link_ms: float = 0.0  # player -> drone command transport
    response_ms: Dict[str, float] = field(default_factory=dict)  # per primitive: command -> visible motion

    @classmethod
    def from_yaml(cls, path) -> "Latency":
        with open(Path(path), "r") as f:
            d = yaml.safe_load(f) or {}
        return cls(
            audio_output_ms=float(d.get("audio_output_ms", 0.0)),
            link_ms=float(d.get("link_ms", 0.0)),
            response_ms={k: float(v) for k, v in (d.get("response_ms") or {}).items()},
        )

    def lead_s(self, primitive: str) -> float:
        """Seconds BEFORE the scheduled start that the command must be sent
        (negative = later). send = t_start - link - response + audio_output."""
        return (self.link_ms + self.response_ms.get(primitive, 0.0) - self.audio_output_ms) / 1000.0


@dataclass
class PlannedCommand:
    send_time: float  # song-clock seconds at which to publish
    t_start: float
    t_peak: float
    primitive: str
    params: dict
    duration: float


def load_schedule(path) -> List[dict]:
    with open(Path(path), "r") as f:
        schedule = json.load(f)
    if not isinstance(schedule, list):
        raise ValueError("schedule.json must be a list of moves")
    for i, m in enumerate(schedule):
        for key in ("t_start", "primitive", "duration"):
            if key not in m:
                raise ValueError(f"schedule entry {i} is missing '{key}'")
    return schedule


def build_send_plan(schedule: List[dict], latency: Latency) -> List[PlannedCommand]:
    """Commands sorted by send time, each early by its latency lead."""
    plan = [
        PlannedCommand(
            send_time=float(m["t_start"]) - latency.lead_s(m["primitive"]),
            t_start=float(m["t_start"]),
            t_peak=float(m.get("t_peak", m["t_start"] + m["duration"] / 2.0)),
            primitive=m["primitive"],
            params=dict(m.get("params") or {}),
            duration=float(m["duration"]),
        )
        for m in schedule
    ]
    plan.sort(key=lambda c: c.send_time)
    return plan


def schedule_end_time(schedule: List[dict]) -> float:
    return max((float(m["t_start"]) + float(m["duration"]) for m in schedule), default=0.0)
