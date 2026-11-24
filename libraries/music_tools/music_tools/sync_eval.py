#!/usr/bin/env python3
"""Evaluate how well the dance hit the beats.

    python -m music_tools.sync_eval --log logs/show_X.csv --telemetry telemetry.csv [--value-col z] [-o report.json]

Inputs
------
player log   CSV written by player_node (wall_time_s, song_time_s, event, primitive, beat_s, detail).
             The ``cmd_sent`` rows give the beat each move was aimed at (``beat_s``, song clock)
             and, with song_time_s, the song clock's origin in wall time.
telemetry    CSV with a ``wall_time_s`` column and one signal column (``--value-col``) recorded on the
             same clock as the player log. The signal needs one peak per accent, e.g. altitude for
             bounces or speed magnitude. Motion events are the peaks of |value - median|.

Metrics (ms): per-move offset = motion event time - beat time (positive = late), their mean,
mean absolute error (MAE), standard deviation (jitter; population std), and hit rate = share of
beats whose offset is within +-100 ms. A beat with no motion event within MATCH_WINDOW_S counts
as a miss.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

HIT_TOL_S = 0.100
MATCH_WINDOW_S = 0.300


def compute_sync_metrics(
    beat_times: Sequence[float],
    motion_times: Sequence[float],
    hit_tol_s: float = HIT_TOL_S,
    match_window_s: float = MATCH_WINDOW_S,
) -> Dict:
    """Match each beat to its nearest unused motion event within ``match_window_s``."""
    beats = np.sort(np.asarray(beat_times, dtype=float))
    events = list(np.sort(np.asarray(motion_times, dtype=float)))
    offsets_ms: List[float] = []
    per_move: List[dict] = []
    for b in beats:
        if events:
            j = int(np.argmin(np.abs(np.asarray(events) - b)))
            if abs(events[j] - b) <= match_window_s:
                off = (events.pop(j) - b) * 1000.0
                offsets_ms.append(off)
                per_move.append({"beat_s": float(b), "offset_ms": off})
                continue
        per_move.append({"beat_s": float(b), "offset_ms": None})
    off = np.asarray(offsets_ms, dtype=float)
    n_hit = int(np.sum(np.abs(off) <= hit_tol_s * 1000.0)) if len(off) else 0
    return {
        "n_beats": int(len(beats)),
        "n_matched": int(len(off)),
        "mean_offset_ms": float(off.mean()) if len(off) else None,
        "mean_abs_error_ms": float(np.abs(off).mean()) if len(off) else None,
        "std_ms": float(off.std()) if len(off) else None,
        "hit_rate": (n_hit / len(beats)) if len(beats) else None,
        "per_move": per_move,
    }


def detect_motion_events(
    times: Sequence[float], values: Sequence[float], min_prominence: float = 0.03, min_separation_s: float = 0.3
) -> np.ndarray:
    """Times of peaks of |value - median| (e.g. altitude excursions)."""
    from scipy.signal import find_peaks

    t = np.asarray(times, dtype=float)
    v = np.abs(np.asarray(values, dtype=float) - np.median(values))
    if len(t) < 3:
        return np.array([])
    dt = float(np.median(np.diff(t)))
    idx, _ = find_peaks(v, prominence=min_prominence, distance=max(1, int(min_separation_s / dt)))
    return t[idx]


def load_player_log(path) -> Tuple[float, List[float]]:
    """(song clock origin in wall time, beat targets of the non-hover commands)."""
    beats: List[float] = []
    t0_wall = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["event"] == "cmd_sent":
                if t0_wall is None and row["song_time_s"]:
                    t0_wall = float(row["wall_time_s"]) - float(row["song_time_s"])
                if row["primitive"] != "hover" and row["beat_s"]:
                    beats.append(float(row["beat_s"]))
    if t0_wall is None:
        raise ValueError("no cmd_sent rows in the player log")
    return t0_wall, beats


def load_telemetry(path, value_col: str) -> Tuple[np.ndarray, np.ndarray]:
    wall, val = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            wall.append(float(row["wall_time_s"]))
            val.append(float(row[value_col]))
    return np.asarray(wall), np.asarray(val)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Beat/motion synchronisation metrics")
    ap.add_argument("--log", required=True, help="player log CSV")
    ap.add_argument("--telemetry", required=True, help="telemetry CSV (wall_time_s + signal column)")
    ap.add_argument("--value-col", default="z")
    ap.add_argument("--min-prominence", type=float, default=0.03)
    ap.add_argument("-o", "--out", default=None, help="write the full report (with per-move offsets) as JSON")
    args = ap.parse_args(argv)

    t0_wall, beats = load_player_log(args.log)
    wall, val = load_telemetry(args.telemetry, args.value_col)
    events = detect_motion_events(wall - t0_wall, val, args.min_prominence)
    report = compute_sync_metrics(beats, events)
    summary = {k: v for k, v in report.items() if k != "per_move"}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
