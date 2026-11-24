#!/usr/bin/env python3
"""Extract musical features from an audio file with Librosa and write features.json.

    python -m music_tools.analyze clip.wav -o features.json

Output (see assets/schemas/features.example.json); all times in seconds:
    beats        beat times
    downbeats    estimated downbeats: every 4th beat, with the phase (0-3) whose
                 beats have the strongest onsets (assumes 4/4; an estimate only)
    tempo_curve  [[t, bpm], ...] local tempo from successive beat intervals
                 (median-filtered over 5 beats)
    onsets       onset times
    rms          [[t, 0-1], ...] RMS energy normalized by the clip's maximum
    centroid     [[t, Hz], ...] spectral centroid
    sections     [[start, end, "A"], ...] structural segments (agglomerative
                 clustering of MFCCs); similar segments share a label
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SR = 22050
HOP = 512
SERIES_HZ = 20.0  # sample rate of the rms / centroid curves
MAX_LABELS = 3
MIN_SECTION_S = 4.0  # shorter segments are merged into the previous one


def _decimate(times: np.ndarray, values: np.ndarray, hz: float) -> list:
    step = max(1, int(round((SR / HOP) / hz)))
    return [[round(float(t), 4), round(float(v), 4)] for t, v in zip(times[::step], values[::step])]


def _sections(y: np.ndarray, sr: int, duration: float) -> list:
    import librosa
    from scipy.cluster.hierarchy import fcluster, linkage

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-9)
    n_frames = mfcc.shape[1]
    k = int(np.clip(round(duration / 10.0), 2, 6))
    k = min(k, n_frames)
    bounds = librosa.segment.agglomerative(mfcc, k)
    edges = list(bounds) + [n_frames]
    edge_t = librosa.frames_to_time(edges, sr=sr, hop_length=HOP)
    edge_t[-1] = duration
    # drop boundaries that would create segments shorter than MIN_SECTION_S
    keep = [0]
    for i in range(1, len(edges) - 1):
        if edge_t[i] - edge_t[keep[-1]] >= MIN_SECTION_S and duration - edge_t[i] >= MIN_SECTION_S:
            keep.append(i)
    keep.append(len(edges) - 1)
    edges = [edges[i] for i in keep]
    edge_t = edge_t[keep]
    means = np.array([mfcc[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])])
    if len(means) > 1:
        raw = fcluster(linkage(means, "average"), t=min(MAX_LABELS, len(means)), criterion="maxclust")
    else:
        raw = np.array([1])
    names, out = {}, []
    for (start, end), r in zip(zip(edge_t[:-1], edge_t[1:]), raw):
        names.setdefault(int(r), chr(ord("A") + len(names)))
        out.append([round(float(start), 4), round(float(end), 4), names[int(r)]])
    return out


def analyze(path) -> dict:
    import librosa
    from scipy.signal import medfilt

    y, sr = librosa.load(str(path), sr=SR, mono=True)
    duration = len(y) / sr

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)

    # downbeats: assume 4/4, choose the phase whose beats carry the most onset energy
    if len(beats) >= 4:
        strength = onset_env[beat_frames]
        phase = int(np.argmax([strength[p::4].mean() for p in range(4)]))
        downbeats = beats[phase::4]
    else:
        downbeats = beats[:1]

    # local tempo from beat intervals
    if len(beats) >= 2:
        bpm = 60.0 / np.diff(beats)
        size = min(5, len(bpm) if len(bpm) % 2 == 1 else len(bpm) - 1)
        if size >= 3:
            bpm = medfilt(bpm, kernel_size=size)
        tempo_curve = [[round(float(t), 4), round(float(b), 2)] for t, b in zip(beats[1:], bpm)]
    else:
        tempo_curve = [[0.0, round(float(np.atleast_1d(tempo)[0]), 2)]]

    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP, units="time")

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms = rms / (rms.max() if rms.max() > 0 else 1.0)
    rms_t = librosa.times_like(rms, sr=sr, hop_length=HOP)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    cent_t = librosa.times_like(centroid, sr=sr, hop_length=HOP)

    return {
        "beats": [round(float(t), 4) for t in beats],
        "downbeats": [round(float(t), 4) for t in downbeats],
        "tempo_curve": tempo_curve,
        "onsets": [round(float(t), 4) for t in onsets],
        "rms": _decimate(rms_t, rms, SERIES_HZ),
        "centroid": _decimate(cent_t, centroid, SERIES_HZ),
        "sections": _sections(y, sr, duration),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audio file -> features.json")
    ap.add_argument("audio", help="audio file (wav/mp3/flac...)")
    ap.add_argument("-o", "--out", default=None, help="output path (default: <audio>.features.json)")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.audio).with_suffix(".features.json")
    features = analyze(args.audio)
    out.write_text(json.dumps(features))
    print(
        f"wrote {out}: {len(features['beats'])} beats, {len(features['onsets'])} onsets, "
        f"{len(features['sections'])} sections",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
