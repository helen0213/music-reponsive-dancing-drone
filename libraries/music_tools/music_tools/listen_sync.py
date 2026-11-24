#!/usr/bin/env python3
"""Find where in a known clip the microphone is hearing (laptop side).

The music is played on a phone and only predefined clips are used, so the
alignment is a cross-correlation of the microphone's onset envelope against the
onset envelope of the reference clip.

    estimate_offset(recording, reference, sr, offset_range_s=None) -> (offset_s, confidence)
        offset_s: time in the recording at which the reference's t=0 occurs
                  (negative if the recording started mid-song).

CLI (needs ``sounddevice`` and a microphone):
    python -m music_tools.listen_sync clip.wav --seconds 4

TODO(tutorial/interface): delivering the result to the drone. The player expects
``show/start_in`` (std_msgs/Float64: seconds from now until song t=0; negative
if the song already started). How the laptop publishes into the drone's ROS 2
graph (macOS has no native ROS 2) is not defined yet, so the CLI only prints it.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

SR = 22050
HOP = 256  # ~11.6 ms per envelope frame


def _envelope(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    env = env - env.mean()
    return env


def estimate_offset(recording: np.ndarray, reference: np.ndarray, sr: int = SR, offset_range_s=None) -> tuple:
    """Normalized cross-correlation of onset envelopes.

    ``offset_range_s=(lo, hi)`` restricts the answer (seconds, position of the
    reference's t=0 in the recording). Use it: repetitive music is ambiguous
    over a few seconds, and the operator knows roughly when the song starts.
    Returns (offset_s, confidence in [-1, 1]).
    """
    from scipy.signal import correlate

    rec, ref = _envelope(recording, sr), _envelope(reference, sr)
    corr = correlate(rec, ref, mode="full", method="fft")
    lags = np.arange(len(corr)) - (len(ref) - 1)  # frames: rec[n + lag] ~ ref[n]

    # local energies over the overlapping part only (removes the loudness bias)
    cs_ref = np.concatenate([[0.0], np.cumsum(ref**2)])
    cs_rec = np.concatenate([[0.0], np.cumsum(rec**2)])
    lo_ref, hi_ref = np.maximum(0, -lags), np.minimum(len(ref), len(rec) - lags)
    overlap = hi_ref - lo_ref
    e_ref = cs_ref[np.clip(hi_ref, 0, len(ref))] - cs_ref[np.clip(lo_ref, 0, len(ref))]
    lo_rec, hi_rec = lo_ref + lags, hi_ref + lags
    e_rec = cs_rec[np.clip(hi_rec, 0, len(rec))] - cs_rec[np.clip(lo_rec, 0, len(rec))]
    with np.errstate(divide="ignore", invalid="ignore"):
        ncc = corr / np.sqrt(e_ref * e_rec)
    valid = (overlap >= 0.5 * min(len(rec), len(ref))) & np.isfinite(ncc)
    if offset_range_s is not None:
        secs = lags * HOP / sr
        valid &= (secs >= offset_range_s[0]) & (secs <= offset_range_s[1])
    if not valid.any():
        raise ValueError("no admissible alignment (check offset_range_s and recording length)")
    ncc = np.where(valid, ncc, -np.inf)
    k = int(np.argmax(ncc))
    frac = 0.0  # parabolic refinement around the peak
    if 0 < k < len(ncc) - 1 and np.isfinite(ncc[k - 1]) and np.isfinite(ncc[k + 1]):
        a_, b_, c_ = ncc[k - 1], ncc[k], ncc[k + 1]
        denom = a_ - 2 * b_ + c_
        frac = 0.5 * (a_ - c_) / denom if denom != 0 else 0.0
    return (lags[k] + frac) * HOP / sr, float(ncc[k])


def main(argv=None) -> int:
    import librosa
    import sounddevice as sd

    ap = argparse.ArgumentParser(description="Align the microphone with a known clip")
    ap.add_argument("reference", help="the clip being played")
    ap.add_argument("--seconds", type=float, default=4.0, help="how long to listen")
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-midsong", action="store_true", help="accept the song having started long before the recording")
    args = ap.parse_args(argv)

    ref, _ = librosa.load(args.reference, sr=SR, mono=True)
    print(f"listening for {args.seconds}s ...", file=sys.stderr)
    rec_start = time.time()
    rec = sd.rec(int(args.seconds * SR), samplerate=SR, channels=1, device=args.device, dtype="float32")
    sd.wait()
    rec = rec[:, 0]
    # by default assume the song's t=0 falls inside the recording (or <= 1 s before it)
    rng = None if args.allow_midsong else (-1.0, args.seconds)
    offset_s, conf = estimate_offset(rec, ref, SR, rng)
    song_t0_wall = rec_start + offset_s  # wall time at which the song's t=0 occurred
    start_in = song_t0_wall - time.time()  # negative: the song started that long ago
    print(f"offset_in_recording={offset_s:.3f}s confidence={conf:.3f} start_in={start_in:.3f}s")
    # TODO: publish start_in to show/start_in on the drone (interface not defined yet).
    return 0


if __name__ == "__main__":
    sys.exit(main())
