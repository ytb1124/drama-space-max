#!/usr/bin/env python3
# ir_4ch_to_mono48k_hybrid_v2.py
# Multi-channel IR → mono 48 kHz (avg / first / ch / weighted / hybrid)
# hybrid: Early(0~early_ms) = best single channel, Late(late_start_ms~) = aligned average,
#         smooth crossfade between the two. Prints a single JSON line on success.
# NOTE: argparse uses dest="inp"/"outp" to avoid reserved-word issues.

import argparse, json, sys
from typing import Optional, Sequence, Tuple
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SCRIPT_TAG = "ir_4ch_to_mono48k_hybrid_v2 (inp/outp-safe)"

# ------------------------- small utils -------------------------

def ms_to_samples(ms: float, sr: int) -> int:
    return int(round(ms * 1e-3 * sr))

def peak_limit(sig: np.ndarray, limit: float = 0.98) -> np.ndarray:
    if not (isinstance(limit, (int, float)) and limit > 0):
        return sig
    if sig.size == 0:
        return sig
    p = float(np.max(np.abs(sig)))
    if p > limit:
        sig = sig * (limit / p)
    return sig

def find_arrival_idx(x: np.ndarray, sr: int, search_ms: float = 100.0) -> int:
    """Rough direct arrival: max of lightly smoothed |x| in the first search_ms."""
    n = min(len(x), ms_to_samples(search_ms, sr))
    if n <= 0:
        return 0
    seg = np.abs(x[:n])
    if n >= 3:
        seg = (np.roll(seg, 1) + seg + np.roll(seg, -1)) / 3.0
    return int(np.argmax(seg))

# --------------------- channel selection/align ------------------

def best_channel_by_metric(X: np.ndarray, sr: int,
                           early_ms: float, late_start_ms: float) -> int:
    """Pick channel maximizing score = early_peak * DRR (simple, robust)."""
    nch = X.shape[1]
    eN = ms_to_samples(early_ms, sr)
    l0 = ms_to_samples(late_start_ms, sr)
    scores = []
    for ch in range(nch):
        x = X[:, ch]
        idx = find_arrival_idx(x, sr)
        a0 = max(0, idx - eN // 4)
        a1 = min(len(x), idx + eN - eN // 4)
        early = x[a0:a1]
        late  = x[l0:l0 + ms_to_samples(200.0, sr)]
        e_energy = float(np.sum(early ** 2)) + 1e-12
        l_energy = float(np.sum(late  ** 2)) + 1e-12
        e_peak   = float(np.max(np.abs(early))) if early.size else 0.0
        drr      = e_energy / l_energy
        score    = e_peak * drr
        scores.append(score)
    return int(np.argmax(scores)) if scores else 0

def align_to_ref(X: np.ndarray, sr: int, ref_ch: int,
                 window_ms: float = 16.0, search_ms: float = 8.0) -> np.ndarray:
    """Align channels to ref by maximizing dot product in early window (integer shift)."""
    Y = np.zeros_like(X)
    ref = X[:, ref_ch]
    idx_ref = find_arrival_idx(ref, sr)
    half_win = ms_to_samples(window_ms, sr) // 2
    ref_a = max(0, idx_ref - half_win)
    ref_b = min(len(ref), idx_ref + half_win)
    ref_seg = ref[ref_a:ref_b]
    if ref_seg.size == 0:
        return X.copy()

    max_shift = ms_to_samples(search_ms, sr)
    for ch in range(X.shape[1]):
        x = X[:, ch]
        if ch == ref_ch:
            Y[:, ch] = x
            continue
        best_lag = 0
        best_val = -1e30
        for lag in range(-max_shift, max_shift + 1):
            a = ref_a + lag
            b = a + ref_seg.size
            if a < 0 or b > len(x):
                continue
            val = float(np.dot(ref_seg, x[a:b]))
            if val > best_val:
                best_val = val
                best_lag = lag
        # apply integer shift
        if best_lag == 0:
            Y[:, ch] = x
        elif best_lag > 0:
            Y[best_lag:, ch] = x[:len(x)-best_lag]
            Y[:best_lag, ch] = 0.0
        else:
            lag = -best_lag
            Y[:len(x)-lag, ch] = x[lag:]
            Y[len(x)-lag:, ch] = 0.0
    return Y

# -------------------------- downmixers --------------------------

def to_mono_basic(X: np.ndarray, mode: str, ch: int = 0,
                  weights: Optional[Sequence[float]] = None) -> np.ndarray:
    ns, nch = X.shape
    if nch == 1:
        return X[:, 0]
    if mode == "first":
        return X[:, 0]
    if mode == "ch":
        ch = max(0, min(int(ch), nch-1))
        return X[:, ch]
    if mode == "weighted":
        if weights is None or len(weights) != nch:
            raise ValueError("weights length must match channel count")
        w = np.asarray(weights, dtype=np.float64)
        if np.allclose(w.sum(), 0.0):
            raise ValueError("weights sum is zero")
        w = w / np.sum(np.abs(w))
        return (X * w).sum(axis=1)
    # avg
    return X.mean(axis=1)

def hybrid_downmix(X: np.ndarray, sr: int,
                   early_ms: float = 40.0, late_start_ms: float = 60.0, xfade_ms: float = 10.0,
                   align_window_ms: float = 16.0, align_search_ms: float = 8.0) -> Tuple[np.ndarray, int]:
    """Early: best channel, Late: aligned average. Smooth crossfade."""
    ns, nch = X.shape
    if nch == 1:
        return X[:, 0].copy(), 0

    best_ch = best_channel_by_metric(X, sr, early_ms, late_start_ms)
    Xal = align_to_ref(X, sr, best_ch, window_ms=align_window_ms, search_ms=align_search_ms)
    y_best = Xal[:, best_ch]
    y_avg  = Xal.mean(axis=1)

    eN = max(0, min(ns, ms_to_samples(early_ms, sr)))
    sN = max(eN, min(ns, ms_to_samples(late_start_ms, sr)))
    xN = max(1, min(ns, ms_to_samples(xfade_ms, sr)))

    w = np.zeros(ns, dtype=np.float64)
    if eN > 0:
        w[:eN] = 1.0
    if sN > eN:
        w[eN:sN] = np.linspace(1.0, 0.0, sN - eN, endpoint=False)
    if ns >= 3:
        w = (np.roll(w,1)+w+np.roll(w,-1)) / 3.0

    y = w * y_best + (1.0 - w) * y_avg
    return y, best_ch

# ------------------------------ main ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Multi-channel IR → mono @ 48kHz (avg/first/ch/weighted/hybrid)",
        add_help=True
    )
    # Use safe dest names
    ap.add_argument("--in",  dest="inp",  required=True, help="Input WAV absolute path")
    ap.add_argument("--out", dest="outp", required=True, help="Output WAV path (.wav)")

    ap.add_argument("--mode", choices=["avg","first","ch","weighted","hybrid"], default="hybrid")
    ap.add_argument("--ch", type=int, default=0, help="mode=ch: channel index (0-based)")
    ap.add_argument("--weights", type=str, help="mode=weighted: e.g., 1,1,0.5,0.5")

    ap.add_argument("--target_sr", type=int, default=48000)
    ap.add_argument("--peak_limit", type=float, default=0.98)

    # hybrid/align params
    ap.add_argument("--early_ms", type=float, default=40.0)
    ap.add_argument("--late_start_ms", type=float, default=60.0)
    ap.add_argument("--xfade_ms", type=float, default=10.0)
    ap.add_argument("--align_window_ms", type=float, default=16.0)
    ap.add_argument("--align_search_ms", type=float, default=8.0)

    args = ap.parse_args()

    # Read
    try:
        x, sr = sf.read(args.inp, always_2d=True)  # (nsamp, nch)
    except Exception as e:
        print(json.dumps({"error": f"Failed reading: {e}", "tag": SCRIPT_TAG}))
        return 1

    x = np.asarray(x, dtype=np.float64)
    ns, nch = x.shape

    try:
        if args.mode == "hybrid":
            y, best_ch = hybrid_downmix(
                x, sr,
                early_ms=args.early_ms,
                late_start_ms=args.late_start_ms,
                xfade_ms=args.xfade_ms,
                align_window_ms=args.align_window_ms,
                align_search_ms=args.align_search_ms
            )
        else:
            weights = None
            if args.mode == "weighted" and args.weights:
                weights = [float(v) for v in str(args.weights).split(",")]
            y = to_mono_basic(x, args.mode, ch=args.ch, weights=weights)
            best_ch = 0 if nch == 1 else None
    except Exception as e:
        print(json.dumps({"error": f"Downmix failed: {e}", "tag": SCRIPT_TAG}))
        return 1

    # Resample
    try:
        if sr != args.target_sr and y.size:
            y = resample_poly(y, args.target_sr, sr)
            sr_out = int(args.target_sr)
        else:
            sr_out = int(sr)
    except Exception as e:
        print(json.dumps({"error": f"Resample failed: {e}", "tag": SCRIPT_TAG}))
        return 1

    # Peak limit & write
    try:
        y = peak_limit(y.astype(np.float64), args.peak_limit)
        out_path = args.outp
        if not out_path.lower().endswith(".wav"):
            out_path += ".wav"
        sf.write(out_path, y, sr_out, subtype="FLOAT")
    except Exception as e:
        print(json.dumps({"error": f"Write failed: {e}", "tag": SCRIPT_TAG}))
        return 1

    info = {
        "tag": SCRIPT_TAG,
        "in_sr": int(sr),
        "in_channels": int(nch),
        "mode": args.mode,
        "selected_best_ch": (int(best_ch) if best_ch is not None else None),
        "early_ms": float(args.early_ms),
        "late_start_ms": float(args.late_start_ms),
        "target_sr": int(sr_out),
        "out_len_samples": int(len(y)),
        "peak": float(np.max(np.abs(y))) if y.size else 0.0,
        "out": out_path
    }
    print(json.dumps(info, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
