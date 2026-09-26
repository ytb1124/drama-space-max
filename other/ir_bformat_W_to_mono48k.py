#!/usr/bin/env python3
# ir_bformat_W_to_mono48k.py
# Extract W-channel (FOA B-format) to mono 48 kHz.
# Designed to be drop-in compatible with your existing Max runner:
# accepts --in/--out and also tolerates the 'hybrid' runner's extra flags.

import argparse, json, sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SCRIPT_TAG = "ir_bformat_W_to_mono48k (W-only, 48k)"

def peak_limit(sig: np.ndarray, limit: float = 0.98) -> np.ndarray:
    if sig.size == 0: 
        return sig
    p = float(np.max(np.abs(sig)))
    if np.isfinite(p) and p > 0 and limit > 0 and p > limit:
        sig = sig * (limit / p)
    return sig

def main():
    ap = argparse.ArgumentParser(description="Extract W channel from FOA B-format to mono 48 kHz")
    # Safe dest names to avoid reserved words
    ap.add_argument("--in",  dest="inp",  required=True, help="Input WAV absolute path")
    ap.add_argument("--out", dest="outp", required=True, help="Output WAV path (.wav)")
    ap.add_argument("--target_sr", type=int, default=48000)
    ap.add_argument("--peak_limit", type=float, default=0.98)
    ap.add_argument("--w_index", type=int, default=0, help="Index of W channel (0-based). FOA(AmbiX/FuMa) usually 0.")
    ap.add_argument("--gain_db", type=float, default=0.0, help="Optional gain (dB) applied to output")
    ap.add_argument("--trim_silence_db", type=float, default=None, help="Optional head/tail trim threshold (dBFS). None=off")
    # Accept & ignore runner's extra flags for compatibility
    ap.add_argument("--mode", default=None)
    ap.add_argument("--ch", type=int, default=None)
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--early_ms", type=float, default=None)
    ap.add_argument("--late_start_ms", type=float, default=None)
    ap.add_argument("--xfade_ms", type=float, default=None)
    ap.add_argument("--align_window_ms", type=float, default=None)
    ap.add_argument("--align_search_ms", type=float, default=None)

    args = ap.parse_args()

    # Read file
    try:
        X, sr = sf.read(args.inp, always_2d=True)
    except Exception as e:
        print(json.dumps({"error": f"Failed reading: {e}", "tag": SCRIPT_TAG}))
        return 1

    X = np.asarray(X, dtype=np.float64)
    ns, nch = X.shape

    # Pick W index (clamp)
    w_idx = int(max(0, min(args.w_index, max(0, nch-1))))
    y = X[:, w_idx].copy() if nch else np.zeros((0,), dtype=np.float64)

    # Optional trim
    if args.trim_silence_db is not None and y.size:
        thr = 10.0 ** (float(args.trim_silence_db) / 20.0)
        # find first/last above threshold on abs
        a = np.where(np.abs(y) >= thr)[0]
        if a.size:
            y = y[a[0]:a[-1]+1]

    # Resample to target_sr if needed
    if y.size and int(sr) != int(args.target_sr):
        y = resample_poly(y, int(args.target_sr), int(sr))
        sr_out = int(args.target_sr)
    else:
        sr_out = int(sr)

    # Gain dB, then peak limit
    if args.gain_db:
        y = y * (10.0 ** (float(args.gain_db)/20.0))
    y = peak_limit(y, float(args.peak_limit))

    # Ensure .wav extension
    out_path = str(args.outp)
    if not out_path.lower().endswith(".wav"):
        out_path += ".wav"

    try:
        sf.write(out_path, y.astype(np.float32), sr_out, subtype="FLOAT")
    except Exception as e:
        print(json.dumps({"error": f"Write failed: {e}", "tag": SCRIPT_TAG}))
        return 1

    info = {
        "tag": SCRIPT_TAG,
        "in_sr": int(sr),
        "in_channels": int(nch),
        "used_w_index": int(w_idx),
        "target_sr": int(sr_out),
        "out_len_samples": int(len(y)),
        "peak": float(np.max(np.abs(y))) if y.size else 0.0,
        "out": out_path
    }
    print(json.dumps(info, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
