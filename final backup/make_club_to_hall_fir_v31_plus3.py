#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_club_to_hall_fir_v31_plus3.py

This script builds on the v31_plus2 hybrid design and introduces an additional
"late_mix" parameter that allows you to adjust the amplitude of the late
reverberation tail relative to the early inverse section.  A higher
``--late_mix`` value (>1.0) will boost the late reverberation energy, while
values less than 1.0 will reduce it.  The goal is to allow you to dial in
more audible reverberation without sacrificing the early coherence in the
specified passband.

Features inherited from v31_plus2:

* Early section: stable inverse filter derived from v31 (regularised and
  smoothed), cross-faded to unity outside the passband.  Gating and
  windowing parameters control the extent of the early portion.
* Late section: ratio of the late parts of the hall and club IRs, with
  controllable regularisation and smoothing, and a choice of phase handling
  (keep complex phase or minimum-phase).  The late tail is faded in/out and
  appended after a cross-over time.
* Final alignment: optionally performs a small time shift of the FIR to
  maximise early coherence.
* Coherence reporting: computes coherence in octave-band windows for both
  the early (0–40 ms) and late (60–200 ms) portions of the convolved output.

The new ``--late_mix`` parameter multiplies the late tail before adding
it to the export, giving finer control over how strong the late reverberation
appears in the final filter.

Usage example:

```
python3 make_club_to_hall_fir_v31_plus3.py \
  --club CLUB.wav --hall HALL.wav --out OUT.wav \
  --mode mono \
  --band_low 125 --band_high 4000 \
  --transition_hz 900 \
  --reg_db -32 --smooth_bins 121 \
  --gate_db -33 --gate_pre_ms 4 --gate_tail_ms 16 --gate_fade_ms 5 \
  --early_ms 40 --xover_ms 16 --late_ms 300 \
  --reg_db_late -28 --smooth_bins_late 129 \
  --late_phase keep --late_mix 2.0 --final_align 1 \
  --peak_limit_ir 0.95 --export_full 1
```

This will produce a FIR that matches the club IR to the hall IR in the
125–4000 Hz passband, includes a 300 ms late tail (with complex phase
preserved), boosts the late reverberation energy by a factor of 2.0, and
aligns the filter for maximum early coherence.  Coherence metrics for the
early and late windows are reported at the end.
"""

import argparse, os, sys, math, json
import numpy as np

# Optional dependencies
HAVE_SF = False
HAVE_SCIPY = False
try:
    import soundfile as sf
    HAVE_SF = True
except Exception:
    pass
try:
    from scipy.io import wavfile as wavfile_sci
    from scipy import signal as spsig
    HAVE_SCIPY = True
except Exception:
    pass


def _print_json(d):
    sys.stdout.write(json.dumps(d, indent=2) + "\n")
    sys.stdout.flush()


def clean_path(p):
    p = (p or "").strip().strip('"').strip("'")
    return os.path.expanduser(p)


def load_wav_any(path):
    p = clean_path(path)
    last = None
    if HAVE_SF:
        try:
            x, sr = sf.read(p, always_2d=True)
            return sr, x.astype(np.float64)
        except Exception as e:
            last = e
    if HAVE_SCIPY:
        try:
            sr, x = wavfile_sci.read(p)
            if np.issubdtype(x.dtype, np.integer):
                x = x.astype(np.float64) / (np.iinfo(x.dtype).max)
            else:
                x = x.astype(np.float64)
            if x.ndim == 1:
                x = x[:, None]
            return sr, x
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed reading {p}: {last}")


def save_wav(path, sr, x):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    y = np.clip(x, -1.0, 1.0)
    if HAVE_SF:
        sf.write(path, y, sr, subtype="PCM_24")
    else:
        from scipy.io import wavfile as wavfile_sci2
        wavfile_sci2.write(path, sr, (y * 32767.0).astype(np.int16))


def next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


def quadratic_peak(y_minus, y0, y_plus):
    d = (y_minus - 2 * y0 + y_plus)
    return 0.0 if d == 0 else 0.5 * (y_minus - y_plus) / d


def gcc_phat_delay(a, b, n_fft=None, eps=1e-12):
    na, nb = len(a), len(b)
    L = n_fft or next_pow2(na + nb)
    A = np.fft.rfft(a, n=L)
    B = np.fft.rfft(b, n=L)
    R = A * np.conj(B)
    R /= (np.abs(R) + eps)
    cc = np.fft.irfft(R, n=L)
    cc = np.concatenate((cc[-(L // 2):], cc[:(L - L // 2)]))
    idx = int(np.argmax(cc))
    lag = idx - (L // 2)
    im = (idx - 1) % len(cc)
    ip = (idx + 1) % len(cc)
    delta = quadratic_peak(cc[im], cc[idx], cc[ip])
    return lag + delta


def fractional_advance_in_freq(N, d):
    k = np.arange(N // 2 + 1, dtype=np.float64)
    omega = 2.0 * np.pi * k / N
    return np.exp(1j * omega * d)


def align_b_to_a(A, B):
    N = min(A.shape[0], B.shape[0])
    A = A[:N, :]
    B = B[:N, :]
    C = min(A.shape[1], B.shape[1])
    out = np.zeros((N, C))
    delays = []
    Nfft = next_pow2(2 * N)
    for ch in range(C):
        d = gcc_phat_delay(A[:, ch], B[:, ch], n_fft=Nfft)
        Hadv = fractional_advance_in_freq(N, d)
        X = np.fft.rfft(B[:, ch], n=N)
        out[:, ch] = np.fft.irfft(X * Hadv, n=N)
        delays.append(d)
    return A[:, :C], out, delays


def build_gate(a, sr, thresh_db=-40.0, pre_ms=5.0, tail_ms=20.0, fade_ms=5.0):
    N = len(a)
    pre = int(sr * pre_ms / 1000.0)
    tail = int(sr * tail_ms / 1000.0)
    fade = max(1, int(sr * fade_ms / 1000.0))
    i0 = int(np.argmax(np.abs(a)))
    pk = float(np.max(np.abs(a)) + 1e-12)
    thr = pk * (10 ** (thresh_db / 20.0))
    j1 = N - 1
    for i in range(i0, N):
        if abs(a[i]) < thr:
            j1 = min(N - 1, i + tail)
            break
    j0 = max(0, i0 - pre)
    w = np.zeros(N)
    w[j0 : j1 + 1] = 1.0
    fi0 = max(0, j0 - fade)
    fi1 = j0
    fo0 = j1 + 1
    fo1 = min(N, j1 + 1 + fade)
    if fi1 > fi0:
        t = np.linspace(0, np.pi / 2, fi1 - fi0, endpoint=False)
        w[fi0:fi1] = np.sin(t) ** 2
    if fo1 > fo0:
        t = np.linspace(np.pi / 2, 0, fo1 - fo0, endpoint=False)
        w[fo0:fo1] = np.sin(t) ** 2
    return w


def moving_average_complex(H, win):
    if win <= 1:
        return H
    k = np.ones(win) / float(win)
    Hr = np.convolve(np.real(H), k, mode="same")
    Hi = np.convolve(np.imag(H), k, mode="same")
    return Hr + 1j * Hi


def crossfade_to_one(H, sr, low, high, trans):
    Nfull = (len(H) - 1) * 2
    f = (sr / Nfull) * np.arange(len(H))
    t = np.zeros_like(f)
    if low > 0 and trans > 0:
        lo0 = max(0.0, low - trans)
        lo1 = low
        idx = (f >= lo0) & (f < lo1)
        x = (f[idx] - lo0) / max(1e-12, (lo1 - lo0))
        t[idx] = 0.5 * (1 - np.cos(np.pi * x))
        t[f < lo0] = 1.0
    else:
        t[f < low] = 1.0
    if trans > 0:
        hi0 = high
        hi1 = high + trans
        idx = (f > hi0) & (f <= hi1)
        x = (f[idx] - hi0) / max(1e-12, (hi1 - hi0))
        t[idx] = 0.5 * (1 - np.cos(np.pi * (1 - x)))
        t[f > hi1] = 1.0
    else:
        t[f > high] = 1.0
    return (1.0 - t) * H + t * np.ones_like(H)


def band_geom_mean_gain(H, sr, low, high):
    Nfull = (len(H) - 1) * 2
    f = (sr / Nfull) * np.arange(len(H))
    band = (f >= low) & (f <= high)
    mag = np.maximum(np.abs(H[band]), 1e-12)
    return float(np.exp(np.mean(np.log(mag)))) if np.any(band) else 1.0


def minphase_from_mag(mag):
    mag = np.maximum(mag, 1e-12)
    logm = np.log(mag)
    cep = np.fft.irfft(logm)
    N = (len(mag) - 1) * 2
    cep[1 : N // 2] *= 2.0
    cep[N // 2 + 1 :] = 0.0
    logm_min = np.fft.rfft(cep)
    return np.exp(logm_min)


def welch_coherence(x, y, fs, nper=4096, band=(125, 4000)):
    if len(x) < nper or len(y) < nper:
        nper = min(len(x), len(y))
        if nper < 512:
            return np.nan
    nov = nper // 2
    try:
        from scipy import signal as spsig
        f, C = spsig.coherence(x, y, fs=fs, nperseg=nper, noverlap=nov, nfft=nper)
    except Exception:
        step = nper - nov
        win = np.hanning(nper)
        Sxy = Sxx = Syy = None
        k = 0
        for i in range(0, min(len(x), len(y)) - nper + 1, step):
            A = np.fft.rfft(x[i : i + nper] * win, n=nper)
            B = np.fft.rfft(y[i : i + nper] * win, n=nper)
            Sxy = A * np.conj(B) if Sxy is None else Sxy + A * np.conj(B)
            Sxx = A * np.conj(A) if Sxx is None else Sxx + A * np.conj(A)
            Syy = B * np.conj(B) if Syy is None else Syy + B * np.conj(B)
            k += 1
        if k == 0:
            return np.nan
        C = (np.abs(Sxy) ** 2) / (np.maximum(np.abs(Sxx) * np.abs(Syy), 1e-20))
        f = (fs / nper) * np.arange(len(C))
    m = (f >= band[0]) & (f <= band[1])
    return float(np.mean(C[m])) if np.any(m) else np.nan


def coherence_bands(x, y, fs, centers=(125, 250, 500, 1000, 2000, 4000)):
    out = {}
    for c in centers:
        band = (c / np.sqrt(2), c * np.sqrt(2))
        out[str(int(c))] = welch_coherence(x, y, fs, nper=4096, band=band)
    return out


def design_inverse(a_w, b_w, reg_db=-55.0, smooth_bins=33, gain_clip_db=18.0, passband_db=-50.0):
    N = len(a_w)
    Af = np.fft.rfft(a_w, n=N)
    Bf = np.fft.rfft(b_w, n=N)
    magB = np.abs(Bf)
    lam = (np.max(magB) * (10 ** (reg_db / 20.0))) ** 2
    H = (Af * np.conj(Bf)) / (np.abs(Bf) ** 2 + lam)
    Hs = moving_average_complex(H, int(smooth_bins))
    mag = np.abs(Hs)
    mag_db = np.clip(20 * np.log10(np.maximum(mag, 1e-12)), -gain_clip_db, gain_clip_db)
    Hlim = (10 ** (mag_db / 20.0)) * np.exp(1j * np.angle(Hs))
    b_db = 20 * np.log10(np.maximum(magB, 1e-12))
    pk = float(np.max(b_db))
    mask = (b_db >= pk + passband_db).astype(float)
    if len(mask) >= 5:
        mask = np.convolve(mask, np.ones(5) / 5.0, mode="same")
    return mask * Hlim + (1.0 - mask) * np.ones_like(Hlim)


def design_late_ratio(a_l, b_l, sr, reg_db=-28.0, smooth_bins=129, band_low=125, band_high=4000, transition_hz=900, max_nfft=262144, late_phase="keep"):
    N = len(a_l)
    Nfft = 1
    Ndes = min(N, max_nfft)
    while (Nfft << 1) <= Ndes:
        Nfft <<= 1
    Nfft = max(8192, Nfft)
    Af = np.fft.rfft(a_l, n=Nfft)
    Bf = np.fft.rfft(b_l, n=Nfft)
    magB = np.abs(Bf)
    lam = (np.max(magB) * (10 ** (reg_db / 20.0))) ** 2
    H = (Af * np.conj(Bf)) / (np.abs(Bf) ** 2 + lam)
    Hs = moving_average_complex(H, int(max(3, smooth_bins)))
    Hsb = crossfade_to_one(Hs, sr, band_low, band_high, transition_hz)
    if late_phase == "min":
        mag = np.abs(Hsb)
        Huse = minphase_from_mag(mag)
    else:
        Huse = Hsb
    h = np.fft.irfft(Huse, n=Nfft)
    return h


def linear_conv_first(x, h, out_len):
    Lh = len(h)
    Lx = min(len(x), out_len)
    n = next_pow2(Lh + Lx - 1)
    X = np.fft.rfft(x[:Lx], n=n)
    H = np.fft.rfft(h, n=n)
    y = np.fft.irfft(X * H, n=n)[:out_len]
    if len(y) < out_len:
        y = np.pad(y, (0, out_len - len(y)))
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--club", required=True)
    ap.add_argument("--hall", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="mono", choices=["mono", "stereo"])
    ap.add_argument("--band_low", type=float, default=125.0)
    ap.add_argument("--band_high", type=float, default=4000.0)
    ap.add_argument("--transition_hz", type=float, default=900.0)
    ap.add_argument("--reg_db", type=float, default=-32.0)
    ap.add_argument("--smooth_bins", type=int, default=121)
    ap.add_argument("--gain_clip_db", type=float, default=10.0)
    ap.add_argument("--passband_db", type=float, default=-35.0)
    ap.add_argument("--gate_db", type=float, default=-33.0)
    ap.add_argument("--gate_pre_ms", type=float, default=4.0)
    ap.add_argument("--gate_tail_ms", type=float, default=16.0)
    ap.add_argument("--gate_fade_ms", type=float, default=5.0)
    ap.add_argument("--early_ms", type=float, default=40.0)
    ap.add_argument("--xover_ms", type=float, default=16.0)
    ap.add_argument("--late_ms", type=float, default=200.0)
    ap.add_argument("--reg_db_late", type=float, default=-28.0)
    ap.add_argument("--smooth_bins_late", type=int, default=129)
    ap.add_argument("--max_nfft", type=int, default=262144)
    ap.add_argument("--late_phase", choices=["keep", "min"], default="keep")
    ap.add_argument("--late_mix", type=float, default=1.0, help="Multiply late tail amplitude by this factor")
    ap.add_argument("--final_align", type=int, default=1)
    ap.add_argument("--peak_limit_ir", type=float, default=0.95)
    ap.add_argument("--export_full", type=int, default=1)
    args = ap.parse_args()

    srA, A = load_wav_any(args.hall)
    srB, B = load_wav_any(args.club)
    # Resample hall to club SR
    if srA != srB:
        if HAVE_SCIPY:
            g = math.gcd(srA, srB)
            up = srB // g
            down = srA // g
            A = spsig.resample_poly(A, up, down, axis=0)
        else:
            N = A.shape[0]
            M = int(round(N * (srB / srA)))
            X = np.fft.rfft(A, n=N, axis=0)
            Y = np.zeros((M // 2 + 1, A.shape[1]), dtype=np.complex128)
            L = min(len(X), len(Y))
            Y[:L] = X[:L]
            A = np.fft.irfft(Y, n=M, axis=0)
        sr = srB
    else:
        sr = srA

    def to_mono(x):
        return x[:, 0:1] if x.ndim == 2 and x.shape[1] == 1 else np.mean(x, axis=1, keepdims=True)
    if args.mode == "mono":
        if A.shape[1] != 1:
            A = to_mono(A)
        if B.shape[1] != 1:
            B = to_mono(B)

    A_use, B_al, _ = align_b_to_a(A, B)
    N, C = A_use.shape

    # Early inverse filter
    gates = np.stack([
        build_gate(A_use[:, ch], sr, args.gate_db, args.gate_pre_ms, args.gate_tail_ms, args.gate_fade_ms)
        for ch in range(C)
    ], axis=1)
    H = np.zeros((N // 2 + 1, C), dtype=np.complex128)
    for ch in range(C):
        a_w = A_use[:, ch] * gates[:, ch]
        b_w = B_al[:, ch] * gates[:, ch]
        Hc = design_inverse(a_w, b_w, args.reg_db, args.smooth_bins, args.gain_clip_db, args.passband_db)
        H[:, ch] = crossfade_to_one(Hc, sr, args.band_low, args.band_high, args.transition_hz)
    mags = [band_geom_mean_gain(H[:, ch], sr, args.band_low, args.band_high) for ch in range(C)]
    beta = float(np.exp(np.mean(np.log(np.maximum(mags, 1e-12)))))
    h_early = np.fft.irfft(H / beta, n=N, axis=0)

    pk_common = max(int(np.argmax(np.abs(h_early[:, ch]))) for ch in range(C))
    Learly = max(1, int(round(args.early_ms * sr / 1000.0)))
    hend_len = pk_common + Learly
    hend = np.zeros((hend_len, C))
    for ch in range(C):
        end = min(N, hend_len)
        seg = h_early[pk_common:end, ch].copy()
        if len(seg) > 1:
            w = np.hanning(len(seg) * 2)[: len(seg)]
            hend[pk_common:end, ch] = seg * w

    # Late tail design
    x0 = max(pk_common, int(round(args.xover_ms * sr / 1000.0)))
    Llate = max(1, int(round(args.late_ms * sr / 1000.0)))
    fade = max(1, int(round(0.005 * sr)))  # 5 ms fade
    a_l = np.zeros(N)
    b_l = np.zeros(N)
    a_l[x0:] = A_use[: N - x0, 0]
    b_l[x0:] = B_al[: N - x0, 0]
    h_tail_long = design_late_ratio(
        a_l,
        b_l,
        sr,
        args.reg_db_late,
        args.smooth_bins_late,
        args.band_low,
        args.band_high,
        args.transition_hz,
        args.max_nfft,
        late_phase=args.late_phase,
    )

    # Adjust tail to fade-in position if using keep phase
    seg_len = Llate + 2 * fade
    if len(h_tail_long) < seg_len:
        h_tail_long = np.pad(h_tail_long, (0, seg_len - len(h_tail_long)))
    if args.late_phase == "keep":
        idx_peak = int(np.argmax(np.abs(h_tail_long)))
        shift = max(0, fade - idx_peak)
        if shift > 0:
            h_tail_long = np.pad(h_tail_long, (shift, 0))[: len(h_tail_long)]
    # Extract segment and apply fades
    seg = h_tail_long[: seg_len].copy()
    if fade > 0:
        fi = np.linspace(0, np.pi / 2, fade, endpoint=False)
        fo = np.linspace(np.pi / 2, 0, fade, endpoint=False)
        seg[:fade] *= np.sin(fi) ** 2
        seg[-fade:] *= np.sin(fo) ** 2
    # Apply late_mix scaling
    seg *= args.late_mix

    end_len = max(hend_len, x0 + seg_len)
    export = np.zeros((end_len, C))
    export[: hend.shape[0], :] += hend
    for ch in range(C):
        e = export[:, ch]
        Ladd = min(seg_len, end_len - x0)
        if Ladd > 0:
            e[x0 : x0 + Ladd] += seg[:Ladd]

    # Final micro alignment
    if args.final_align:
        prev_len = min(len(B_al[:, 0]) + len(export[:, 0]) - 1, max(hend_len, x0 + Llate))
        y_prev = linear_conv_first(B_al[:, 0], export[:, 0], out_len=prev_len)
        a_ref = A_use[: len(y_prev), 0]
        d = gcc_phat_delay(a_ref[: int(sr * 0.05)], y_prev[: int(sr * 0.05)])
        max_shift = int(sr * 0.01)  # 10 ms
        s = int(np.clip(round(-d), -max_shift, max_shift))
        if s != 0:
            if s > 0:
                export = np.pad(export, ((s, 0), (0, 0)))[: len(export), :]
            else:
                s2 = -s
                export = np.pad(export, ((0, s2), (0, 0)))[s2:, :]

    # Peak limiting
    pk = float(np.max(np.abs(export)))
    if args.peak_limit_ir > 0 and pk > args.peak_limit_ir:
        export *= (args.peak_limit_ir / pk)

    # Save (trim if export_full=0)
    out_path = clean_path(args.out)
    if not args.export_full:
        export = export[: hend_len, :]
    save_wav(out_path, sr, export)

    # Metrics
    centers = [125, 250, 500, 1000, 2000, 4000]
    prev_len = min(len(B_al[:, 0]) + len(export[:, 0]) - 1, max(hend_len, x0 + Llate))
    y_prev = linear_conv_first(B_al[:, 0], export[:, 0], out_len=prev_len)
    # Early window (0–40 ms)
    i0 = int(np.argmax(np.abs(A_use[:, 0])))
    wE0, wE1 = i0, min(len(y_prev), i0 + int(round(0.04 * sr)))
    wL0, wL1 = i0 + int(round(0.06 * sr)), min(len(y_prev), i0 + int(round(0.20 * sr)))
    aa = A_use[wE0:wE1, 0]
    bb = y_prev[wE0:wE1]
    if len(aa) > 8 and len(bb) > 8:
        aa = aa - np.mean(aa)
        bb = bb - np.mean(bb)
        corr = float(np.sum(aa * bb) / (np.sqrt(np.sum(aa ** 2)) * np.sqrt(np.sum(bb ** 2)) + 1e-20))
    else:
        corr = float("nan")
    def coh_bands(x, y, fs, centers=(125, 250, 500, 1000, 2000, 4000)):
        out = {}
        for c in centers:
            band = (c / np.sqrt(2), c * np.sqrt(2))
            out[str(int(c))] = welch_coherence(x, y, fs, nper=4096, band=band)
        return out
    coh_early = coh_bands(A_use[wE0:wE1, 0], y_prev[wE0:wE1], sr, centers=centers) if wE1 > wE0 else {}
    coh_late = coh_bands(A_use[wL0:wL1, 0], y_prev[wL0:wL1], sr, centers=centers) if wL1 > wL0 else {}
    _print_json(
        {
            "sr": sr,
            "channels": int(export.shape[1]),
            "out": out_path,
            "peak": float(np.max(np.abs(export))),
            "length_samples": int(len(export)),
            "CorrCoef_early40ms": corr,
            "CohBands_Early_0_40ms": coh_early,
            "CohBands_Late_60_200ms": coh_late,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _print_json({"error": repr(e)})
