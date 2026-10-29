"""Harmonic analysis of uniformly sampled waveforms covering whole fundamental cycles."""
from __future__ import annotations

import numpy as np


def harmonic_analysis(x: np.ndarray, dt: float, f1: float, n_harm: int = 50) -> dict:
    """Fundamental, THD (harmonics 2..n_harm) and the high-frequency residue.

    The record is truncated to an integer number of fundamental cycles.
    ``hf_rms``/``hf_pp`` describe everything above harmonic ``n_harm``
    (switching ripple), obtained by zeroing the low-order bins.
    """
    x = np.asarray(x, float)
    spc = 1.0 / (f1 * dt)                    # samples per cycle
    n_cyc = int(np.floor(len(x) / spc + 1e-6))
    if n_cyc < 1:
        raise ValueError("record shorter than one fundamental cycle")
    n = int(round(n_cyc * spc))
    x = x[-n:]
    spec = np.fft.rfft(x) / n
    amp = 2 * np.abs(spec)
    amp[0] = abs(spec[0])
    k1 = n_cyc
    fund = amp[k1]
    harm_bins = [k1 * h for h in range(2, n_harm + 1) if k1 * h < len(amp)]
    thd = np.sqrt(np.sum(amp[harm_bins] ** 2)) / fund if fund > 0 else np.nan
    hf = spec.copy()
    hf[: k1 * n_harm + 1] = 0
    x_hf = np.fft.irfft(hf * n, n)
    return {
        "fund_peak": float(fund),
        "fund_rms": float(fund / np.sqrt(2)),
        "thd_pct": float(100 * thd),
        "hf_rms": float(np.sqrt(np.mean(x_hf ** 2))),
        "hf_pp": float(x_hf.max() - x_hf.min()),
        "dc": float(spec[0].real),
    }
