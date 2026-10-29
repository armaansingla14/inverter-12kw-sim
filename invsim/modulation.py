"""Carrier-based modulation for a two-level three-phase leg set.

All schemes use symmetric regular sampling: the reference is sampled once per
carrier period (at the carrier valley) and each leg produces one centred
pulse per period. Duty ratios are in [0, 1] (fraction of the period with the
upper switch commanded on).

* SPWM   - sinusoidal references, linear up to m = 1
           (phase peak = m * Vdc / 2).
* SVPWM  - min-max zero-sequence injection, equivalent to centred
           space-vector modulation; linear up to m = 2/sqrt(3).
* THIPWM - 1/6 third-harmonic injection; linear up to m = 2/sqrt(3).
"""
from __future__ import annotations

import numpy as np

PHASE_SHIFT = np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])
M_MAX = {"spwm": 1.0, "svpwm": 2 / np.sqrt(3), "thipwm": 2 / np.sqrt(3)}


def references(theta: float, m: float, scheme: str = "svpwm") -> np.ndarray:
    """Normalized leg references in [-1, 1] for electrical angle ``theta``."""
    s = m * np.cos(theta + PHASE_SHIFT)
    if scheme == "spwm":
        return s
    if scheme == "svpwm":
        return s - 0.5 * (s.max() + s.min())
    if scheme == "thipwm":
        return s - (m / 6.0) * np.cos(3 * theta)
    raise ValueError(f"unknown scheme {scheme}")


def duties(theta: float, m: float, scheme: str = "svpwm") -> np.ndarray:
    return np.clip(0.5 * (1.0 + references(theta, m, scheme)), 0.0, 1.0)


def leg_edges(d: float, t_sw: float) -> tuple[float, float]:
    """Commanded on/off instants of a centred pulse within one period."""
    t_on = 0.5 * (1.0 - d) * t_sw
    t_off = 0.5 * (1.0 + d) * t_sw
    return t_on, t_off


def single_phase_duties(theta: float, m: float, unipolar: bool = True) -> np.ndarray:
    """Full-bridge (H-bridge) leg duties. Unipolar: legs driven by +/- reference."""
    s = m * np.cos(theta)
    if unipolar:
        return np.clip(0.5 * (1 + np.array([s, -s])), 0, 1)
    d = 0.5 * (1 + s)
    return np.clip(np.array([d, 1 - d]), 0, 1)
