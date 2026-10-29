"""Switching-level simulator for an N-phase interleaved boost converter.

Per phase k::

    Vin --L_k (DCR)--+--[diode]--+-- Vout (Cout || R_load)
                     |
                  [switch]
                     |
                    GND

Each phase is in one of three states: switch on, diode conducting, or idle
(discontinuous conduction: inductor current zero, diode blocking). Device
drops are affine (V0 + r i), so every state combination is an LTI system
with a constant input and is integrated exactly between events. Inductor
current zero-crossings in DCM are located by bisection on the time quantum.

Control (updated once per switching period): an outer PI loop on the
period-averaged output voltage sets the inductor-current reference, and an
inner proportional loop per phase on the period-averaged inductor current
sets each phase's duty (average-current-mode control). The inner loop
damps the L-C resonance so the converter settles within the warm-up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .devices import Device, Diode
from .linsys import DiscreteLTI
from .spectrum import harmonic_analysis  # noqa: F401  (kept for API symmetry)

QUANTUM = 1e-9
ON, DIODE, IDLE = 0, 1, 2


@dataclass(frozen=True)
class BoostDesign:
    name: str
    switch: Device
    diode: Diode
    n_phases: int
    fsw: float              # per-phase switching frequency
    l: float                # per-phase inductance
    r_l: float              # per-phase DCR
    r_ac: float             # ripple-current equivalent resistance (AC copper + core)
    c_out: float
    r_esr: float = 5e-3     # output capacitor ESR (adds loss, ripple)


@dataclass(frozen=True)
class BoostOp:
    v_in: float = 400.0
    v_out: float = 700.0
    p_out: float = 12e3

    @property
    def r_load(self) -> float:
        return self.v_out ** 2 / self.p_out


@dataclass
class BoostResult:
    design: BoostDesign
    op: BoostOp
    t: np.ndarray
    i_l: np.ndarray
    v_out: np.ndarray
    metrics: dict = field(default_factory=dict)


class _Systems:
    """Cache of discretized systems keyed by the per-phase state tuple."""

    def __init__(self, d: BoostDesign, r_load: float):
        self.d = d
        self.r_load = r_load
        self.cache: dict[tuple, DiscreteLTI] = {}

    def get(self, states: tuple) -> DiscreteLTI:
        sysm = self.cache.get(states)
        if sysm is None:
            sysm = self._build(states)
            self.cache[states] = sysm
        return sysm

    def _build(self, states: tuple) -> DiscreteLTI:
        d = self.d
        n = d.n_phases
        a = np.zeros((n + 1, n + 1))
        # inputs: [V_in, 1] where "1" multiplies the constant knee voltages
        b = np.zeros((n + 1, 2))
        for k, s in enumerate(states):
            if s == ON:
                a[k, k] = -(d.r_l + d.switch.r_on) / d.l
                b[k, 0] = 1 / d.l
                b[k, 1] = -d.switch.v0 / d.l
            elif s == DIODE:
                a[k, k] = -(d.r_l + d.diode.rf) / d.l
                a[k, n] = -1 / d.l
                b[k, 0] = 1 / d.l
                b[k, 1] = -d.diode.vf0 / d.l
                a[n, k] = 1 / d.c_out
            else:
                a[k, k] = -1.0  # current pinned at zero (decays from 0 -> stays 0)
        a[n, n] = -1 / (self.r_load * d.c_out)
        return DiscreteLTI(a, b, QUANTUM)


def simulate(d: BoostDesign, op: BoostOp, warmup_s: float = 40e-3, measure_periods: int = 200,
             n_sub: int = 16, f_cross_v: float = 100.0) -> BoostResult:
    n = d.n_phases
    sysc = _Systems(d, op.r_load)
    t_sw_q = int(round(1 / d.fsw / QUANTUM))
    shift_q = [int(round(k * t_sw_q / n)) for k in range(n)]
    i_in0 = op.p_out / op.v_in
    x = np.array([i_in0 / n] * n + [op.v_out])
    d_ff = feedforward_duty(d, op)
    if d_ff < 1 - op.v_in / op.v_out - 1e-9:   # DCM expected: start with empty inductors
        x[:n] = 0.0
    duty = np.full(n, d_ff)
    # controller gains derived from the plant (see module docstring)
    k_i = 0.3 * d.l * d.fsw / op.v_out                    # duty per amp (per phase)
    w_v = 2 * np.pi * f_cross_v
    k_pv = w_v * op.v_out * d.c_out / op.v_in             # amps (total) per volt
    k_iv = k_pv * (w_v / 5) / d.fsw                       # amps per volt per period
    i_int = i_in0
    warmup_periods = int(round(warmup_s * d.fsw))
    rec_pts = sorted({int(round(j * t_sw_q / n_sub)) for j in range(1, n_sub + 1)})
    total = warmup_periods + measure_periods
    rec_t, rec_il, rec_v = [], [], []
    loss = dict(cond_sw=0.0, cond_d=0.0, e_on=0.0, e_off=0.0, e_rr=0.0, dcr=0.0, esr=0.0)
    e_in = 0.0
    vmin, vmax = np.inf, -np.inf
    u = np.array([op.v_in, 1.0])

    def phase_state(k, tq, on_len, il_k):
        tl = (tq - shift_q[k]) % t_sw_q
        if tl < on_len:
            return ON
        return DIODE if il_k > 0 else IDLE

    for per in range(total):
        measuring = per >= warmup_periods
        t0 = per * t_sw_q
        on_lens = [int(round(dk * t_sw_q)) for dk in duty]
        events = set(rec_pts)
        for k in range(n):
            for e in (shift_q[k], shift_q[k] + on_lens[k]):
                e %= t_sw_q
                if 0 < e < t_sw_q:
                    events.add(e)
        bounds = sorted(events)
        prev = 0
        v_acc = 0.0
        i_acc = np.zeros(n)
        for bnd in bounds:
            # switching events at the start of the sub-interval
            if measuring:
                for k in range(n):
                    tl = (prev - shift_q[k]) % t_sw_q
                    on_len = on_lens[k]
                    if tl == 0 and on_len > 0:          # turn-on: diode recovers
                        if x[k] > 0:
                            loss["e_on"] += d.switch.switch_energy(d.switch.e_on, x[k], x[n])
                            loss["e_rr"] += d.diode.rr_energy(x[k], x[n])
                    elif tl == on_len and on_len < t_sw_q:  # turn-off
                        loss["e_off"] += d.switch.switch_energy(d.switch.e_off, x[k], x[n])
            seg_start = prev
            while seg_start < bnd:
                st = tuple(phase_state(k, seg_start, on_lens[k], x[k]) for k in range(n))
                sysm = sysc.get(st)
                h = bnd - seg_start
                phi, gam = sysm.mats(h)
                xn = phi @ x + gam @ u
                # DCM: find the first diode-current zero crossing inside the interval
                cross = [k for k in range(n) if st[k] == DIODE and xn[k] < 0]
                if cross:
                    lo, hi = 0, h
                    while hi - lo > 1:
                        mid = (lo + hi) // 2
                        pm, gm = sysm.mats(mid)
                        xm = pm @ x + gm @ u
                        if min(xm[k] for k in cross) < 0:
                            hi = mid
                        else:
                            lo = mid
                    h = max(hi, 1)
                    phi, gam = sysm.mats(h)
                    xn = phi @ x + gam @ u
                    for k in cross:
                        xn[k] = 0.0
                for k in range(n):
                    if st[k] == IDLE:
                        xn[k] = 0.0
                if measuring:
                    hs = h * QUANTUM
                    for k in range(n):
                        i0, i1 = max(x[k], 0), max(xn[k], 0)
                        ia = 0.5 * (i0 + i1)
                        i2 = (i0 * i0 + i0 * i1 + i1 * i1) / 3
                        loss["dcr"] += d.r_l * i2 * hs
                        if st[k] == ON:
                            loss["cond_sw"] += (d.switch.v0 * ia + d.switch.r_on * i2) * hs
                        elif st[k] == DIODE:
                            loss["cond_d"] += (d.diode.vf0 * ia + d.diode.rf * i2) * hs
                    i_in_avg = 0.5 * (np.sum(np.maximum(x[:n], 0)) + np.sum(np.maximum(xn[:n], 0)))
                    e_in += op.v_in * i_in_avg * hs
                    vmin = min(vmin, xn[n])
                    vmax = max(vmax, xn[n])
                v_acc += 0.5 * (x[n] + xn[n]) * h
                i_acc += 0.5 * (x[:n] + xn[:n]) * h
                x = xn
                seg_start += h
            prev = bnd
            if bnd in rec_pts:
                rec_t.append((t0 + bnd) * QUANTUM)
                rec_il.append(x[:n].copy())
                rec_v.append(x[n])
        # outer voltage PI -> total current reference; inner per-phase current P
        v_avg = v_acc / t_sw_q
        i_avg = i_acc / t_sw_q
        err = op.v_out - v_avg
        i_int += k_iv * err
        i_ref = (i_int + k_pv * err) / n
        duty = np.clip(d_ff + k_i * (i_ref - i_avg), 0.0, 0.95)

    t = np.array(rec_t)
    il = np.array(rec_il)
    vo = np.array(rec_v)
    res = BoostResult(d, op, t, il, vo)
    t_meas = measure_periods * t_sw_q * QUANTUM
    m0 = len(t) - measure_periods * len(rec_pts)
    il_m, vo_m = il[m0:], vo[m0:]
    i_in = il_m.sum(axis=1)
    p_out = float(np.mean(vo_m ** 2) / op.r_load)
    ripple_l = il_m[:, 0] - il_m[:, 0].mean()
    i_l_rms_ripple = np.array([np.std(il_m[:, k]) for k in range(n)])
    p_lac = float(d.r_ac * np.sum(i_l_rms_ripple ** 2))
    # output capacitor ripple current: diode currents minus load current
    i_cap_rms = _cap_rms_estimate(t[m0:], vo_m, d.c_out)
    p_esr = d.r_esr * i_cap_rms ** 2
    p_cond = (loss["cond_sw"] + loss["cond_d"]) / t_meas
    p_sw = (loss["e_on"] + loss["e_off"] + loss["e_rr"]) / t_meas
    p_dcr = loss["dcr"] / t_meas
    p_loss = p_cond + p_sw + p_dcr + p_lac + p_esr
    p_in_circuit = e_in / t_meas
    res.metrics = {
        "design": d.name, "n_phases": n, "fsw_khz": d.fsw / 1e3, "l_uh": d.l * 1e6,
        "v_in": op.v_in, "v_out_target": op.v_out, "p_target_kw": op.p_out / 1e3,
        "v_out_avg": float(vo_m.mean()), "p_out_kw": p_out / 1e3, "duty": float(np.mean(duty)),
        "il_avg_a": float(il_m[:, 0].mean()),
        "il_ripple_pp_a": float(ripple_l.max() - ripple_l.min()),
        "iin_ripple_pp_a": float(i_in.max() - i_in.min()),
        "iin_ripple_pct": float(100 * (i_in.max() - i_in.min()) / i_in.mean()),
        "vout_ripple_pp_v": float(vmax - vmin),
        "ccm": bool(il_m.min() > 0),
        "p_cond_w": p_cond, "p_sw_w": p_sw, "p_on_w": loss["e_on"] / t_meas, "p_off_w": loss["e_off"] / t_meas,
        "p_rr_w": loss["e_rr"] / t_meas, "p_inductor_w": p_dcr + p_lac, "p_cap_esr_w": p_esr,
        "p_loss_w": p_loss, "efficiency_pct": 100 * p_out / (p_out + p_loss),
        "balance_err_w": p_in_circuit - p_out - p_cond - p_dcr,
    }
    return res


def feedforward_duty(d: BoostDesign, op: BoostOp) -> float:
    """Ideal duty: CCM value, or the DCM value if the converter is in DCM.

    DCM boost: D = sqrt(K M (M - 1)) with K = 2 L / (R_phase Ts), where each
    of the N phases carries 1/N of the load (R_phase = N R).
    """
    m = op.v_out / op.v_in
    d_ccm = 1 - 1 / m
    k = 2 * d.l * d.fsw / (op.r_load * d.n_phases)
    d_dcm = np.sqrt(k * m * (m - 1))
    return float(min(d_ccm, d_dcm))


def _cap_rms_estimate(t: np.ndarray, v: np.ndarray, c: float) -> float:
    """RMS capacitor current from C dv/dt on the recorded (uniform) samples."""
    if len(t) < 3:
        return 0.0
    i = c * np.gradient(v, t)
    return float(np.sqrt(np.mean((i - i.mean()) ** 2)))


def ripple_ccm_analytic(v_in: float, v_out: float, l: float, fsw: float) -> float:
    """Ideal CCM inductor peak-to-peak ripple: Vin * D / (L fsw)."""
    dd = 1 - v_in / v_out
    return v_in * dd / (l * fsw)
