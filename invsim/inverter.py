"""Switching-level simulator for a two-level three-phase voltage-source inverter.

Circuit (per phase x in {a, b, c}, three-wire, common star point n)::

    DC source Vs --Rs--Ls--+-- DC link (Cdc) --[leg x]-- Lf (+ESR) --+-- vC_x --- load R + jX
                                                                     |
                                                                     Cf (star, to n)

* The legs are ideal switches with on-state drops taken from
  :mod:`invsim.devices` (knee + slope, diode or synchronous MOSFET channel),
  including dead time, during which the freewheeling diode conducts.
* Because the system is three-wire with identical phases, each phase sees
  u_x = v_xN - mean(v_aN, v_bN, v_cN) and the per-phase filter/load is the
  same LTI system, integrated exactly between switching events.
* The DC link is a second LTI system driven by the source voltage and the
  leg input current, which is held constant over each sub-interval.
* Switching energies are added at each hard-switched commutation using the
  instantaneous phase current and DC-link voltage.
* A slow outer loop adjusts the modulation index once per fundamental cycle
  so that the load phase voltage reaches its target amplitude.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .devices import Device
from .linsys import DiscreteLTI
from .modulation import M_MAX, duties
from .spectrum import harmonic_analysis

QUANTUM = 1e-9  # time base (s)


@dataclass(frozen=True)
class InverterDesign:
    name: str
    device: Device
    fsw: float
    l_f: float              # filter inductance per phase (H)
    r_l: float              # inductor DC resistance (ohm)
    r_ac: float             # equivalent resistance for ripple current (AC copper + core), ohm
    c_f: float              # filter capacitance per phase, star (F)
    c_dc: float = 470e-6
    scheme: str = "svpwm"
    deadtime: float | None = None   # default: device.deadtime

    @property
    def td(self) -> float:
        return self.device.deadtime if self.deadtime is None else self.deadtime


@dataclass(frozen=True)
class OperatingPoint:
    v_dc: float = 700.0
    p_out: float = 12e3          # rated active power at nominal voltage
    pf: float = 1.0              # load displacement power factor (lagging)
    v_ll_rms: float = 400.0
    f1: float = 50.0
    r_s: float = 0.05            # DC source resistance
    l_s: float = 20e-6           # DC source inductance

    @property
    def v_ph_peak(self) -> float:
        return self.v_ll_rms / np.sqrt(3) * np.sqrt(2)

    def load_rl(self) -> tuple[float, float]:
        v_ph = self.v_ll_rms / np.sqrt(3)
        s_ph = self.p_out / 3 / self.pf
        z = v_ph ** 2 / s_ph
        r = z * self.pf
        x = z * np.sqrt(max(0.0, 1 - self.pf ** 2))
        return r, x / (2 * np.pi * self.f1)


@dataclass
class LossTally:
    cond_sw: float = 0.0     # J, active-switch conduction (incl. MOSFET reverse channel)
    cond_diode: float = 0.0  # J, diode conduction
    e_on: float = 0.0
    e_off: float = 0.0
    e_rr: float = 0.0

    def total(self) -> float:
        return self.cond_sw + self.cond_diode + self.e_on + self.e_off + self.e_rr


@dataclass
class SimResult:
    design: InverterDesign
    op: OperatingPoint
    t: np.ndarray
    i_l: np.ndarray
    v_c: np.ndarray
    i_o: np.ndarray
    v_dc: np.ndarray
    m_final: float
    metrics: dict = field(default_factory=dict)


def _phase_system(d: InverterDesign, r_load: float, l_load: float) -> DiscreteLTI:
    lf, rl, cf = d.l_f, d.r_l, d.c_f
    if l_load > 0:
        a = np.array([[-rl / lf, -1 / lf, 0.0],
                      [1 / cf, 0.0, -1 / cf],
                      [0.0, 1 / l_load, -r_load / l_load]])
        b = np.array([[1 / lf], [0.0], [0.0]])
    else:
        a = np.array([[-rl / lf, -1 / lf],
                      [1 / cf, -1 / (r_load * cf)]])
        b = np.array([[1 / lf], [0.0]])
    return DiscreteLTI(a, b, QUANTUM)


def _dc_system(d: InverterDesign, op: OperatingPoint) -> DiscreteLTI:
    # x = [i_s, v_dc], u = [V_s, i_dc]
    a = np.array([[-op.r_s / op.l_s, -1 / op.l_s],
                  [1 / d.c_dc, 0.0]])
    b = np.array([[1 / op.l_s, 0.0],
                  [0.0, -1 / d.c_dc]])
    return DiscreteLTI(a, b, QUANTUM)


def simulate(design: InverterDesign, op: OperatingPoint, warmup_cycles: int = 5,
             measure_cycles: int = 2, n_sub: int = 8, m0: float | None = None,
             regulate: bool = True) -> SimResult:
    dev = design.device
    r_load, l_load = op.load_rl()
    ph = _phase_system(design, r_load, l_load)
    dc = _dc_system(design, op)
    nst = ph.n

    t_sw_q = int(round(1.0 / design.fsw / QUANTUM))
    td_q = int(round(design.td / QUANTUM))
    t1 = 1.0 / op.f1
    periods_per_cycle = int(round(t1 / (t_sw_q * QUANTUM)))
    n_cycles = warmup_cycles + measure_cycles
    rec_pts = [int(round(k * t_sw_q / n_sub)) for k in range(1, n_sub + 1)]
    rec_set = set(rec_pts)

    x = np.zeros((nst, 3))           # columns = phases
    xdc = np.array([0.0, op.v_dc])   # start with the link charged
    xdc[0] = 0.0
    m = m0 if m0 is not None else min(op.v_ph_peak / (op.v_dc / 2), M_MAX[design.scheme])
    w1 = 2 * np.pi * op.f1

    rec_n = n_cycles * periods_per_cycle * n_sub
    rec_t = np.empty(rec_n)
    rec_il = np.empty((rec_n, 3))
    rec_vc = np.empty((rec_n, 3))
    rec_io = np.empty((rec_n, 3))
    rec_vdc = np.empty(rec_n)
    rec_is = np.empty(rec_n)
    ri = 0

    tally = LossTally()
    meas_start_q = warmup_cycles * periods_per_cycle * t_sw_q
    e_rl = 0.0          # inductor DC copper energy (J) during measurement
    e_cap_i2 = 0.0      # integral of DC-link capacitor current squared
    vdc_min, vdc_max = np.inf, -np.inf
    e_in = 0.0          # source energy during measurement
    e_rs = 0.0

    def leg_state(k_q: int, edges) -> np.ndarray:
        """0 = lower on, 1 = upper on, 2 = dead time (both off)."""
        st = np.empty(3, dtype=np.int8)
        for p in range(3):
            on_c, off_c, mode = edges[p]
            if mode == 1:
                st[p] = 1
            elif mode == 0:
                st[p] = 0
            elif k_q < on_c or k_q >= off_c + td_q:
                st[p] = 0
            elif k_q < on_c + td_q:
                st[p] = 2
            elif k_q < off_c:
                st[p] = 1
            else:
                st[p] = 2
        return st

    def drops(st, il, vdc_now):
        """Return pole voltages v_xN, conduction powers (sw, diode) and upper-path flags."""
        v = np.empty(3)
        p_sw = 0.0
        p_d = 0.0
        upper = np.zeros(3, dtype=bool)
        for p in range(3):
            i = il[p]
            ia = abs(i)
            s = st[p]
            if s == 1:
                upper[p] = True
                if i >= 0:
                    vd = dev.fwd_drop(ia)
                    v[p] = vdc_now - vd
                    p_sw += vd * ia
                else:
                    vd = dev.rev_drop(ia, False)
                    v[p] = vdc_now + vd
                    if dev.sync_rect:
                        p_sw += vd * ia
                    else:
                        p_d += vd * ia
            elif s == 0:
                if i >= 0:
                    vd = dev.rev_drop(ia, False)
                    v[p] = -vd
                    if dev.sync_rect:
                        p_sw += vd * ia
                    else:
                        p_d += vd * ia
                else:
                    vd = dev.fwd_drop(ia)
                    v[p] = vd
                    p_sw += vd * ia
            else:  # dead time: diode by current direction
                vd = dev.rev_drop(ia, True)
                p_d += vd * ia
                if i >= 0:
                    v[p] = -vd
                else:
                    v[p] = vdc_now + vd
                    upper[p] = True
        return v, p_sw, p_d, upper

    k_q = 0  # absolute time in quanta
    m_hist = []
    for cyc in range(n_cycles):
        cyc_start_ri = ri
        for per in range(periods_per_cycle):
            t0 = k_q
            theta = w1 * (t0 + 0.5 * t_sw_q) * QUANTUM
            dvec = duties(theta, m, design.scheme)
            edges = []
            pts = set(rec_pts)
            for p in range(3):
                on_c = int(round(0.5 * (1 - dvec[p]) * t_sw_q))
                off_c = int(round(0.5 * (1 + dvec[p]) * t_sw_q))
                if off_c - on_c >= t_sw_q:
                    edges.append((0, t_sw_q, 1))      # clamped high for the whole period
                    continue
                if off_c <= on_c:
                    edges.append((0, 0, 0))           # clamped low
                    continue
                edges.append((on_c, off_c, 2))
                for e in (on_c, on_c + td_q, off_c, off_c + td_q):
                    if 0 < e < t_sw_q:
                        pts.add(e)
            bounds = sorted(pts)
            measuring = t0 >= meas_start_q
            # switching events at commanded edges (current at that instant)
            sw_events = {}
            for p in range(3):
                on_c, off_c, mode = edges[p]
                if mode == 2:
                    sw_events.setdefault(on_c, []).append((p, "on"))
                    sw_events.setdefault(off_c, []).append((p, "off"))
            prev = 0
            for b in bounds:
                if b == 0:
                    continue
                if prev in sw_events and measuring:
                    for p, kind in sw_events[prev]:
                        i = x[0, p]
                        vnow = xdc[1]
                        if kind == "on":
                            if i > 0:      # upper hard turn-on, lower diode recovers
                                tally.e_on += dev.switch_energy(dev.e_on, i, vnow)
                                tally.e_rr += dev.switch_energy(dev.e_rr, i, vnow)
                            else:          # lower switch hard turn-off
                                tally.e_off += dev.switch_energy(dev.e_off, -i, vnow)
                        else:
                            if i > 0:      # upper hard turn-off
                                tally.e_off += dev.switch_energy(dev.e_off, i, vnow)
                            else:          # lower hard turn-on, upper diode recovers
                                tally.e_on += dev.switch_energy(dev.e_on, -i, vnow)
                                tally.e_rr += dev.switch_energy(dev.e_rr, -i, vnow)
                st = leg_state(prev, edges)
                il0 = x[0].copy()
                vpole, p_sw0, p_d0, upper = drops(st, il0, xdc[1])
                u = vpole - vpole.mean()
                h = b - prev
                phi, gam = ph.mats(h)
                x = phi @ x + gam @ u[None, :]
                # leg input current averaged over the sub-interval (currents ramp)
                i_dc = float(0.5 * (np.sum(il0[upper]) + np.sum(x[0][upper])))
                phid, gamd = dc.mats(h)
                is0 = xdc[0]
                xdc = phid @ xdc + gamd @ np.array([op.v_dc, i_dc])
                if measuring:
                    hs = h * QUANTUM
                    # trapezoid with the same conducting devices at the end point
                    _, p_sw1, p_d1, _ = drops(st, x[0], xdc[1])
                    tally.cond_sw += 0.5 * (p_sw0 + p_sw1) * hs
                    tally.cond_diode += 0.5 * (p_d0 + p_d1) * hs
                    e_rl += design.r_l * 0.5 * (np.sum(il0 ** 2) + np.sum(x[0] ** 2)) * hs
                    is_avg = 0.5 * (is0 + xdc[0])
                    e_in += op.v_dc * is_avg * hs
                    e_rs += op.r_s * 0.5 * (is0 ** 2 + xdc[0] ** 2) * hs
                    ic0, ic1 = is0 - i_dc, xdc[0] - i_dc
                    e_cap_i2 += (ic0 ** 2 + ic0 * ic1 + ic1 ** 2) / 3 * hs
                    vdc_min = min(vdc_min, xdc[1])
                    vdc_max = max(vdc_max, xdc[1])
                prev = b
                _record = b in rec_set
                if _record and ri < rec_n:
                    rec_t[ri] = (t0 + b) * QUANTUM
                    rec_il[ri] = x[0]
                    rec_vc[ri] = x[1]
                    rec_io[ri] = x[2] if nst == 3 else x[1] / r_load
                    rec_vdc[ri] = xdc[1]
                    rec_is[ri] = xdc[0]
                    ri += 1
            k_q += t_sw_q
        # outer amplitude loop: fundamental of vC over this cycle
        if regulate and cyc < warmup_cycles:
            seg = rec_vc[cyc_start_ri:ri, 0]
            tt = rec_t[cyc_start_ri:ri]
            amp = 2 * abs(np.mean(seg * np.exp(-1j * w1 * tt)))
            if amp > 1e-3:
                m = float(np.clip(m * (1 + 0.8 * (op.v_ph_peak / amp - 1)), 0.05, M_MAX[design.scheme]))
            m_hist.append(m)

    res = SimResult(design, op, rec_t[:ri], rec_il[:ri], rec_vc[:ri], rec_io[:ri], rec_vdc[:ri], m)
    t_meas = measure_cycles * t1
    sel = res.t > meas_start_q * QUANTUM + 1e-12
    vc, io, il = res.v_c[sel], res.i_o[sel], res.i_l[sel]
    if nst == 3:
        p_out = float(np.mean(np.sum(vc * io, axis=1)))
    else:
        p_out = float(np.mean(np.sum(vc ** 2, axis=1)) / r_load)
    dt_rec = t_sw_q * QUANTUM / n_sub
    hv = harmonic_analysis(vc[:, 0], dt_rec, op.f1)
    hi = harmonic_analysis(io[:, 0], dt_rec, op.f1)
    hl = harmonic_analysis(il[:, 0], dt_rec, op.f1)
    ripple_rms = np.array([harmonic_analysis(il[:, p], dt_rec, op.f1)["hf_rms"] for p in range(3)])
    p_l_ac = float(design.r_ac * np.sum(ripple_rms ** 2))
    p_l_dc = e_rl / t_meas
    p_cond = (tally.cond_sw + tally.cond_diode) / t_meas
    p_swl = (tally.e_on + tally.e_off + tally.e_rr) / t_meas
    p_loss = p_cond + p_swl + p_l_dc + p_l_ac
    p_in_circuit = e_in / t_meas - e_rs / t_meas
    res.metrics = {
        "design": design.name, "device": dev.name, "fsw_khz": design.fsw / 1e3, "scheme": design.scheme,
        "v_dc": op.v_dc, "p_target_kw": op.p_out / 1e3, "pf": op.pf,
        "p_out_kw": p_out / 1e3, "m": m,
        "v_ph_rms": hv["fund_rms"], "thd_v_pct": hv["thd_pct"], "thd_i_pct": hi["thd_pct"],
        "v_hf_ripple_pct": 100 * hv["hf_rms"] / hv["fund_rms"],
        "il_ripple_rms_a": float(ripple_rms.mean()),
        "il_ripple_pp_a": float(hl["hf_pp"]),
        "p_cond_w": p_cond, "p_cond_sw_w": tally.cond_sw / t_meas, "p_cond_diode_w": tally.cond_diode / t_meas,
        "p_sw_w": p_swl, "p_on_w": tally.e_on / t_meas, "p_off_w": tally.e_off / t_meas,
        "p_rr_w": tally.e_rr / t_meas, "p_filter_w": p_l_dc + p_l_ac,
        "p_loss_w": p_loss, "efficiency_pct": 100 * p_out / (p_out + p_loss),
        # energy balance of the simulated circuit (switching energies are not in the circuit)
        "balance_err_w": p_in_circuit - p_out - p_cond - p_l_dc,
        "vdc_ripple_pp_v": float(vdc_max - vdc_min),
        "icap_rms_a": float(np.sqrt(e_cap_i2 / t_meas)),
    }
    return res


def with_fsw(d: InverterDesign, fsw: float) -> InverterDesign:
    return replace(d, fsw=fsw)
