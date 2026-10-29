"""Cross-check the Python switching simulators against ngspice.

Both simulators run the same circuit with a matched, simplified device
(equal forward/reverse on-resistance, no knee voltage, no dead time, no
switching-energy model) and a fixed modulation index / duty, so the
comparison isolates the circuit solution: output voltage, power, distortion
and ripple. ngspice uses natural-sampled PWM, Python regular-sampled PWM.

Writes results/spice_crosscheck.csv and .md.
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from invsim import spice  # noqa: E402
from invsim.dcdc import BoostOp  # noqa: E402
from invsim.dcdc import simulate as sim_boost  # noqa: E402
from invsim.designs import BOOST_BASELINE, INV_BASELINE  # noqa: E402
from invsim.devices import Device, Diode  # noqa: E402
from invsim.inverter import OperatingPoint  # noqa: E402
from invsim.inverter import simulate as sim_inv  # noqa: E402
from invsim.spectrum import harmonic_analysis  # noqa: E402

RON = 0.03


def matched_device() -> Device:
    return Device("matched", "igbt", v0=0.0, r_on=RON, vf0=0.0, rf=RON,
                  e_on=0.0, e_off=0.0, e_rr=0.0, deadtime=0.0)


def uniform(t, *ys, dt):
    tu = np.arange(t[0], t[-1], dt)
    return (tu,) + tuple(np.interp(tu, t, y) for y in ys)


def inverter_case(fsw: float, m: float, scheme: str):
    op = OperatingPoint(r_s=0.05, l_s=20e-6)
    d = replace(INV_BASELINE, device=matched_device(), fsw=fsw, scheme=scheme, deadtime=0.0, r_ac=0.0,
                name=f"xchk_{scheme}_{int(fsw/1e3)}k")
    r_load, _ = op.load_rl()
    t0 = time.time()
    py = sim_inv(d, op, warmup_cycles=4, measure_cycles=1, m0=m, regulate=False)
    t_py = time.time() - t0
    deck = spice.inverter_deck("inv", op.v_dc, fsw, m, op.f1, d.l_f, d.r_l, d.c_f, r_load, d.c_dc,
                               op.r_s, op.l_s, RON, scheme, t_stop=0.1, t_step=0.2e-6)
    t0 = time.time()
    ng = spice.run(deck, "inv")
    t_ng = time.time() - t0
    dt = 0.2e-6
    t, va, vb, ia = uniform(ng["time"], ng["vfa"], ng["vfb"], ng["ila"], dt=dt)
    hv = harmonic_analysis(va, dt, op.f1)
    p_ng = float(np.mean(va ** 2 + vb ** 2) / r_load * 1.5)  # balanced three-phase: 3 x mean(v^2)/R
    il = harmonic_analysis(ia, dt, op.f1)
    rows = []
    for name, pyv, ngv in (("v_ph_rms_V", py.metrics["v_ph_rms"], hv["fund_rms"]),
                           ("p_out_kW", py.metrics["p_out_kw"], p_ng / 1e3),
                           ("thd_v_pct", py.metrics["thd_v_pct"], hv["thd_pct"]),
                           ("il_ripple_pp_A", py.metrics["il_ripple_pp_a"], il["hf_pp"])):
        rows.append({"case": d.name, "quantity": name, "python": pyv, "ngspice": ngv,
                     "diff_pct": 100 * (pyv - ngv) / ngv if ngv else float("nan"),
                     "t_python_s": t_py, "t_ngspice_s": t_ng})
    return rows


def boost_case():
    d = replace(BOOST_BASELINE, switch=matched_device(), diode=Diode("matched", 0.0, RON, 0.0), r_ac=0.0)
    op = BoostOp()
    t0 = time.time()
    py = sim_boost(d, op)
    t_py = time.time() - t0
    duty = py.metrics["duty"]
    deck = spice.boost_deck("boost", op.v_in, op.v_out, duty, d.fsw, d.l, d.r_l, d.c_out, op.r_load,
                            RON, n_phases=1, t_stop=60e-3, t_step=0.1e-6, il0=op.p_out / op.v_in)
    t0 = time.time()
    ng = spice.run(deck, "boost")
    t_ng = time.time() - t0
    t = ng["time"]
    sel = t >= t[-1] - 10 / d.fsw
    vo = ng["v(out)"][sel]
    il = ng["il0"][sel]
    rows = []
    for name, pyv, ngv in (("v_out_avg_V", py.metrics["v_out_avg"], float(vo.mean())),
                           ("il_avg_A", py.metrics["il_avg_a"], float(il.mean())),
                           ("il_ripple_pp_A", py.metrics["il_ripple_pp_a"], float(il.max() - il.min())),
                           ("vout_ripple_pp_V", py.metrics["vout_ripple_pp_v"], float(vo.max() - vo.min()))):
        rows.append({"case": f"boost_duty_{duty:.4f}", "quantity": name, "python": pyv, "ngspice": ngv,
                     "diff_pct": 100 * (pyv - ngv) / ngv, "t_python_s": t_py, "t_ngspice_s": t_ng})
    return rows


def main():
    if not spice.available():
        print("ngspice not available; skipping")
        return
    rows = []
    rows += inverter_case(10e3, 0.9, "svpwm")
    rows += inverter_case(10e3, 0.9, "spwm")
    rows += boost_case()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "spice_crosscheck.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    lines = ["# Python vs ngspice cross-check", "",
             "Matched simplified device (30 mOhm in both directions), no dead time, fixed modulation; "
             "ngspice uses a behavioural switched-pole leg model.",
             "Python uses regular-sampled PWM; ngspice uses natural (continuous) sampling.", "",
             "| case | quantity | python | ngspice | diff % |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['quantity']} | {r['python']:.4g} | {r['ngspice']:.4g} | {r['diff_pct']:.2f} |")
    (out / "spice_crosscheck.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
