"""Operating-condition sweeps for the inverter and boost stages, baseline vs improved.

Writes results/*.csv, results/summary.md and plots in results/plots/.
Usage: python scripts/run_sweeps.py [--quick]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from invsim import analytic  # noqa: E402
from invsim.dcdc import BoostOp, ripple_ccm_analytic  # noqa: E402
from invsim.dcdc import simulate as sim_boost  # noqa: E402
from invsim.designs import BOOST_BASELINE, BOOST_IMPROVED, INV_BASELINE, INV_IMPROVED  # noqa: E402
from invsim.inverter import OperatingPoint  # noqa: E402
from invsim.inverter import simulate as sim_inv  # noqa: E402

RES = ROOT / "results"
PLOTS = RES / "plots"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK2, GRID = "#52514e", "#e4e3df"


def write_csv(name, rows):
    RES.mkdir(exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(RES / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, (float, np.floating)):
                v = "nan" if math.isnan(v) else f"{v:.4g}"
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def style(ax, xl, yl, title):
    ax.set_xlabel(xl, color=INK2)
    ax.set_ylabel(yl, color=INK2)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=90)
    plt.close(fig)


def inv_run(design, op, cycles):
    r = sim_inv(design, op, warmup_cycles=cycles[0], measure_cycles=cycles[1])
    m = dict(r.metrics)
    r_load, _ = op.load_rl()
    ip = math.sqrt(2) * m["v_ph_rms"] / r_load * op.pf if op.pf == 1.0 else float("nan")
    if op.pf == 1.0:
        a = analytic.inverter_losses(design.device, ip, m["m"], 0.0, op.v_dc, design.fsw)
        m["analytic_p_cond_w"] = a["p_cond_w"]
        m["analytic_p_sw_w"] = a["p_sw_w"]
    return r, m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer points and cycles")
    args = ap.parse_args(argv)
    cycles = (3, 1) if args.quick else (4, 2)
    t_start = time.time()
    designs = [INV_BASELINE, INV_IMPROVED]

    # ---------------- inverter: load sweep ----------------
    loads = [0.25, 0.5, 1.0] if args.quick else [0.1, 0.25, 0.5, 0.75, 1.0, 1.1]
    load_rows, wave = [], {}
    for d in designs:
        for lf in loads:
            r, m = inv_run(d, OperatingPoint(p_out=12e3 * lf), cycles)
            load_rows.append({"sweep": "load", "load_frac": lf, **m})
            if lf == 1.0:
                wave[d.name] = r
            print(f"[inv load] {d.name} {lf:.2f}: eta={m['efficiency_pct']:.3f}% thd_v={m['thd_v_pct']:.3f}%", flush=True)
    write_csv("inverter_load_sweep.csv", load_rows)

    # ---------------- inverter: DC-link voltage sweep ----------------
    vdcs = [650, 700, 800] if args.quick else [620, 650, 700, 750, 800]
    vdc_rows = []
    for d in designs:
        for v in vdcs:
            _, m = inv_run(d, OperatingPoint(v_dc=v), cycles)
            vdc_rows.append({"sweep": "vdc", **m})
            print(f"[inv vdc] {d.name} {v}: eta={m['efficiency_pct']:.3f}% vph={m['v_ph_rms']:.1f} m={m['m']:.3f}", flush=True)
    write_csv("inverter_vdc_sweep.csv", vdc_rows)

    # ---------------- inverter: switching-frequency sweep ----------------
    fsw_sets = {INV_BASELINE.name: [5e3, 10e3, 20e3] if args.quick else [5e3, 10e3, 15e3, 20e3, 30e3],
                INV_IMPROVED.name: [20e3, 40e3, 80e3] if args.quick else [20e3, 30e3, 40e3, 60e3, 80e3]}
    fsw_rows = []
    for d in designs:
        for f in fsw_sets[d.name]:
            _, m = inv_run(replace(d, fsw=f), OperatingPoint(), cycles)
            fsw_rows.append({"sweep": "fsw", **m})
            print(f"[inv fsw] {d.name} {f/1e3:.0f}k: eta={m['efficiency_pct']:.3f}% ripple={m['il_ripple_pp_a']:.2f}A", flush=True)
    # same device at both frequencies with the same filter to separate device and filter effects
    for f in ([10e3, 40e3]):
        for d in designs:
            dd = replace(d, fsw=f, l_f=INV_IMPROVED.l_f, r_l=INV_IMPROVED.r_l, name=d.name + "_Lsmall")
            _, m = inv_run(dd, OperatingPoint(), cycles)
            fsw_rows.append({"sweep": "fsw_same_filter", **m})
    write_csv("inverter_fsw_sweep.csv", fsw_rows)

    # ---------------- inverter: modulation, dead time, power factor ----------------
    misc_rows = []
    for scheme in ("spwm", "svpwm"):
        for v in (650, 700):
            for d in designs:
                _, m = inv_run(replace(d, scheme=scheme, name=f"{d.name}|{scheme}"), OperatingPoint(v_dc=v), cycles)
                misc_rows.append({"sweep": "modulation", **m})
    for td in (0.5e-6, 1e-6, 2e-6):
        _, m = inv_run(replace(INV_BASELINE, deadtime=td, name=f"baseline_td{td*1e6:.1f}us"), OperatingPoint(), cycles)
        misc_rows.append({"sweep": "deadtime", "deadtime_us": td * 1e6, **m})
    for d in designs:
        _, m = inv_run(d, OperatingPoint(pf=0.8), cycles)
        misc_rows.append({"sweep": "pf0.8", **m})
    write_csv("inverter_misc.csv", misc_rows)

    # ---------------- boost: load x Vin sweep ----------------
    boost_rows, bwave = [], {}
    b_loads = [0.1, 0.5, 1.0] if args.quick else [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    vins = [350, 400, 450]
    for d in (BOOST_BASELINE, BOOST_IMPROVED):
        for vin in vins:
            for lf in b_loads:
                op = BoostOp(v_in=vin, p_out=12e3 * lf)
                r = sim_boost(d, op)
                m = dict(r.metrics)
                m["load_frac"] = lf
                m["il_ripple_analytic_ccm_a"] = ripple_ccm_analytic(vin, op.v_out, d.l, d.fsw)
                boost_rows.append(m)
                if vin == 400 and lf == 1.0:
                    bwave[d.name] = r
                print(f"[boost] {d.name} vin={vin} {lf:.2f}: eta={m['efficiency_pct']:.3f}% "
                      f"iin_rip={m['iin_ripple_pp_a']:.2f}A ccm={m['ccm']}", flush=True)
    write_csv("boost_sweep.csv", boost_rows)

    bf_rows = []
    for d, fs in ((BOOST_BASELINE, [10e3, 20e3, 40e3]), (BOOST_IMPROVED, [25e3, 50e3, 100e3])):
        for f in fs:
            r = sim_boost(replace(d, fsw=f), BoostOp())
            bf_rows.append(dict(r.metrics))
    write_csv("boost_fsw_sweep.csv", bf_rows)

    # ---------------- plots ----------------
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=90)
    for i, d in enumerate(designs):
        rows = [r for r in load_rows if r["design"] == d.name]
        ax.plot([r["p_out_kw"] for r in rows], [r["efficiency_pct"] for r in rows], "-o", ms=4,
                color=SERIES[i], lw=1.6, label=d.name)
    ax.legend(frameon=False, fontsize=8)
    style(ax, "output power (kW)", "efficiency (%)", "Inverter efficiency vs load (Vdc = 700 V, simulated)")
    save(fig, "inverter_efficiency.png")

    fig, axes = plt.subplots(2, 1, figsize=(5, 4.6), dpi=90, sharex=True)
    for i, d in enumerate(designs):
        r = wave[d.name]
        t0 = r.t[-1] - 0.02
        sel = r.t >= t0
        axes[0].plot((r.t[sel] - t0) * 1e3, r.i_l[sel, 0], color=SERIES[i], lw=1.0, label=d.name)
        axes[1].plot((r.t[sel] - t0) * 1e3, r.v_c[sel, 0], color=SERIES[i], lw=1.0, label=d.name)
    axes[0].legend(frameon=False, fontsize=7)
    style(axes[0], "", "i_L phase a (A)", "Inverter waveforms at 12 kW (one 50 Hz cycle)")
    style(axes[1], "time (ms)", "v_load phase a (V)", "")
    save(fig, "inverter_waveforms.png")

    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=90)
    for i, d in enumerate((BOOST_BASELINE, BOOST_IMPROVED)):
        rows = [r for r in boost_rows if r["design"] == d.name and r["v_in"] == 400]
        ax.plot([r["p_out_kw"] for r in rows], [r["efficiency_pct"] for r in rows], "-o", ms=4,
                color=SERIES[i], lw=1.6, label=d.name)
    ax.legend(frameon=False, fontsize=8)
    style(ax, "output power (kW)", "efficiency (%)", "Boost efficiency vs load (Vin = 400 V, simulated)")
    save(fig, "boost_efficiency.png")

    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=90)
    for i, d in enumerate((BOOST_BASELINE, BOOST_IMPROVED)):
        r = bwave[d.name]
        n = 4 * 16
        tt = (r.t[-n:] - r.t[-n]) * 1e6
        ax.plot(tt, r.i_l[-n:].sum(axis=1), color=SERIES[i], lw=1.4, label=f"{d.name} input current")
    ax.legend(frameon=False, fontsize=7)
    style(ax, "time (us)", "A", "Boost input current ripple at 12 kW, Vin = 400 V")
    save(fig, "boost_input_ripple.png")

    # ---------------- summary ----------------
    def pick(rows, name, **kw):
        for r in rows:
            if r["design"] == name and all(abs(r[k] - v) < 1e-9 if isinstance(v, float) else r[k] == v
                                           for k, v in kw.items()):
                return r
        return None

    rated = [pick(load_rows, d.name, load_frac=1.0) for d in designs]
    cmp_cols = ["design", "fsw_khz", "scheme", "p_out_kw", "efficiency_pct", "p_loss_w", "p_cond_w", "p_sw_w",
                "p_filter_w", "thd_v_pct", "thd_i_pct", "il_ripple_pp_a", "vdc_ripple_pp_v", "icap_rms_a",
                "balance_err_w"]
    b_rated = [pick(boost_rows, d.name, v_in=400, load_frac=1.0) for d in (BOOST_BASELINE, BOOST_IMPROVED)]
    b_cols = ["design", "n_phases", "fsw_khz", "l_uh", "p_out_kw", "efficiency_pct", "p_loss_w", "p_cond_w",
              "p_sw_w", "p_inductor_w", "il_ripple_pp_a", "il_ripple_analytic_ccm_a", "iin_ripple_pp_a",
              "iin_ripple_pct", "vout_ripple_pp_v", "balance_err_w"]
    text = [
        "# Simulated results", "",
        f"Generated by `python scripts/run_sweeps.py{' --quick' if args.quick else ''}` "
        f"(runtime {time.time() - t_start:.0f} s). All values are from the switching-level Python simulators "
        "with the generic device/magnetics parameters in `invsim/devices.py` and `invsim/designs.py`.", "",
        "## Inverter, rated point (12 kW, 400 V LL, 50 Hz, Vdc = 700 V, unity PF)", "",
        md_table(rated, cmp_cols), "",
        "Analytic (averaged) loss check at rated point:", "",
        md_table(rated, ["design", "p_cond_w", "analytic_p_cond_w", "p_sw_w", "analytic_p_sw_w"]), "",
        "## Inverter load sweep", "",
        md_table(load_rows, ["design", "load_frac", "p_out_kw", "efficiency_pct", "p_loss_w", "thd_i_pct", "thd_v_pct"]), "",
        "## Inverter DC-link sweep (rated load)", "",
        md_table(vdc_rows, ["design", "v_dc", "m", "v_ph_rms", "efficiency_pct", "thd_v_pct"]), "",
        "## Inverter switching-frequency sweep (rated load)", "",
        md_table(fsw_rows, ["sweep", "design", "fsw_khz", "efficiency_pct", "p_sw_w", "il_ripple_pp_a", "thd_v_pct", "v_hf_ripple_pct"]), "",
        "## Modulation, dead time and power factor", "",
        md_table(misc_rows, ["sweep", "design", "v_dc", "pf", "m", "v_ph_rms", "efficiency_pct", "thd_v_pct", "thd_i_pct"]), "",
        "## Boost stage, rated point (Vin = 400 V, Vout = 700 V, 12 kW)", "",
        md_table(b_rated, b_cols), "",
        "## Boost load x input-voltage sweep", "",
        md_table(boost_rows, ["design", "v_in", "load_frac", "p_out_kw", "efficiency_pct", "ccm", "il_ripple_pp_a", "iin_ripple_pp_a", "vout_ripple_pp_v"]), "",
        "## Boost switching-frequency sweep (rated)", "",
        md_table(bf_rows, ["design", "fsw_khz", "efficiency_pct", "p_sw_w", "il_ripple_pp_a", "iin_ripple_pp_a", "vout_ripple_pp_v"]), "",
    ]
    (RES / "summary.md").write_text("\n".join(text))
    print("\n".join(text[:12]))


if __name__ == "__main__":
    main()
