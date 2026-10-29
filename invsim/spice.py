"""ngspice netlists for cross-checking the Python switching simulators.

Each converter leg is modelled as a switched pole: a behavioural voltage
source equal to the rail voltage selected by a (smoothed) natural-sampled
PWM gate signal, minus an on-resistance drop, with the matching behavioural
current drawn from the DC rail. This keeps ngspice robust (hard ``sw`` +
diode commutation at 12 kW repeatedly hit "timestep too small") while
still solving the full switched circuit: DC source impedance, DC-link
capacitor, LC filter and load. The Python simulators are run with a matched
device (equal forward/reverse resistance, no knee, no dead time) for the
comparison.

Decks run in batch mode, natively if ``ngspice`` is on PATH or else inside
the ``devtools`` docker image.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
IMAGE = os.environ.get("INVSIM_DOCKER_IMAGE", "devtools:latest")



def available() -> bool:
    if shutil.which("ngspice"):
        return True
    if shutil.which("docker") is None:
        return False
    r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return r.returncode == 0


def run(netlist: str, name: str, timeout: int = 900) -> dict[str, np.ndarray]:
    """Run one deck that writes ``{name}.dat`` with wrdata; return its columns."""
    work = Path(tempfile.mkdtemp(prefix="spice_", dir=REPO / "build" if (REPO / "build").exists() else None))
    try:
        (work / f"{name}.cir").write_text(netlist)
        if shutil.which("ngspice"):
            cmd, cwd = ["ngspice", "-b", f"{name}.cir"], work
        else:
            cmd = ["docker", "run", "--rm", "--memory", "1g", "--user", f"{os.getuid()}:{os.getgid()}",
                   "-v", f"{work}:/work", "-w", "/work", IMAGE, "ngspice", "-b", f"{name}.cir"]
            cwd = None
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        out = work / f"{name}.dat"
        if not out.exists():
            raise RuntimeError(f"ngspice produced no output:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
        with open(out) as fh:
            header = fh.readline().split()
            data = np.loadtxt(fh)
        return {h: data[:, i] for i, h in enumerate(header)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def inverter_deck(name: str, v_dc: float, fsw: float, m: float, f1: float, l_f: float, r_l: float,
                  c_f: float, r_load: float, c_dc: float, r_s: float, l_s: float, ron: float,
                  scheme: str = "svpwm", t_stop: float = 60e-3, t_step: float = 0.2e-6,
                  t_save: float = 0.02, edge_gain: float = 2000.0) -> str:
    """Three-phase two-level inverter, LC filter, star resistive load.

    Only the last ``t_save`` seconds are saved (tran tstart)."""
    tri = f"Vtri tri 0 pulse(-1 1 0 {0.5 / fsw:.6g} {0.5 / fsw:.6g} 1e-12 {1 / fsw:.6g})"
    if scheme == "svpwm":
        zs = "0.5*(max(v(sa),max(v(sb),v(sc)))+min(v(sa),min(v(sb),v(sc))))"
    elif scheme == "spwm":
        zs = "0"
    else:
        raise ValueError(scheme)
    lines = [
        f"* {name}: three-phase inverter cross-check deck (switched-pole model)",
        f"Vs src 0 dc {v_dc}",
        f"Rs src s1 {r_s}", f"Ls s1 s2 {l_s}", "Visrc s2 p dc 0",
        f"Cdc p 0 {c_dc} ic={v_dc}",
        tri,
        f"Bsa sa 0 v = {m}*cos(2*pi*{f1}*time)",
        f"Bsb sb 0 v = {m}*cos(2*pi*{f1}*time - 2*pi/3)",
        f"Bsc sc 0 v = {m}*cos(2*pi*{f1}*time + 2*pi/3)",
        f"Bz zs 0 v = {zs}",
    ]
    for ph in "abc":
        lines += [
            f"Bg{ph} g{ph} 0 v = 0.5*(1 + tanh({edge_gain}*(v(s{ph}) - v(zs) - v(tri))))",
            f"Bp{ph} o{ph} 0 v = v(p)*v(g{ph}) - {ron}*i(Vi{ph})",
            f"Vi{ph} o{ph} m{ph} dc 0",
            f"L{ph} m{ph} x{ph} {l_f} ic=0",
            f"RL{ph} x{ph} f{ph} {r_l}",
            f"C{ph} f{ph} n {c_f} ic=0",
            f"R{ph} f{ph} n {r_load}",
        ]
    lines += [
        "Bdc p 0 i = v(ga)*i(Via) + v(gb)*i(Vib) + v(gc)*i(Vic)",
        ".options method=gear reltol=1e-3 abstol=1e-6 vntol=1e-4",
        ".control",
        "set wr_vecnames", "set wr_singlescale",
        f"tran {t_step:.3g} {t_stop:.6g} {t_stop - t_save:.6g} {t_step:.3g} uic",
        "let vfa = v(fa) - v(n)", "let vfb = v(fb) - v(n)", "let isrc = i(Visrc)", "let ila = i(Via)",
        f"wrdata {name}.dat vfa vfb isrc v(p) ila",
        ".endc", ".end",
    ]
    return "\n".join(lines) + "\n"


def boost_deck(name: str, v_in: float, v_out0: float, duty: float, fsw: float, l: float, r_l: float,
               c_out: float, r_load: float, ron: float, n_phases: int = 1,
               t_stop: float = 30e-3, t_step: float = 0.1e-6, il0: float = 0.0,
               t_save: float = 1e-3) -> str:
    """Open-loop N-phase interleaved boost (CCM) with fixed duty, switched-pole model."""
    period = 1 / fsw
    lines = [
        f"* {name}: interleaved boost cross-check deck (switched-pole model)",
        f"Vin in 0 dc {v_in}",
        f"Cout out 0 {c_out} ic={v_out0}",
        f"Rload out 0 {r_load}",
    ]
    for k in range(n_phases):
        delay = k * period / n_phases
        lines += [
            f"Vg{k} g{k} 0 pulse(0 1 {delay:.6g} 10n 10n {duty * period - 10e-9:.6g} {period:.6g})",
            f"Vi{k} in m{k} dc 0",
            f"L{k} m{k} x{k} {l} ic={il0 / n_phases}",
            f"RL{k} x{k} sw{k} {r_l}",
            f"Bsw{k} sw{k} 0 v = (1 - v(g{k}))*v(out) + {ron}*i(Vi{k})",
            f"Bo{k} 0 out i = (1 - v(g{k}))*i(Vi{k})",
        ]
    lines += [
        ".options method=gear reltol=1e-3 abstol=1e-6 vntol=1e-4",
        ".control", "set wr_vecnames", "set wr_singlescale",
        f"tran {t_step:.3g} {t_stop:.6g} {t_stop - t_save:.6g} {t_step:.3g} uic",
        *[f"let il{k} = i(Vi{k})" for k in range(n_phases)],
        "wrdata " + name + ".dat v(out) " + " ".join(f"il{k}" for k in range(n_phases)),
        ".endc", ".end",
    ]
    return "\n".join(lines) + "\n"
