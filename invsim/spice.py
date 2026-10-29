"""ngspice netlists for cross-checking the Python switching simulators.

The netlists use voltage-controlled switches (``sw`` model, on-resistance
Ron) with antiparallel/freewheeling diodes and natural-sampled PWM produced
by behavioural comparators. They are run in batch mode, natively if
``ngspice`` is on PATH or else inside the ``devtools`` docker image.
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

SW_MODEL = ".model swm sw(vt=0.5 vh=0.05 ron={ron} roff=1e7)"
D_MODEL = ".model dfw d(is=1e-12 n=1 rs={rs} cjo=1p)"


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
                  rs_diode: float, scheme: str = "svpwm", t_stop: float = 60e-3,
                  t_step: float = 0.2e-6) -> str:
    """Three-phase two-level inverter, LC filter, star resistive load."""
    tri = (f"Vtri tri 0 pulse(-1 1 0 {0.5 / fsw:.6g} {0.5 / fsw:.6g} 1e-12 {1 / fsw:.6g})")
    if scheme == "svpwm":
        zs = "0.5*(max(v(sa),max(v(sb),v(sc)))+min(v(sa),min(v(sb),v(sc))))"
    elif scheme == "spwm":
        zs = "0"
    else:
        raise ValueError(scheme)
    lines = [
        f"* {name}: three-phase inverter cross-check deck",
        SW_MODEL.format(ron=ron), D_MODEL.format(rs=rs_diode),
        f"Vs src 0 dc {v_dc}",
        f"Rs src s1 {r_s}", f"Ls s1 p {l_s}",
        f"Cdc p 0 {c_dc} ic={v_dc}",
        tri,
        f"Bsa sa 0 v = {m}*cos(2*pi*{f1}*time)",
        f"Bsb sb 0 v = {m}*cos(2*pi*{f1}*time - 2*pi/3)",
        f"Bsc sc 0 v = {m}*cos(2*pi*{f1}*time + 2*pi/3)",
        f"Bz zs 0 v = {zs}",
    ]
    for ph in "abc":
        lines += [
            f"Bg{ph} g{ph} 0 v = (v(s{ph}) - v(zs)) > v(tri) ? 1 : 0",
            f"Bgn{ph} gn{ph} 0 v = 1 - v(g{ph})",
            f"Su{ph} p o{ph} g{ph} 0 swm",
            f"Sl{ph} o{ph} 0 gn{ph} 0 swm",
            f"Du{ph} o{ph} p dfw",
            f"Dl{ph} 0 o{ph} dfw",
            f"L{ph} o{ph} x{ph} {l_f} ic=0",
            f"RL{ph} x{ph} f{ph} {r_l}",
            f"C{ph} f{ph} n {c_f} ic=0",
            f"R{ph} f{ph} n {r_load}",
        ]
    lines += [
        ".options method=gear reltol=1e-4 abstol=1e-6 vntol=1e-5",
        ".control",
        "set wr_vecnames", "set wr_singlescale",
        f"tran {t_step:.3g} {t_stop:.6g} 0 {t_step:.3g} uic",
        f"linearize",
        f"wrdata {name}.dat v(fa,n) v(fb,n) i(Ls) v(p) i(La)",
        ".endc", ".end",
    ]
    return "\n".join(lines) + "\n"


def boost_deck(name: str, v_in: float, v_out0: float, duty: float, fsw: float, l: float, r_l: float,
               c_out: float, r_load: float, ron: float, rs_diode: float, n_phases: int = 1,
               t_stop: float = 30e-3, t_step: float = 0.1e-6, il0: float = 0.0) -> str:
    """Open-loop N-phase interleaved boost with fixed duty."""
    period = 1 / fsw
    lines = [
        f"* {name}: interleaved boost cross-check deck",
        SW_MODEL.format(ron=ron), D_MODEL.format(rs=rs_diode),
        f"Vin in 0 dc {v_in}",
        f"Cout out 0 {c_out} ic={v_out0}",
        f"Rload out 0 {r_load}",
    ]
    for k in range(n_phases):
        delay = k * period / n_phases
        lines += [
            f"Vg{k} g{k} 0 pulse(0 1 {delay:.6g} 1e-9 1e-9 {duty * period - 2e-9:.6g} {period:.6g})",
            f"L{k} in x{k} {l} ic={il0 / n_phases}",
            f"RL{k} x{k} sw{k} {r_l}",
            f"S{k} sw{k} 0 g{k} 0 swm",
            f"D{k} sw{k} out dfw",
        ]
    lines += [
        ".options method=gear reltol=1e-4",
        ".control", "set wr_vecnames", "set wr_singlescale",
        f"tran {t_step:.3g} {t_stop:.6g} 0 {t_step:.3g} uic",
        "linearize",
        "wrdata " + name + ".dat v(out) " + " ".join(f"i(L{k})" for k in range(n_phases)),
        ".endc", ".end",
    ]
    return "\n".join(lines) + "\n"
