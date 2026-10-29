"""Baseline and improved design points used by the experiment scripts.

Filter and magnetics values are design choices for this study; the
ripple-loss resistance ``r_ac`` lumps AC copper and core loss into a single
equivalent resistance and is an assumption (not a measured inductor).
"""
from __future__ import annotations

from .dcdc import BoostDesign
from .devices import SI_FRD, SI_IGBT, SIC_MOSFET, SIC_SBD
from .inverter import InverterDesign

# Three-phase inverter, 12 kW, 400 V line-line 50 Hz output from a ~700 V DC link
INV_BASELINE = InverterDesign(
    name="baseline_si_spwm_10k", device=SI_IGBT, fsw=10e3, scheme="spwm",
    l_f=2.5e-3, r_l=0.030, r_ac=0.5, c_f=10e-6, c_dc=470e-6,
)
INV_IMPROVED = InverterDesign(
    name="improved_sic_svpwm_40k", device=SIC_MOSFET, fsw=40e3, scheme="svpwm",
    l_f=0.8e-3, r_l=0.015, r_ac=0.5, c_f=10e-6, c_dc=470e-6,
)

# Boost stage 400 V -> 700 V, 12 kW
BOOST_BASELINE = BoostDesign(
    name="baseline_1ph_si_20k", switch=SI_IGBT, diode=SI_FRD, n_phases=1, fsw=20e3,
    l=1.0e-3, r_l=0.030, r_ac=0.3, c_out=470e-6,
)
BOOST_IMPROVED = BoostDesign(
    name="improved_2ph_sic_50k", switch=SIC_MOSFET, diode=SIC_SBD, n_phases=2, fsw=50e3,
    l=400e-6, r_l=0.025, r_ac=0.3, c_out=470e-6,
)


def stored_energy_l(d: InverterDesign, i_peak: float) -> float:
    """Total peak stored energy of the three filter inductors (J), a size proxy."""
    return 3 * 0.5 * d.l_f * i_peak ** 2
