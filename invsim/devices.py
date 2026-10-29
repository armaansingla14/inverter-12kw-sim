"""Semiconductor loss models.

Parameters are generic, datasheet-shaped approximations for 1200 V class
devices at a junction temperature of roughly 100-125 C. They are not taken
from any specific vendor part and should be treated as representative only.

Conduction:  v = V0 + r * |i|              (IGBT / diode)
             v = r_ds * |i|                (MOSFET channel, either direction)
Switching:   E = E_ref * (|i| / I_ref)^ki * (V / V_ref)^kv
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    name: str
    kind: str              # "igbt" or "mosfet"
    v0: float              # IGBT knee voltage (0 for MOSFET)
    r_on: float            # on-state slope resistance / Rds(on)
    vf0: float             # antiparallel / body diode knee voltage
    rf: float              # diode slope resistance
    e_on: float            # J at (v_ref, i_ref)
    e_off: float
    e_rr: float            # diode reverse-recovery energy
    v_ref: float = 600.0
    i_ref: float = 40.0
    ki: float = 1.0
    kv: float = 1.3
    sync_rect: bool = False  # MOSFET conducts reverse current through the channel
    deadtime: float = 1e-6

    def switch_energy(self, e_ref: float, i: float, v: float) -> float:
        if i <= 0.0:
            return 0.0
        return e_ref * (i / self.i_ref) ** self.ki * (v / self.v_ref) ** self.kv

    def fwd_drop(self, i_abs: float) -> float:
        """Drop of the active switch carrying forward current."""
        return self.v0 + self.r_on * i_abs

    def rev_drop(self, i_abs: float, in_deadtime: bool) -> float:
        """Drop for reverse current (diode, or MOSFET channel when sync-rectifying)."""
        if self.sync_rect and not in_deadtime:
            return self.r_on * i_abs
        return self.vf0 + self.rf * i_abs


# Generic 1200 V / 40 A class trench-field-stop IGBT with Si fast-recovery diode
SI_IGBT = Device(
    name="si_igbt_1200v_40a", kind="igbt",
    v0=0.85, r_on=0.028, vf0=1.0, rf=0.022,
    e_on=2.6e-3, e_off=1.9e-3, e_rr=1.1e-3,
    kv=1.3, deadtime=1.0e-6,
)

# Generic 1200 V / ~40 mOhm SiC MOSFET (Rds(on) at ~125 C), synchronous rectification
SIC_MOSFET = Device(
    name="sic_mosfet_1200v_40mohm", kind="mosfet",
    v0=0.0, r_on=0.060, vf0=3.0, rf=0.030,
    e_on=0.55e-3, e_off=0.18e-3, e_rr=0.06e-3,
    kv=1.2, sync_rect=True, deadtime=0.2e-6,
)

# Diodes for the boost stage (no active switch): Si ultrafast vs SiC Schottky
@dataclass(frozen=True)
class Diode:
    name: str
    vf0: float
    rf: float
    e_rr: float  # J at (v_ref, i_ref), charged to the boost switch turn-on
    v_ref: float = 600.0
    i_ref: float = 40.0

    def drop(self, i_abs: float) -> float:
        return self.vf0 + self.rf * i_abs

    def rr_energy(self, i: float, v: float) -> float:
        if i <= 0:
            return 0.0
        return self.e_rr * (i / self.i_ref) * (v / self.v_ref)


SI_FRD = Diode("si_frd_1200v", vf0=1.1, rf=0.025, e_rr=1.1e-3)
SIC_SBD = Diode("sic_sbd_1200v", vf0=0.9, rf=0.035, e_rr=0.03e-3)  # capacitive charge only

DEVICES = {d.name: d for d in (SI_IGBT, SIC_MOSFET)}
DIODES = {d.name: d for d in (SI_FRD, SIC_SBD)}
