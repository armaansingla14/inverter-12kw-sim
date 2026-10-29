"""Closed-form (averaged) loss estimates used to sanity-check the switching simulation.

Classic sinusoidal-PWM expressions for a two-level leg with peak phase
current Ip, modulation index m and displacement angle phi (no dead time,
no ripple):

    P_cond,T = (1/(2 pi) + m cos(phi)/8) V0 Ip + (1/8 + m cos(phi)/(3 pi)) r Ip^2
    P_cond,D = (1/(2 pi) - m cos(phi)/8) Vf Ip + (1/8 - m cos(phi)/(3 pi)) rf Ip^2
    P_sw,leg = fsw (Eon + Eoff + Err) (2 Ip / pi) / I_ref * (V / V_ref)^kv

For a synchronous-rectifying MOSFET the channel carries current in both
directions: P_cond = r Irms^2 per conducting position, i.e. r Ip^2 / 4 per
device.
"""
from __future__ import annotations

import numpy as np

from .devices import Device


def inverter_losses(dev: Device, ip: float, m: float, phi: float, v_dc: float, fsw: float) -> dict:
    c = np.cos(phi)
    if dev.sync_rect:
        p_t = dev.r_on * ip ** 2 / 4.0
        p_d = 0.0
    else:
        p_t = (1 / (2 * np.pi) + m * c / 8) * dev.v0 * ip + (1 / 8 + m * c / (3 * np.pi)) * dev.r_on * ip ** 2
        p_d = (1 / (2 * np.pi) - m * c / 8) * dev.vf0 * ip + (1 / 8 - m * c / (3 * np.pi)) * dev.rf * ip ** 2
    i_avg = 2 * ip / np.pi
    e_sum = dev.e_on + dev.e_off + dev.e_rr
    p_sw_leg = fsw * e_sum * (i_avg / dev.i_ref) ** dev.ki * (v_dc / dev.v_ref) ** dev.kv
    return {
        "p_cond_w": 6 * (p_t + p_d),
        "p_sw_w": 3 * p_sw_leg,
    }


def lc_ripple_pp(v_dc: float, l: float, fsw: float) -> float:
    """Worst-case inductor ripple of a two-level leg (at duty 0.5): Vdc / (4 L fsw)."""
    return v_dc / (4 * l * fsw)
