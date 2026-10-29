from dataclasses import replace

import numpy as np
import pytest

from invsim.designs import INV_BASELINE, INV_IMPROVED
from invsim.devices import Device
from invsim.inverter import OperatingPoint, simulate

IDEAL = Device("ideal", "igbt", v0=0.0, r_on=1e-4, vf0=0.0, rf=1e-4, e_on=0.0, e_off=0.0, e_rr=0.0,
               deadtime=0.0)


@pytest.fixture(scope="module")
def ideal_run():
    d = replace(INV_IMPROVED, device=IDEAL, r_l=1e-4, fsw=20e3, name="ideal")
    return simulate(d, OperatingPoint(), warmup_cycles=3, measure_cycles=1)


def test_regulates_output_voltage(ideal_run):
    m = ideal_run.metrics
    assert m["v_ph_rms"] == pytest.approx(230.94, rel=0.01)
    assert m["p_out_kw"] == pytest.approx(12.0, rel=0.02)


def test_ideal_inverter_is_lossless_and_balanced(ideal_run):
    m = ideal_run.metrics
    assert m["p_loss_w"] - m["p_filter_w"] < 2.0
    assert abs(m["balance_err_w"]) < 0.005 * 12e3


def test_ideal_inverter_low_thd(ideal_run):
    assert ideal_run.metrics["thd_v_pct"] < 1.0


def test_energy_balance_with_real_devices():
    r = simulate(INV_BASELINE, OperatingPoint(), warmup_cycles=3, measure_cycles=1)
    m = r.metrics
    assert abs(m["balance_err_w"]) < 0.1 * m["p_cond_w"] + 5
    assert 95 < m["efficiency_pct"] < 99.9


def test_deadtime_increases_low_order_distortion():
    base = replace(INV_BASELINE, deadtime=0.0)
    big = replace(INV_BASELINE, deadtime=3e-6)
    t0 = simulate(base, OperatingPoint(), warmup_cycles=3, measure_cycles=1).metrics["thd_i_pct"]
    t1 = simulate(big, OperatingPoint(), warmup_cycles=3, measure_cycles=1).metrics["thd_i_pct"]
    assert t1 > t0
