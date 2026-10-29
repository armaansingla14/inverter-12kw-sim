import numpy as np
import pytest
from scipy.linalg import expm

from invsim.analytic import inverter_losses
from invsim.devices import SI_IGBT, SIC_MOSFET
from invsim.linsys import DiscreteLTI
from invsim.spectrum import harmonic_analysis


def test_discrete_lti_matches_expm():
    a = np.array([[-10.0, -400.0], [1e5, -50.0]])
    b = np.array([[400.0], [0.0]])
    sys_ = DiscreteLTI(a, b, 1e-9)
    phi, gam = sys_.mats(12345)
    h = 12345e-9
    m = np.zeros((3, 3))
    m[:2, :2] = a * h
    m[:2, 2:] = b * h
    em = expm(m)
    assert np.allclose(phi, em[:2, :2], rtol=1e-9, atol=1e-12)
    assert np.allclose(gam, em[:2, 2:], rtol=1e-9, atol=1e-12)


def test_harmonic_analysis_known_signal():
    f1, dt = 50.0, 1e-5
    t = np.arange(0, 0.04, dt)
    x = 100 * np.sin(2 * np.pi * f1 * t) + 3 * np.sin(2 * np.pi * 5 * f1 * t) \
        + 0.5 * np.sin(2 * np.pi * 10e3 * t)
    h = harmonic_analysis(x, dt, f1)
    assert h["fund_peak"] == pytest.approx(100, rel=1e-6)
    assert h["thd_pct"] == pytest.approx(3.0, rel=1e-4)
    assert h["hf_rms"] == pytest.approx(0.5 / np.sqrt(2), rel=1e-3)


def test_switch_energy_scaling():
    e1 = SI_IGBT.switch_energy(SI_IGBT.e_on, 40, 600)
    assert e1 == pytest.approx(SI_IGBT.e_on)
    assert SI_IGBT.switch_energy(SI_IGBT.e_on, 20, 600) == pytest.approx(e1 / 2)
    assert SI_IGBT.switch_energy(SI_IGBT.e_on, -5, 600) == 0.0


def test_mosfet_reverse_conduction_uses_channel():
    assert SIC_MOSFET.rev_drop(10, in_deadtime=False) == pytest.approx(0.6)
    assert SIC_MOSFET.rev_drop(10, in_deadtime=True) > 3.0


def test_analytic_losses_positive_and_scale():
    a = inverter_losses(SI_IGBT, 24.5, 0.93, 0.0, 700, 10e3)
    b = inverter_losses(SI_IGBT, 24.5, 0.93, 0.0, 700, 20e3)
    assert a["p_cond_w"] > 0 and b["p_sw_w"] == pytest.approx(2 * a["p_sw_w"])
