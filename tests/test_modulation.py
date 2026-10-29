import numpy as np
import pytest

from invsim.modulation import M_MAX, duties, leg_edges, references, single_phase_duties


@pytest.mark.parametrize("scheme", ["spwm", "svpwm", "thipwm"])
def test_line_voltages_are_sinusoidal(scheme):
    m = 0.9
    th = np.linspace(0, 2 * np.pi, 721)
    ref = np.array([references(t, m, scheme) for t in th])
    vab = ref[:, 0] - ref[:, 1]
    expected = m * (np.cos(th) - np.cos(th - 2 * np.pi / 3))
    assert np.allclose(vab, expected, atol=1e-12)


def test_svpwm_extends_linear_range():
    m = M_MAX["svpwm"]
    th = np.linspace(0, 2 * np.pi, 1441)
    peak = max(np.abs(references(t, m, "svpwm")).max() for t in th)
    assert peak == pytest.approx(1.0, abs=1e-9)
    peak_spwm = max(np.abs(references(t, m, "spwm")).max() for t in th)
    assert peak_spwm > 1.1


def test_duties_bounded_and_edges_centered():
    d = duties(0.3, 1.0, "svpwm")
    assert np.all((d >= 0) & (d <= 1))
    on, off = leg_edges(0.4, 1e-4)
    assert on == pytest.approx(0.3e-4) and off == pytest.approx(0.7e-4)


def test_single_phase_unipolar():
    d = single_phase_duties(0.0, 0.8)
    assert d[0] - d[1] == pytest.approx(0.8)
