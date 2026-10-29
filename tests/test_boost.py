from dataclasses import replace

import pytest

from invsim.dcdc import BoostOp, feedforward_duty, ripple_ccm_analytic, simulate
from invsim.designs import BOOST_BASELINE, BOOST_IMPROVED


def test_ccm_ripple_matches_analytic():
    r = simulate(BOOST_BASELINE, BoostOp())
    m = r.metrics
    assert m["ccm"]
    assert m["v_out_avg"] == pytest.approx(700, abs=1.0)
    assert m["il_ripple_pp_a"] == pytest.approx(ripple_ccm_analytic(400, 700, BOOST_BASELINE.l, BOOST_BASELINE.fsw), rel=0.05)
    assert abs(m["balance_err_w"]) < 10


def test_interleaving_cancels_input_ripple():
    single = replace(BOOST_IMPROVED, n_phases=1, l=BOOST_IMPROVED.l / 2)
    m1 = simulate(single, BoostOp()).metrics
    m2 = simulate(BOOST_IMPROVED, BoostOp()).metrics
    assert m2["iin_ripple_pp_a"] < 0.5 * m1["iin_ripple_pp_a"]


def test_light_load_enters_dcm_and_regulates():
    m = simulate(BOOST_BASELINE, BoostOp(p_out=600)).metrics
    assert not m["ccm"]
    assert m["v_out_avg"] == pytest.approx(700, abs=2.0)


def test_feedforward_duty_ccm_value():
    assert feedforward_duty(BOOST_BASELINE, BoostOp()) == pytest.approx(1 - 400 / 700)
