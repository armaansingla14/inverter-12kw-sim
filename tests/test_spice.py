"""ngspice cross-check of the boost simulator (skipped if ngspice is unavailable)."""
import sys
from pathlib import Path

import pytest

from invsim import spice

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.mark.spice
@pytest.mark.skipif(not spice.available(), reason="ngspice not available")
def test_boost_matches_ngspice():
    import validate_spice
    rows = {r["quantity"]: r for r in validate_spice.boost_case()}
    assert abs(rows["v_out_avg_V"]["diff_pct"]) < 0.5
    assert abs(rows["il_avg_A"]["diff_pct"]) < 1.0
    assert abs(rows["il_ripple_pp_A"]["diff_pct"]) < 3.0
