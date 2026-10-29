"""Write the ngspice cross-check decks used by validate_spice.py into spice/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from invsim import spice  # noqa: E402
from invsim.dcdc import BoostOp  # noqa: E402
from invsim.designs import BOOST_BASELINE, INV_BASELINE  # noqa: E402
from invsim.inverter import OperatingPoint  # noqa: E402

out = ROOT / "spice"
out.mkdir(exist_ok=True)
op = OperatingPoint()
r_load, _ = op.load_rl()
d = INV_BASELINE
for scheme in ("svpwm", "spwm"):
    deck = spice.inverter_deck(f"inv_{scheme}", op.v_dc, 10e3, 0.9, op.f1, d.l_f, d.r_l, d.c_f, r_load, d.c_dc,
                               op.r_s, op.l_s, 0.03, scheme, t_stop=0.1, t_step=0.2e-6)
    (out / f"inverter_3ph_{scheme}_10k.cir").write_text(deck)
b, bop = BOOST_BASELINE, BoostOp()
deck = spice.boost_deck("boost", bop.v_in, bop.v_out, 0.4312, b.fsw, b.l, b.r_l, b.c_out, bop.r_load, 0.03,
                        n_phases=1, t_stop=60e-3, t_step=0.1e-6, il0=bop.p_out / bop.v_in)
(out / "boost_1ph_20k.cir").write_text(deck)
deck = spice.boost_deck("boost2", bop.v_in, bop.v_out, 0.4312, 50e3, 400e-6, 0.025, b.c_out, bop.r_load, 0.06,
                        n_phases=2, t_stop=30e-3, t_step=0.05e-6, il0=bop.p_out / bop.v_in)
(out / "boost_2ph_50k.cir").write_text(deck)
print("wrote", sorted(p.name for p in out.glob("*.cir")))
