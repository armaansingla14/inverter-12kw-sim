# inverter-12kw-sim

Simulation study of a 12 kW power-conversion chain: a 400 V to 700 V boost
front end and a three-phase two-level inverter (400 V line-line, 50 Hz) with
an LC output filter. Both stages are modelled at switching level, with
conduction and switching losses, and a baseline design is compared against an
improved one over load, DC-link voltage, switching frequency, modulation, dead
time and power factor. **Nothing was built in hardware; every number below is
from simulation.**

## Features

- **Exact piecewise-linear switching simulator** (`invsim/inverter.py`,
  `invsim/dcdc.py`): between switching events the circuit is LTI and is
  advanced with the matrix exponential (`invsim/linsys.py`), so there is no
  integration-step error; events (PWM edges, dead-time edges, DCM current
  zero-crossings found by bisection) are placed on a 1 ns time base.
- **Inverter**: SPWM, SVPWM (min-max injection) and THIPWM, regular-sampled
  centred PWM, dead time with diode freewheeling, DC source impedance and
  DC-link capacitor, LC filter with DCR and a ripple-loss resistance, R or
  R-L load, outer loop regulating the load voltage once per fundamental cycle.
- **Boost**: N-phase interleaving, CCM/DCM, average-current-mode control
  (outer voltage PI, inner per-phase current loop).
- **Loss model** (`invsim/devices.py`): IGBT knee + slope, diode knee + slope,
  synchronous MOSFET channel conduction, Eon/Eoff/Err scaled with current and
  voltage, inductor DCR and AC ripple loss, capacitor ESR.
- **Checks**: energy balance of the simulated circuit, closed-form averaged
  loss formulas (`invsim/analytic.py`), analytic CCM ripple, and an
  **ngspice** cross-check of both stages (`scripts/validate_spice.py`).
- LTspice-syntax netlists in `ltspice/` (not run here).

## Designs compared

| | Baseline | Improved |
|---|---|---|
| Inverter devices | generic 1200 V Si IGBT + FRD | generic 1200 V SiC MOSFET (sync. rect.) |
| Inverter PWM | SPWM, 10 kHz, 1 us dead time | SVPWM, 40 kHz, 0.2 us dead time |
| Inverter filter | 2.5 mH / 10 uF per phase | 0.8 mH / 10 uF per phase |
| Boost | 1 phase, Si IGBT + Si FRD, 20 kHz, 1 mH | 2-phase interleaved, SiC MOSFET + SiC SBD, 50 kHz, 400 uH/phase |

Device parameters are generic datasheet-shaped approximations, not a
specific vendor part; the inductor ripple-loss resistance is an assumed value.

## Layout

```
invsim/devices.py     device loss parameters
invsim/modulation.py  SPWM / SVPWM / THIPWM references and duty cycles
invsim/linsys.py      exact discretization with caching
invsim/inverter.py    three-phase inverter switching simulator + metrics
invsim/dcdc.py        interleaved boost switching simulator + metrics
invsim/spectrum.py    harmonic analysis (THD to 50th, HF ripple)
invsim/analytic.py    averaged loss formulas for cross-checking
invsim/designs.py     baseline / improved design points
invsim/spice.py       ngspice deck generators and runner
scripts/run_sweeps.py       all sweeps -> results/*.csv, results/summary.md, plots
scripts/validate_spice.py   Python vs ngspice -> results/spice_crosscheck.*
scripts/export_spice.py     writes spice/*.cir
spice/                ngspice decks (run by validate_spice.py)
ltspice/              LTspice netlists (not run)
tests/                pytest (modulation, LTI stepping, spectra, inverter, boost, ngspice)
```

## How to run

```bash
pip install numpy scipy matplotlib pytest
python -m pytest -q                  # 21 tests; one uses ngspice (native or docker)
python scripts/run_sweeps.py         # ~7 min, writes results/
python scripts/validate_spice.py     # ~9 min (two 100 ms inverter transients in ngspice)
```

ngspice runs natively if on `PATH`, otherwise in the `devtools:latest` docker image.

## Results (simulated)

Produced by `python scripts/run_sweeps.py`; full tables in `results/summary.md`.

### Inverter at rated point (12 kW, Vdc = 700 V, unity PF)

| Design | Efficiency | Total loss | Conduction | Switching | Filter | THD v (2-50) | i_L ripple p-p | DC-link ripple p-p |
|---|---|---|---|---|---|---|---|---|
| baseline Si SPWM 10 kHz | 98.58 % | 172.3 W | 65.4 W | 79.0 W | 27.9 W | 0.74 % | 3.79 A | 1.00 V |
| improved SiC SVPWM 40 kHz | 99.06 % | 113.4 W | 56.0 W | 43.6 W | 13.9 W | 0.58 % | 3.01 A | 0.14 V |

The improved design cuts inverter loss by 34 % (172 to 113 W) while using a
filter inductor 3.1x smaller (stored energy is proportional to L). The
averaged closed-form loss estimate agrees with the switching simulation
(conduction 65.3 vs 65.4 W and 54.0 vs 56.0 W; switching 80.0 vs 79.0 W and
44.5 vs 43.6 W), and the simulated circuit's energy balance closes within 0.4 W.

Other inverter observations from the sweeps:

- **Load sweep** (10-110 %): baseline 98.54-98.86 %, improved 99.01-99.53 %
  (peak at light load for the SiC design, where switching loss dominates less).
- **DC-link sweep**: with SPWM the baseline saturates at m = 1 below 700 V and
  misses the 230.9 V phase target (222.2 V at 650 V, 211.9 V at 620 V). SVPWM
  holds 230.9 V down to 620 V (m = 1.081).
- **Switching frequency**: Si efficiency falls from 98.88 % (5 kHz) to 97.29 %
  (30 kHz) while low-order THD *rises* (0.32 % to 2.35 %) because the dead-time
  error grows with fsw x td. SiC goes from 99.24 % (20 kHz) to 98.69 % (80 kHz).
- **Same filter (0.8 mH) for both devices at 40 kHz**: Si 96.71 % vs SiC
  99.06 %, isolating the device effect from the filter choice.
- **Dead time** (Si, 10 kHz): THD 0.42 / 0.74 / 1.36 % for 0.5 / 1 / 2 us.
- **PF 0.8 load**: 98.16 % (baseline) vs 98.71 % (improved).

### Boost at rated point (Vin = 400 V, Vout = 700 V, 12 kW)

| Design | Efficiency | Loss | Switching | Inductor ripple p-p (sim / analytic) | Input ripple p-p | Vout ripple p-p |
|---|---|---|---|---|---|---|
| baseline 1-ph Si 20 kHz | 98.50 % | 182.4 W | 97.8 W | 8.52 / 8.57 A | 8.52 A (28.2 %) | 0.80 V |
| improved 2-ph SiC 50 kHz | 99.33 % | 81.5 W | 29.2 W | 8.49 / 8.57 A | 1.89 A (6.3 %) | 0.049 V |

Interleaving cancels most of the input ripple; at Vin = 350 V (D = 0.5) the
two-phase input ripple falls to 0.04-0.08 A, i.e. near-complete cancellation.
Across Vin = 350-450 V and 5-100 % load the baseline spans 98.22-99.02 % and
the improved design 99.19-99.56 %; both enter DCM at 5-10 % load, and the
improved design (smaller per-phase inductance) also at 25 % load.

### ngspice cross-check (`python scripts/validate_spice.py`)

Matched simplified devices (30 mOhm, no knee, no dead time), fixed modulation.

| Case | Quantity | Python | ngspice | Difference |
|---|---|---|---|---|
| inverter SVPWM 10 kHz | phase voltage (V rms) | 221.7 | 221.6 | 0.05 % |
| inverter SVPWM 10 kHz | output power (kW) | 11.06 | 11.05 | 0.09 % |
| inverter SVPWM 10 kHz | inductor ripple p-p (A) | 3.58 | 3.64 | -1.7 % |
| inverter SPWM 10 kHz | phase voltage (V rms) | 221.7 | 221.6 | 0.03 % |
| boost 20 kHz | Vout average (V) | 700.0 | 699.9 | 0.02 % |
| boost 20 kHz | inductor ripple p-p (A) | 8.49 | 8.62 | -1.5 % |
| boost 20 kHz | Vout ripple p-p (V) | 0.79 | 0.84 | -5.9 % |

Low-order THD in this idealized case is below 0.11 % in both tools (0.017 %
Python vs 0.109 % ngspice); at that level the difference reflects regular vs
natural sampling and the ngspice time step rather than circuit behaviour.

## Limitations

- Switching-level, not device-physics: switching transitions are
  instantaneous in the circuit and their energy is added from scaled
  Eon/Eoff/Err tables; no gate-drive, parasitic-inductance, EMI or
  thermal (junction-temperature feedback) modelling.
- Device and magnetics parameters are generic; the inductor AC/core loss is a
  single assumed equivalent resistance.
- The inverter regulator is a slow per-cycle amplitude loop (no current
  control); the boost controller is tuned only to settle within the warm-up.
- The ngspice decks use a behavioural switched-pole leg model (hard `sw` +
  diode commutation at this power level did not converge reliably), so the
  cross-check validates the circuit solution, not device models.
- LTspice netlists were not run.

## License

MIT, see `LICENSE`.
