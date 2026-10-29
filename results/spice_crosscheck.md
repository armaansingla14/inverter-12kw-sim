# Python vs ngspice cross-check

Matched simplified device (30 mOhm in both directions), no dead time, fixed modulation; ngspice uses a behavioural switched-pole leg model.
Python uses regular-sampled PWM; ngspice uses natural (continuous) sampling.

| case | quantity | python | ngspice | diff % |
|---|---|---|---|---|
| xchk_svpwm_10k | v_ph_rms_V | 221.7 | 221.6 | 0.05 |
| xchk_svpwm_10k | p_out_kW | 11.06 | 11.05 | 0.09 |
| xchk_svpwm_10k | thd_v_pct | 0.01659 | 0.1086 | -84.72 |
| xchk_svpwm_10k | il_ripple_pp_A | 3.579 | 3.642 | -1.74 |
| xchk_spwm_10k | v_ph_rms_V | 221.7 | 221.6 | 0.03 |
| xchk_spwm_10k | p_out_kW | 11.06 | 11.05 | 0.05 |
| xchk_spwm_10k | thd_v_pct | 0.007462 | 0.08872 | -91.59 |
| xchk_spwm_10k | il_ripple_pp_A | 3.607 | 3.648 | -1.13 |
| boost_duty_0.4312 | v_out_avg_V | 700 | 699.9 | 0.02 |
| boost_duty_0.4312 | il_avg_A | 30.13 | 30.22 | -0.31 |
| boost_duty_0.4312 | il_ripple_pp_A | 8.491 | 8.616 | -1.46 |
| boost_duty_0.4312 | vout_ripple_pp_V | 0.7941 | 0.8437 | -5.88 |
