# Digitised benchmark data: Qu and Mudawar (2002), IJHMT 45, 2549–2565

Produced by `digitise/` in the order `extract_figs.py`, `digitise.py`, `fig4b_templates.py`, `fig7_curves.py`, `clean.py`. The `*_raw.csv` files are the detector output; the files below are the cleaned data.

| File | Columns | Content |
|---|---|---|
| `fig4a_dp_vs_re.csv` | q_eff_Wcm2, Re, dP_bar, n_markers | Measured ΔP: 25 points at 100 W/cm<sup>2</sup>, 14 points at 200 W/cm<sup>2</sup> |
| `fig4b_dp_num_vs_exp.csv` | q_eff_Wcm2, source (exp/num), Re, dP_bar | Paper CFD and experiment, 3 Re per heat flux |
| `fig5_dT_vs_re.csv` | q_eff_Wcm2, Re, dT_out_in_C, n_markers | Water temperature rise T<sub>out</sub> − T<sub>in</sub> |
| `fig6_tc_vs_re.csv` | q_eff_Wcm2, location, x_tc_mm, Re, dT_th_in_C, n_markers | Thermocouples 1 to 4, T<sub>th</sub> − T<sub>in</sub> |
| `fig7_tc_along_x.csv` | q_eff_Wcm2, Re, location, x_m, T_th_C, n_markers | Thermocouple-line temperature, absolute T<sub>th</sub> |
| `fig7a_num_curves_100.csv`, `fig7b_num_curves_200.csv` | x, re… | Paper CFD along the thermocouple line, 0.1 mm spacing |

Estimated uncertainty: about 0.2 % of the axis span from calibration, about 0.3 K for markers overlapping a curve or another marker. Corrections and repeat-run handling are listed in [`ASSUMPTIONS.md`](../../ASSUMPTIONS.md).
