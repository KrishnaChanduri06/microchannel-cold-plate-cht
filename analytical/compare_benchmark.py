"""1-D model vs the digitised Qu & Mudawar (2002) data: error tables.

Run from analytical/:  python compare_benchmark.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from inputs import T_IN_C, Geometry
from model import flow_from_re, run

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/digitised"
GEO = Geometry()


def sweep(q_wcm2: float, re_values) -> pd.DataFrame:
    """1-D model at a list of Re (inlet properties)."""
    rows = []
    for re in re_values:
        r = run(flow_from_re(re, GEO), q_wcm2 * 1e4, GEO)
        rows.append(dict(Re=re, dP=r.dp / 1e5, dP_ch=r.dp_ch / 1e5, dP_wall=(r.dp_ch_wall + r.dp_minor) / 1e5,
                         dT=r.t_out - T_IN_C, **{f"tc{i + 1}": t - T_IN_C for i, t in enumerate(r.t_tc)},
                         R_cond=r.r_cond, R_conv=r.r_conv, R_cal=r.r_cal, L_th=r.l_th, eta_f=r.eta_f))
    return pd.DataFrame(rows)


def err(model: np.ndarray, exp: np.ndarray) -> str:
    e = 100 * (model - exp) / exp
    return f"mean {e.mean():+5.1f} %, range {e.min():+5.1f} .. {e.max():+5.1f} %"


def main() -> None:
    f4 = pd.read_csv(DATA / "fig4a_dp_vs_re.csv")
    f5 = pd.read_csv(DATA / "fig5_dT_vs_re.csv")
    f6 = pd.read_csv(DATA / "fig6_tc_vs_re.csv")

    print("Pressure drop vs Fig. 4a")
    for q in (100, 200):
        e = f4[f4.q_eff_Wcm2 == q]
        m = sweep(q, e.Re)
        print(f"  {q} W/cm2  bulk mu:      {err(m.dP.to_numpy(), e.dP_bar.to_numpy())}")
        print(f"  {q} W/cm2  wall-mu corr: {err(m.dP_wall.to_numpy(), e.dP_bar.to_numpy())}")
        print(f"  {q} W/cm2  channel only: {err(m.dP_ch.to_numpy(), e.dP_bar.to_numpy())}")

    print("\nWater temperature rise vs Fig. 5")
    for q in (100, 200):
        e = f5[f5.q_eff_Wcm2 == q]
        print(f"  {q} W/cm2: {err(sweep(q, e.Re).dT.to_numpy(), e.dT_out_in_C.to_numpy())}")

    print("\nThermocouple rise vs Fig. 6 (all Re | Re < 400 | Re > 1000)")
    for q in (100, 200):
        for loc in (1, 2, 3, 4):
            e = f6[(f6.q_eff_Wcm2 == q) & (f6.location == loc)]
            m = sweep(q, e.Re)[f"tc{loc}"].to_numpy()
            x = e.dT_th_in_C.to_numpy()
            lo, hi = (e.Re < 400).to_numpy(), (e.Re > 1000).to_numpy()
            print(f"  {q} W/cm2 TC{loc}: {err(m, x)} | {err(m[lo], x[lo]) if lo.any() else '-'} | {err(m[hi], x[hi])}")

    print("\nThermal entry length vs channel length (L = 44.8 mm), 100 W/cm2")
    for re in (139, 300, 500, 1000, 1672):
        r = run(flow_from_re(re, GEO), 100e4, GEO)
        print(f"  Re {re:5d}: L_th = {r.l_th * 1e3:6.1f} mm (L_th/L = {r.l_th / GEO.length:4.1f}), "
              f"R_cond/R_conv/R_cal = {r.r_cond * 1e3:.0f}/{r.r_conv * 1e3:.0f}/{r.r_cal * 1e3:.0f} mK/W, "
              f"eta_f = {r.eta_f:.3f}")


if __name__ == "__main__":
    main()
