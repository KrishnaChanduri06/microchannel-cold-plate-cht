"""Design study tables: thermal resistance, pumping power, Pareto front, 1-D model vs CFD.

    python design_analysis.py      -> results/design_summary.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analytical"))
from model import run_design  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
K = {"copper": 401.0, "silicon": 148.0}


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(RES.glob("d_wch*_q*_flow*.json")):
        r = json.loads(f.read_text())
        g = r["geometry"]
        one = run_design(g["w_ch"], 2 * g["w_w"], g["h_ch"], g["h_w2"], r["flow_ml_min"], r["q_eff_Wcm2"] * 1e4,
                         r["t_in_C"], K[r["solid"]])
        rows.append(dict(
            solid=r["solid"], q=r["q_eff_Wcm2"], w_ch=round(g["w_ch"] * 1e6), w_fin=round(2 * g["w_w"] * 1e6),
            flow=round(r["flow_ml_min"]), Re=r["Re"], n_ch=r["n_channels"], dP_bar=r["dp_ch"] / 1e5,
            P_pump_W=r["pumping_power_W"], T_base_max=r["t_heater_max"] - 273.15,
            T_base_avg=r["t_heater_avg"] - 273.15, T_out=r["t_out"] - 273.15, R_th=r["r_th_max_KW"],
            R_th_avg=r["r_th_avg_KW"], energy_balance=r["energy_balance"],
            oneD_T_base_max=one.t_base_max, oneD_R_th=one.r_th, oneD_dP_bar=one.dp_ch / 1e5,
            oneD_eta_f=one.eta_f, L_th_over_L=one.l_th / 44.764e-3))
    df = pd.DataFrame(rows)
    df["err_1D_Rth_%"] = 100 * (df.oneD_R_th - df.R_th) / df.R_th
    df["err_1D_dP_%"] = 100 * (df.oneD_dP_bar - df.dP_bar) / df.dP_bar
    return df


def pareto(d: pd.DataFrame) -> pd.DataFrame:
    """Points not dominated in (pumping power, R_th)."""
    d = d.sort_values("P_pump_W")
    best, keep = np.inf, []
    for i, r in d.iterrows():
        if r.R_th < best:
            keep.append(i)
            best = r.R_th
    return d.loc[keep]


def main() -> None:
    df = load()
    df.to_csv(RES / "design_summary.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 220)
    cols = ["solid", "q", "w_ch", "w_fin", "flow", "Re", "dP_bar", "P_pump_W", "T_base_max", "R_th",
            "oneD_R_th", "err_1D_Rth_%", "err_1D_dP_%", "L_th_over_L", "energy_balance"]
    print(df[cols].sort_values(["q", "solid", "w_ch", "w_fin", "flow"]).round(4).to_string(index=False))
    for solid in ("copper", "silicon"):
        print(f"\nPareto front, {solid}, 100 W/cm2")
        print(pareto(df[(df.solid == solid) & (df.q == 100)])[["w_ch", "w_fin", "flow", "P_pump_W", "R_th",
                                                                  "T_base_max"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
