"""Validation table: Fluent results vs Qu & Mudawar (2002) experiment, their CFD and the 1-D model.

    python compare_cfd.py [mesh tag, default unitcell_r1]      -> results/validation_summary.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analytical"))
from inputs import Geometry  # noqa: E402
from model import flow_from_re, run  # noqa: E402

DATA = ROOT / "data/digitised"
RES = Path(__file__).resolve().parent / "results"
T_IN = 15.0
GEO = Geometry()
CASES = {100: (226, 890, 1454), 200: (442, 864, 1220)}


def load(tag: str, q: float, re: float, solid: str = "copper") -> dict | None:
    f = RES / f"{tag}_{solid}_q{q:g}_re{re:g}.json"
    return json.loads(f.read_text()) if f.exists() else None


def minor_loss_bar(r: dict, k_total: float) -> float:
    return k_total * 0.5 * 999.1 * r["u_in"] ** 2 / 1e5


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "unitcell_r1"
    f4b = pd.read_csv(DATA / "fig4b_dp_num_vs_exp.csv")
    f5 = pd.read_csv(DATA / "fig5_dT_vs_re.csv")
    f6 = pd.read_csv(DATA / "fig6_tc_vs_re.csv")
    f7 = pd.read_csv(DATA / "fig7_tc_along_x.csv")
    rows = []
    for q, curves in ((100, "fig7a_num_curves_100"), (200, "fig7b_num_curves_200")):
        num = pd.read_csv(DATA / f"{curves}.csv")
        for re in CASES[q]:
            r = load(tag, q, re)
            if not r:
                continue
            one_d = run(flow_from_re(re, GEO), q * 1e4, GEO)
            e7 = f7[(f7.q_eff_Wcm2 == q) & (f7.Re == re)].sort_values("x_m")
            qm = np.interp(np.array(GEO.x_tc), num.x, num[f"re{re}"])
            for loc, (t_cfd, t_exp, t_qm) in enumerate(zip(r["t_tc_C"], e7.T_th_C, qm), start=1):
                rows.append(dict(figure="7", q=q, Re=re, quantity=f"T_tc{loc} [C]", exp=t_exp, cfd=t_cfd,
                                 qm_cfd=t_qm, one_d=one_d.t_tc[loc - 1]))
            e5 = f5[f5.q_eff_Wcm2 == q]
            rows.append(dict(figure="5", q=q, Re=re, quantity="T_out-T_in [K]", exp=np.interp(re, e5.Re, e5.dT_out_in_C),
                             cfd=r["t_out"] - 273.15 - T_IN, one_d=one_d.t_out - T_IN))
            b = f4b[(f4b.q_eff_Wcm2 == q) & ((f4b.Re - re).abs() < 15)]
            dp_exp, dp_qm = b[b.source == "exp"].dP_bar.mean(), b[b.source == "num"].dP_bar.mean()
            rows.append(dict(figure="4b", q=q, Re=re, quantity="dP channel [bar]", exp=dp_exp,
                             cfd=r["dp_ch"] / 1e5, qm_cfd=dp_qm, one_d=one_d.dp_ch / 1e5))
            for k in (1.0, 2.0):
                rows.append(dict(figure="4b", q=q, Re=re, quantity=f"dP + {k:g} velocity heads [bar]", exp=dp_exp,
                                 cfd=r["dp_ch"] / 1e5 + minor_loss_bar(r, k), qm_cfd=dp_qm))
            for loc in (1, 2, 3, 4):
                e6 = f6[(f6.q_eff_Wcm2 == q) & (f6.location == loc)].sort_values("Re")
                rows.append(dict(figure="6", q=q, Re=re, quantity=f"T_tc{loc}-T_in [K]",
                                 exp=np.interp(re, e6.Re, e6.dT_th_in_C), cfd=r["t_tc_C"][loc - 1] - T_IN,
                                 one_d=one_d.t_tc[loc - 1] - T_IN))
    df = pd.DataFrame(rows)
    for col in ("cfd", "qm_cfd", "one_d"):
        df[f"err_{col}_%"] = 100 * (df[col] - df.exp) / df.exp
    df.to_csv(RES / "validation_summary.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    df["group"] = df.quantity.str.replace(r"\d", "", regex=True)
    print("\nMean |error| (%) by quantity:")
    print(df.groupby(["figure", "group"])[[c for c in df.columns if c.startswith("err_")]].apply(
        lambda g: g.abs().mean()).round(1).to_string())


if __name__ == "__main__":
    main()
