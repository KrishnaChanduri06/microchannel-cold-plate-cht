"""Clean the raw digitiser output into the benchmark CSVs; manual corrections are listed here.

Run after digitise.py, fig4b_templates.py and fig7_curves.py:
    .venv/Scripts/python digitise/clean.py
"""
import numpy as np
import pandas as pd

from digitise import OUT

# thermocouple x positions (paper Sec. 2.2): first at 5 mm, then every 11.588 mm
X_TC_MM = [5.0, 16.588, 28.176, 39.764]


def raw(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"{name}_raw.csv")


def merge_repeats(df: pd.DataFrame, by: list[str], key: str = "Re", tol: float = 10.0) -> pd.DataFrame:
    """Average repeat runs (markers within tol in Re)."""
    out = []
    for _, g in df.sort_values(key).groupby(by):
        g = g.sort_values(key)
        cluster = np.cumsum(np.r_[0, np.diff(g[key].to_numpy()) > tol])
        for _, c in g.groupby(cluster):
            row = c.iloc[0].copy()
            for col in c.select_dtypes("number").columns:
                if col not in by:
                    row[col] = c[col].mean()
            row["n_markers"] = len(c)
            out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


def fig4a() -> pd.DataFrame:
    d = raw("fig4a_dp_vs_re")
    d["q_eff_Wcm2"] = d.series.map({"exp_100": 100, "exp_200": 200})
    d = d.rename(columns={"x": "Re", "y": "dP_bar"})[["q_eff_Wcm2", "Re", "dP_bar"]]
    return merge_repeats(d, ["q_eff_Wcm2"])


def fig4b() -> pd.DataFrame:
    d = raw("fig4b_dp_num_vs_exp")
    d["q_eff_Wcm2"] = d.series.str[-3:].astype(int)
    d["source"] = d.series.str[:3]
    return d.rename(columns={"x": "Re", "y": "dP_bar"})[["q_eff_Wcm2", "source", "Re", "dP_bar"]] \
        .sort_values(["q_eff_Wcm2", "source", "Re"]).reset_index(drop=True)


def fig5() -> pd.DataFrame:
    d = raw("fig5_dT_vs_re")
    d["q_eff_Wcm2"] = d.series.map({"exp_100": 100, "exp_200": 200})
    d = d.rename(columns={"x": "Re", "y": "dT_out_in_C"})[["q_eff_Wcm2", "Re", "dT_out_in_C"]]
    # Re ~141: two stacked squares (68.5, 66.5 C), repeat run, averaged
    return merge_repeats(d, ["q_eff_Wcm2"])


def fig6() -> pd.DataFrame:
    parts = []
    for name, q in (("fig6a_tc_vs_re_100", 100), ("fig6b_tc_vs_re_200", 200)):
        d = raw(name)
        if q == 100:
            # Re ~953: two stacked diamonds (repeat run). upper one misread as tc1, set to tc4;
            # lower one half hidden (misread as tc2), dropped
            near = (d.x.between(945, 960))
            d.loc[near & (d.series == "tc1") & d.y.between(37, 40), "series"] = "tc4"
            d = d[~(near & (d.series == "tc2") & d.y.between(35, 38))]
        d["q_eff_Wcm2"] = q
        d["location"] = d.series.str[-1].astype(int)
        parts.append(d.rename(columns={"x": "Re", "y": "dT_th_in_C"})[["q_eff_Wcm2", "location", "Re", "dT_th_in_C"]])
    d = merge_repeats(pd.concat(parts), ["q_eff_Wcm2", "location"])
    d["x_tc_mm"] = d.location.map(dict(enumerate(X_TC_MM, start=1)))
    return d[["q_eff_Wcm2", "location", "x_tc_mm", "Re", "dT_th_in_C", "n_markers"]]


def fig7() -> pd.DataFrame:
    parts = []
    # legend boxes (x_m_max, T_min); legend symbols sit at x ~ 5 mm, same as TC1
    legend = {100: (0.0246, 81.0), 200: (0.0251, 106.0)}
    for name, q in (("fig7a_tc_along_x_100", 100), ("fig7b_tc_along_x_200", 200)):
        d = raw(name)
        d = d[~((d.x < legend[q][0]) & (d.y > legend[q][1]))]
        # keep markers within 0.6 mm of a thermocouple position (drops label letters)
        x_mm = d.x * 1e3
        loc = np.argmin(np.abs(x_mm.to_numpy()[:, None] - np.array(X_TC_MM)[None, :]), axis=1)
        ok = np.abs(x_mm - np.array(X_TC_MM)[loc]) < 0.6
        d = d[ok].assign(location=loc[ok] + 1)
        d["Re"] = d.series.str[2:].astype(int)
        d["q_eff_Wcm2"] = q
        parts.append(d.rename(columns={"x": "x_m", "y": "T_th_C"}))
    d = pd.concat(parts)
    d = d.groupby(["q_eff_Wcm2", "Re", "location"], as_index=False).agg(
        x_m=("x_m", "mean"), T_th_C=("T_th_C", "mean"), n_markers=("T_th_C", "size"))
    return d


def report(f4a, f5, f6, f7) -> None:
    print("counts: fig4a", f4a.groupby("q_eff_Wcm2").size().to_dict(),
          "| fig5", f5.groupby("q_eff_Wcm2").size().to_dict(),
          "| fig6", f6.groupby(["q_eff_Wcm2", "location"]).size().to_dict(),
          "| fig7", len(f7))
    # Figs 4a, 5, 6 share flow rates, Re should match point by point
    for q in (100, 200):
        a = f4a[f4a.q_eff_Wcm2 == q].Re.to_numpy()
        b = f5[f5.q_eff_Wcm2 == q].Re.to_numpy()
        c = f6[(f6.q_eff_Wcm2 == q) & (f6.location == 1)].Re.to_numpy()
        if len(a) == len(b) == len(c):
            print(f"q={q}: max |Re(4a)-Re(5)| = {np.abs(a - b).max():.1f}, "
                  f"max |Re(4a)-Re(6)| = {np.abs(a - c).max():.1f}  (n={len(a)})")
        else:
            print(f"q={q}: point counts differ 4a/5/6 = {len(a)}/{len(b)}/{len(c)}")
    print("Re range 100:", f4a[f4a.q_eff_Wcm2 == 100].Re.agg(["min", "max"]).round(0).tolist(),
          " paper 139-1672")
    print("Re range 200:", f4a[f4a.q_eff_Wcm2 == 200].Re.agg(["min", "max"]).round(0).tolist(),
          " paper 385-1289")


def main() -> None:
    f4a, f4b, f5, f6, f7 = fig4a(), fig4b(), fig5(), fig6(), fig7()
    f4a.to_csv(OUT / "fig4a_dp_vs_re.csv", index=False, float_format="%.4g")
    f4b.to_csv(OUT / "fig4b_dp_num_vs_exp.csv", index=False, float_format="%.4g")
    f5.to_csv(OUT / "fig5_dT_vs_re.csv", index=False, float_format="%.4g")
    f6.to_csv(OUT / "fig6_tc_vs_re.csv", index=False, float_format="%.4g")
    f7.to_csv(OUT / "fig7_tc_along_x.csv", index=False, float_format="%.4g")
    report(f4a, f5, f6, f7)


if __name__ == "__main__":
    main()
