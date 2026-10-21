"""Extract field sections and the local Nusselt number from a saved Fluent case.

    python extract_fields.py runs/unitcell_r1_copper_q100_re890.cas.h5
Writes results/fields_<case>.npz (cross-section and mid-plane) and results/local_nu_<case>.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np

import case as C
import mesh as M
import water_props as WP

RES = Path(__file__).resolve().parent / "results"


def cells(s, zones):
    svd = s.fields.solution_variable_data
    out = {}
    for z in zones:
        c = np.asarray(svd.get_data(variable_name="SV_CENTROID", zone_names=[z], domain_name="mixture")[z]).reshape(-1, 3)
        t = np.asarray(svd.get_data(variable_name="SV_T", zone_names=[z], domain_name="mixture")[z])
        d = dict(x=c[:, 0], y=c[:, 1], z=c[:, 2], T=t - 273.15)
        if z == "water":
            d["U"] = np.asarray(svd.get_data(variable_name="SV_U", zone_names=[z], domain_name="mixture")[z])
        out[z] = d
    return out


def to_grid(d, key):
    """Structured cell data -> 3-D array on the unique x, y, z centroids."""
    xs, ys, zs = (np.unique(np.round(d[a], 10)) for a in "xyz")
    arr = np.full((len(xs), len(ys), len(zs)), np.nan)
    arr[np.searchsorted(xs, np.round(d["x"], 10)), np.searchsorted(ys, np.round(d["y"], 10)),
        np.searchsorted(zs, np.round(d["z"], 10))] = d[key]
    return xs, ys, zs, arr


def main() -> None:
    cas = Path(sys.argv[1]).resolve()
    tag = cas.name.replace(".cas.h5", "")
    s = C.launch(cas.parent)
    try:
        s.settings.file.read_case_data(file_name=str(cas))
        data = cells(s, ["water", "copper", "cover", "air"])
    finally:
        s.exit()
    meta = json.loads((RES / f"{tag}.json").read_text())

    allc = {k: np.concatenate([data[z][k] for z in data]) for k in ("x", "y", "z", "T")}
    xs, ys, zs, T = to_grid(allc, "T")
    wx, wy, wz, U = to_grid(data["water"], "U")
    _, _, _, Tw = to_grid(data["water"], "T")

    # local Nusselt number: bulk T (velocity weighted) and mean temperature of the 3 heated walls
    area = np.outer(np.gradient(wy), np.gradient(wz))
    t_b = np.nansum(U * Tw * area, axis=(1, 2)) / np.nansum(U * area, axis=(1, 2))
    cxs, cys, czs, Tc = to_grid(data["copper"], "T")
    kb = np.where(czs < M.H_W2)[0].max()
    jl = np.where(cys < M.W_W)[0].max()
    jr = np.where(cys > M.W_W + M.W_CH)[0].min()
    kch = np.where((czs > M.H_W2) & (czs < M.H_W2 + M.H_CH))[0]
    jch = np.where((cys > M.W_W) & (cys < M.W_W + M.W_CH))[0]
    t_bot = np.nanmean(Tc[:, jch, kb], axis=1)
    t_side = 0.5 * (np.nanmean(Tc[:, jl, kch], axis=1) + np.nanmean(Tc[:, jr, kch], axis=1))
    t_wall = (M.W_CH * t_bot + 2 * M.H_CH * t_side) / (M.W_CH + 2 * M.H_CH)
    cp = np.polynomial.Polynomial(WP.SPECIFIC_HEAT)(t_b + 273.15)
    k_f = np.polynomial.Polynomial(WP.CONDUCTIVITY)(t_b + 273.15)
    mu = np.polynomial.Polynomial(WP.VISCOSITY)(t_b + 273.15)
    q_per_len = meta["mdot_cell"] * cp * np.gradient(t_b, wx)          # W/m into the water
    d_h = 2 * M.W_CH * M.H_CH / (M.W_CH + M.H_CH)
    nu = q_per_len / ((M.W_CH + 2 * M.H_CH) * (t_wall - t_b)) * d_h / k_f
    pr = float(np.mean(mu * cp / k_f))
    np.savetxt(RES / f"local_nu_{tag}.csv", np.column_stack([wx, t_b, t_wall, nu]), delimiter=",",
               header="x_m,T_bulk_C,T_wall_C,Nu_local", comments="")

    i_mid, j_mid, j_c = np.argmin(np.abs(xs - M.L / 2)), np.argmin(np.abs(wx - M.L / 2)), np.argmin(np.abs(ys - M.W / 2))
    np.savez_compressed(RES / f"fields_{tag}.npz", xs=xs, ys=ys, zs=zs, T_cross=T[i_mid], T_mid=T[:, j_c, :],
                        wy=wy, wz=wz, U_cross=U[j_mid], Re=meta["Re"], Pr=pr, d_h=d_h)
    ok = (wx > 0.3e-3) & (wx < M.L - 0.5e-3)
    print(f"mean Nu = {np.nanmean(nu[ok]):.2f}, L_th = {0.05 * meta['Re'] * pr * d_h * 1e3:.1f} mm")


if __name__ == "__main__":
    main()
