"""Grid Convergence Index (Celik et al. 2008, J. Fluids Eng. 130:078001) on three meshes.

    python gci.py [q] [Re]      (defaults 100 W/cm2, Re 890)
Needs results/unitcell_r{0.67,1,1.5}_copper_q{q}_re{Re}.json. Writes results/gci_q{q}_re{Re}.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "results"
MESHES = {"unitcell_r1.5": 297891, "unitcell_r1": 89760, "unitcell_r0.67": 29820}   # fine, medium, coarse
VOLUME = 0.044764 * 467e-6 * 19.05e-3


def gci3(phi1, phi2, phi3, h1, h2, h3, fs=1.25):
    """phi1 = finest. Returns p, extrapolated value, GCI_fine (%), asymptotic ratio, oscillatory flag."""
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = phi2 - phi1, phi3 - phi2
    s = np.sign(e32 / e21) if e21 != 0 else 1.0
    p = 1.0
    for _ in range(100):                                    # fixed-point iteration, Celik eq. 3
        q = np.log((r21**p - s) / (r32**p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    phi_ext = (r21**p * phi1 - phi2) / (r21**p - 1)
    ea21 = abs((phi1 - phi2) / phi1)
    gci_fine = fs * ea21 / (r21**p - 1)
    gci_med = fs * abs((phi2 - phi3) / phi2) / (r32**p - 1)
    asym = gci_med / (r21**p * gci_fine) if gci_fine else np.nan   # ~1 => asymptotic range
    oscill = e21 * e32 < 0
    return p, phi_ext, 100 * gci_fine, asym, oscill


def main() -> None:
    q = float(sys.argv[1]) if len(sys.argv) > 1 else 100
    re = float(sys.argv[2]) if len(sys.argv) > 2 else 890
    runs = {m: json.loads((RES / f"{m}_copper_q{q:g}_re{re:g}.json").read_text()) for m in MESHES}
    h = {m: (VOLUME / n) ** (1 / 3) for m, n in MESHES.items()}
    f, md, c = MESHES
    # temperatures as rise above the inlet
    qty ={"dP_ch [bar]": lambda r: r["dp_ch"] / 1e5,
           "T_out-T_in [K]": lambda r: r["t_out"] - 288.15,
           "T_heater,max-T_in [K]": lambda r: r["t_heater_max"] - 288.15,
           "T_heater,avg-T_in [K]": lambda r: r["t_heater_avg"] - 288.15}
    for i in range(4):
        qty[f"T_tc{i + 1}-T_in [K]"] = (lambda i: lambda r: r["t_tc_C"][i] - 15.0)(i)
    rows = []
    for name, get in qty.items():
        p1, p2, p3 = get(runs[f]), get(runs[md]), get(runs[c])
        p, ext, g, asym, osc = gci3(p1, p2, p3, h[f], h[md], h[c])
        rows.append(dict(quantity=name, coarse=p3, medium=p2, fine=p1, extrapolated=ext, order_p=p,
                         GCI_fine_pct=g, asymptotic_ratio=asym, oscillatory=osc,
                         medium_vs_ext_pct=100 * (p2 - ext) / ext))
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"gci_q{q:g}_re{re:g}.csv", index=False, float_format="%.5g")
    print(f"GCI at q = {q:g} W/cm2, Re = {re:g}; cells {list(MESHES.values())}; "
          f"refinement ratio {h[md] / h[f]:.2f}, {h[c] / h[md]:.2f}")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
