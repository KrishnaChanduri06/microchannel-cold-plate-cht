"""Check that the inputs reproduce the paper's Reynolds numbers.

Run from analytical/:  python check_inputs.py
"""
from inputs import FLOW_RANGES, T_IN_C, Geometry, water_at

ML_MIN_TO_M3_S = 1e-6 / 60


def reynolds(flow_ml_min: float, geo: Geometry, rho: float, mu: float) -> float:
    """Re = rho u D_h / mu, u = total flow / (N * channel area)."""
    u = flow_ml_min * ML_MIN_TO_M3_S / (geo.n_channels * geo.area_ch)
    return rho * u * geo.d_h / mu


def main() -> None:
    geo = Geometry()
    w = water_at(T_IN_C)

    print(f"Unit-cell width W   = {geo.w_cell * 1e6:.0f} um")
    print(f"Hydraulic diameter  = {geo.d_h * 1e6:.1f} um")
    print(f"Aspect ratio alpha  = {geo.aspect:.3f}")
    print(f"Water @ {T_IN_C} C: rho={w.rho:.1f}, mu={w.mu:.3e}, "
          f"cp={w.cp:.0f}, k={w.k:.3f}, Pr={w.pr:.2f}")
    print(f"Planform area A_t   = {geo.planform_area * 1e4:.3f} cm^2")
    print(f"N * A_bot           = {geo.bottom_area_total * 1e4:.3f} cm^2\n")

    for q_eff, data in FLOW_RANGES.items():
        power = q_eff * geo.planform_area
        print(f"q_eff = {q_eff / 1e4:.0f} W/cm^2 -> P_W = {power:.0f} W")
        for flow, re_paper in zip(data["ml_min"], data["re_paper"]):
            re_ours = reynolds(flow, geo, w.rho, w.mu)
            err = 100 * (re_ours - re_paper) / re_paper
            print(f"  {flow:7.1f} ml/min: Re ours = {re_ours:6.0f}, "
                  f"paper = {re_paper:5d}  ({err:+.1f} %)")


if __name__ == "__main__":
    main()
