"""Fluent CHT case for the Qu & Mudawar (2002) unit cell, run through PyFluent (model: paper Sec. 3.1).
Writes results/<tag>.json for each operating point.

    python case.py --mesh runs/unitcell_r1.msh --q 100 --re 890 [--re 226 ...] [--solid copper|silicon]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import ansys.fluent.core as pyfluent

import mesh as M
import water_props as WP

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
N_CH = 21
A_T = 1.0e-2 * 4.48e-2                                  # planform, Eq. (1)
PLATE_W = 1.0e-2                                        # heat-sink width the channels share
T_IN = 288.15                                           # inlet, K (--t-in overrides)
RHO = WP.RHO_IN                                         # constant water density at the inlet T
SOLIDS = {"copper": 401.0, "silicon": 148.0}             # silicon at ~300 K (for the design study)
G: dict = {}                                            # geometry of the loaded mesh (mesh .json sidecar)


def d_h() -> float:
    return 2 * G["w_ch"] * G["h_ch"] / (G["w_ch"] + G["h_ch"])


def mu_poly(t: float) -> float:
    return float(np.polynomial.Polynomial(WP.VISCOSITY)(t))


def launch(cwd: Path, procs: int = 4, attempts: int = 3):
    """Start Fluent, retrying if the gRPC connection is refused."""
    for k in range(attempts):
        try:
            return pyfluent.launch_fluent(dimension=3, precision="double", processor_count=procs,
                                          mode="solver", ui_mode="no_gui", cleanup_on_exit=True, cwd=str(cwd))
        except Exception as e:                                  # LaunchFluentError / gRPC refusal
            if k == attempts - 1:
                raise
            print(f"  Fluent launch failed ({type(e).__name__}), retrying in 20 s", flush=True)
            time.sleep(20)


def setup(s, msh: Path, solid: str = "copper", lid: str = "polycarbonate") -> None:
    """lid = 'polycarbonate' (test rig, k = 0.2) or 'solid' (same material as the plate)."""
    G.clear()
    G.update(json.loads(msh.with_suffix(".json").read_text()))
    s.settings.file.read_mesh(file_name=str(msh))
    st = s.settings.setup
    st.models.energy.enabled = True
    st.models.viscous.model = "laminar"

    mats = st.materials
    mats.database.copy_by_name(type="fluid", name="water-liquid")
    w = mats.fluid["water-liquid"]
    w.density = {"option": "value", "value": RHO}
    w.viscosity = {"option": "polynomial", "polynomial": {"function_of": "temperature", "coefficients": WP.VISCOSITY}}
    w.thermal_conductivity = {"option": "polynomial",
                              "polynomial": {"function_of": "temperature", "coefficients": WP.CONDUCTIVITY}}
    w.specific_heat = {"option": "polynomial",
                       "polynomial": {"function_of": "temperature", "coefficients": WP.SPECIFIC_HEAT}}
    for name, k in ((solid, SOLIDS[solid]), ("polycarbonate", 0.2), ("slot-air", 0.0263)):
        if name not in mats.solid:
            mats.solid.create(name)
        mats.solid[name].thermal_conductivity = {"option": "value", "value": k}

    cz = st.cell_zone_conditions
    cz.fluid["water"].general.material = "water-liquid"
    cz.solid["copper"].general.material = solid          # zone name stays 'copper'
    cz.solid["cover"].general.material = "polycarbonate" if lid == "polycarbonate" else solid
    if G["slots"]:
        cz.solid["air"].general.material = "slot-air"

    bc = st.boundary_conditions
    bc.wall["cover-top"].thermal.thermal_condition = "Convection"
    bc.wall["cover-top"].thermal.heat_transfer_coeff = {"option": "value", "value": 10.0}
    bc.wall["cover-top"].thermal.free_stream_temp = {"option": "value", "value": 298.15}
    bc.velocity_inlet["inlet"].thermal.temperature = {"option": "value", "value": T_IN}
    bc.pressure_outlet["outlet"].momentum.gauge_pressure = {"option": "value", "value": 0.0}

    meth = s.settings.solution.methods
    meth.p_v_coupling.flow_scheme = "Coupled"
    meth.spatial_discretization.discretization_scheme = {
        "mom": "second-order-upwind", "pressure": "second-order", "temperature": "second-order-upwind"}
    meth.pseudo_time_method.formulation.coupled_solver = "global-time-step"
    res = s.settings.solution.monitor.residual
    res.options.plot = False
    for eq, crit in (("continuity", 1e-7), ("x-velocity", 1e-7), ("y-velocity", 1e-7),
                     ("z-velocity", 1e-7), ("energy", 1e-10)):
        res.equations[eq].absolute_criteria = crit
    add_reports(s)


def set_point(s, q_eff_wcm2: float, re: float | None = None, flow_ml_min: float | None = None) -> dict:
    """Set inlet velocity and heater flux. Test rig: flow from Re, q_bot from Eq. 14.
    Design: flow_ml_min = total plate flow (ml/min), q_bot = q_eff (W/cm^2)."""
    a_ch = G["w_ch"] * G["h_ch"]
    if G["slots"]:
        u_in = re * mu_poly(T_IN) / (RHO * d_h())
        q_bot = q_eff_wcm2 * 1e4 * A_T / (N_CH * G["a_bot"])
        n_ch = N_CH
    else:
        n_ch = PLATE_W / G["w"]                                   # channels per cm (may be fractional)
        u_in = flow_ml_min / 6e7 / n_ch / a_ch
        re = RHO * u_in * d_h() / mu_poly(T_IN)
        q_bot = q_eff_wcm2 * 1e4
    q_cell = q_bot * G["a_bot"]
    bc = s.settings.setup.boundary_conditions
    bc.velocity_inlet["inlet"].momentum.velocity_magnitude = {"option": "value", "value": u_in}
    bc.wall["heater"].thermal.thermal_condition = "Heat Flux"
    bc.wall["heater"].thermal.heat_flux = {"option": "value", "value": q_bot}
    t_out_est = T_IN + q_cell / (RHO * u_in * a_ch * 4180)
    bc.pressure_outlet["outlet"].thermal.backflow_total_temperature = {"option": "value", "value": t_out_est}
    return dict(u_in=u_in, q_bot=q_bot, q_cell=q_cell, Re=re, n_channels=n_ch,
                flow_ml_min=u_in * n_ch * a_ch * 6e7, geometry=dict(G))


def tc_line(s) -> tuple[np.ndarray, np.ndarray]:
    """y-averaged T along x on the thermocouple plane z = H_th (interpolated between cell layers)."""
    svd = s.fields.solution_variable_data
    xs, zs, ts = [], [], []
    for zone in (("copper", "air") if G["slots"] else ("copper",)):
        c = np.asarray(svd.get_data(variable_name="SV_CENTROID", zone_names=[zone], domain_name="mixture")[zone]).reshape(-1, 3)
        t = np.asarray(svd.get_data(variable_name="SV_T", zone_names=[zone], domain_name="mixture")[zone])
        xs.append(c[:, 0]); zs.append(c[:, 2]); ts.append(t)
    x, z, t = map(np.concatenate, (xs, zs, ts))
    zu = np.unique(np.round(z, 9))
    z_lo, z_hi = zu[zu < M.H_TH].max(), zu[zu > M.H_TH].min()
    xu = np.unique(np.round(x, 9))
    lo = np.array([t[(np.round(x, 9) == xv) & (np.round(z, 9) == z_lo)].mean() for xv in xu])
    hi = np.array([t[(np.round(x, 9) == xv) & (np.round(z, 9) == z_hi)].mean() for xv in xu])
    f = (M.H_TH - z_lo) / (z_hi - z_lo)
    return xu, lo + f * (hi - lo)


REPORTS = {   # name: (kind, report_type, field, locations)
    "dp_ch": ("surface", "surface-areaavg", "pressure", ["inlet"]),          # outlet is 0 Pa gauge
    "pt_in": ("surface", "surface-areaavg", "total-pressure", ["inlet"]),
    "pt_out": ("surface", "surface-areaavg", "total-pressure", ["outlet"]),
    "t_out": ("surface", "surface-massavg", "temperature", ["outlet"]),
    "t_heater_avg": ("surface", "surface-areaavg", "temperature", ["heater"]),
    "t_heater_max": ("surface", "surface-facetmax", "temperature", ["heater"]),
    "t_chbase_avg": ("surface", "surface-areaavg", "temperature", ["copper-water"]),
    "q_heater_cell": ("flux", "flux-heattransfer", None, ["heater"]),
    "q_top_cell": ("flux", "flux-heattransfer", None, ["cover-top"]),
    "mdot_cell": ("flux", "flux-massflow", None, ["outlet"]),
}


def add_reports(s) -> None:
    rd = s.settings.solution.report_definitions
    for name, (kind, rtype, field, locs) in REPORTS.items():
        group = getattr(rd, kind)
        if name not in group:
            group.create(name)
        if kind == "surface":
            group[name] = {"report_type": rtype, "field": field, "surface_names": locs}
        else:
            group[name] = {"report_type": rtype, "boundaries": locs}


def scalars(s) -> dict:
    vals = s.settings.solution.report_definitions.compute(report_defs=list(REPORTS))
    out = {k: float(v[0]) for d in vals for k, v in d.items()}
    out["mdot_cell"] = abs(out["mdot_cell"])
    return out


def solve(s, max_iter: int = 2000, chunk: int = 100, tol_t: float = 0.005, tol_p: float = 1e-4) -> dict:
    """Iterate in chunks until the TC-line temperatures and dP stop changing."""
    s.settings.solution.initialization.hybrid_initialize()
    prev, n = None, 0
    while n < max_iter:
        s.settings.solution.run_calculation.iterate(iter_count=chunk)
        n += chunk
        sc = scalars(s)
        t =tc_line(s)[1] if G["slots"] else np.array([sc["t_heater_max"], sc["t_heater_avg"], sc["t_out"]])
        if prev is not None:
            dt = np.abs(t - prev[0]).max()
            dp = abs(sc["dp_ch"] - prev[1]) / max(abs(sc["dp_ch"]), 1e-9)
            print(f"    iter {n}: max dT_tc change {dt:.4f} K, dP change {dp:.2e}, T_out {sc['t_out'] - 273.15:.3f} C",
                  flush=True)
            if dt < tol_t and dp < tol_p:
                break
        prev = (t, sc["dp_ch"])
    sc["iterations"] = n
    return sc


def run_point(s, q: float, re: float | None, tag: str, solid: str, save_case: bool,
              flow_ml_min: float | None = None) -> dict:
    t0 = time.time()
    pt = set_point(s, q, re, flow_ml_min)
    sc = solve(s)
    if G["slots"]:
        x, t = tc_line(s)
        t_tc = np.interp(np.array(M.X_TC), x, t)
    else:
        x = t = t_tc = np.full(1, np.nan)
    q_cell = pt["q_cell"]
    vdot = pt["flow_ml_min"] / 6e7
    dp_minor = 2.0 * 0.5 * RHO * pt["u_in"] ** 2            # plenum losses, 2 velocity heads
    q_total = q_cell * pt["n_channels"]
    out = dict(tag=tag, solid=solid, q_eff_Wcm2=q, t_in_C=T_IN - 273.15, **pt, **sc,
               pumping_power_W=vdot * sc["dp_ch"], pumping_power_with_plenum_W=vdot * (sc["dp_ch"] + dp_minor),
               r_th_max_KW=(sc["t_heater_max"] - T_IN) / q_total if q else None,
               r_th_avg_KW=(sc["t_heater_avg"] - T_IN) / q_total if q else None,
               q_total_W=q_total,
               t_tc_C=(t_tc - 273.15).tolist(), x_line=x.tolist(), t_line_C=(t - 273.15).tolist(),
               # Q_water / Q_in
               energy_balance=(sc["mdot_cell"] * float(np.mean(np.polynomial.Polynomial(WP.SPECIFIC_HEAT)(
                   np.linspace(T_IN, sc["t_out"], 11)))) * (sc["t_out"] - T_IN) / q_cell) if q else None,
               wall_time_s=time.time() - t0)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=1))
    if save_case:
        s.settings.file.write_case_data(file_name=str(HERE / "runs" / f"{tag}.cas.h5"))
    print(f"  {tag}: dP_ch = {sc['dp_ch'] / 1e5:.4f} bar, T_out-T_in = {sc['t_out'] - T_IN:.2f} K, "
          f"T_tc-T_in = {np.round(t_tc - T_IN, 2)}, heater max {sc['t_heater_max'] - 273.15:.1f} C, "
          f"{sc['iterations']} it, {out['wall_time_s']:.0f} s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--q", type=float, nargs="+", required=True, help="q_eff in W/cm2 (one per --re, or one for all)")
    ap.add_argument("--re", type=float, nargs="+", help="validation mode (test-rig mesh)")
    ap.add_argument("--flow", type=float, nargs="+", help="design mode: total plate flow, ml/min")
    ap.add_argument("--solid", default="copper")
    ap.add_argument("--lid", default="polycarbonate", choices=("polycarbonate", "solid"))
    ap.add_argument("--t-in", type=float, default=15.0, help="inlet temperature, C")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-case", action="store_true")
    a = ap.parse_args()
    global T_IN, RHO
    T_IN = a.t_in + 273.15
    RHO = WP.water_at(a.t_in).rho
    msh = Path(a.mesh).resolve()
    pts = a.re or a.flow
    qs = a.q * len(pts) if len(a.q) == 1 else a.q
    s = launch(msh.parent, a.procs)
    try:
        setup(s, msh, a.solid, a.lid)
        for q, v in zip(qs, pts):
            if a.re:
                tag = f"{a.tag or msh.stem}_{a.solid}_q{q:g}_re{v:g}"
                run_point(s, q, v, tag, a.solid, a.save_case)
            else:
                tag = f"{a.tag or msh.stem}_{a.solid}_q{q:g}_flow{v:g}"
                run_point(s, q, None, tag, a.solid, a.save_case, flow_ml_min=v)
    finally:
        s.exit()


if __name__ == "__main__":
    main()
