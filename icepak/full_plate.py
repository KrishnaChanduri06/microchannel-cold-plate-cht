"""Icepak (AEDT 2025 R2 Student) model of the Qu & Mudawar heat sink with N channels, via PyAEDT.
Constant water properties at --t-mean. The full plate (21 channels) exceeds the student cell limit
at a flow-resolving mesh; results use 7 channels.

    ../.venv-aedt/Scripts/python full_plate.py --fine 2 --cover 2 --channels 7 --iters 500
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("ANSYSEMSV_ROOT252", r"D:\ANSYS Inc\ANSYS Student\v252\AnsysEM")
import ansys.aedt.core as pyaedt  # noqa: E402

HERE = Path(__file__).resolve().parent
UM = 1e-3            # model units are mm; 1 um = 1e-3 mm
N = 21
W_W, W_CH, H_CH, H_W2, H_W1 = 118 * UM, 231 * UM, 713 * UM, 5637 * UM, 12700 * UM
H_COVER = H_W1        # mm, set by --cover
L, B = 44.764, 10.0                          # mm: channel length, heat-sink width (planform 1.0 cm)
W = 2 * W_W + W_CH
EDGE = (B - N * W) / 2                       # extra copper at each side wall (0.0965 mm)
X_TC = (5.0, 16.588, 28.176, 39.764)
SLOT_W, SLOT_H = 1.6, 3.175
SLOT_C = tuple(0.5 * (a + b) for a, b in zip(X_TC[:-1], X_TC[1:]))
D_H = 2 * W_CH * H_CH / (W_CH + H_CH) * 1e-3   # m


def water_props(t_c: float) -> dict:
    """Liquid water at t_c from CoolProp (shared with the analytical model)."""
    import sys
    sys.path.insert(0, str(HERE.parent / "analytical"))
    from inputs import water_at
    w = water_at(t_c)
    return dict(rho=w.rho, mu=w.mu, k=w.k, cp=w.cp)


def build(ipk, q_wcm2: float, re: float, t_mean_c: float) -> dict:
    m = ipk.modeler
    m.model_units = "mm"
    wp = water_props(t_mean_c)
    if "water_tmean" not in ipk.materials.material_keys:
        # copy of a library fluid so it keeps the fluid flag
        ipk.materials.duplicate_material("Water(@280K)", "water_tmean")
    wat = ipk.materials["water_tmean"]
    wat.thermal_conductivity = wp["k"]
    wat.mass_density = wp["rho"]
    wat.specific_heat = wp["cp"]
    wat.viscosity = wp["mu"]
    if "copper_401" not in ipk.materials.material_keys:
        cu = ipk.materials.add_material("copper_401")
        cu.thermal_conductivity = 401
        cu.mass_density = 8933
        cu.specific_heat = 385
    if "pc_lexan" not in ipk.materials.material_keys:
        pc = ipk.materials.add_material("pc_lexan")
        pc.thermal_conductivity = 0.2
        pc.mass_density = 1200
        pc.specific_heat = 1200
    if "slot_air" not in ipk.materials.material_keys:
        sa = ipk.materials.add_material("slot_air")
        sa.thermal_conductivity = 0.0263
        sa.mass_density = 1.2
        sa.specific_heat = 1007

    cu_mat, pc_mat, air_mat = "copper_401", "pc_lexan", "slot_air"

    # geometry: base with 3 slot cut-outs, 22 fins, cover
    base = m.create_box([0, 0, 0], [L, B, H_W2], name="base", material=cu_mat)
    slots = []
    for i, sc in enumerate(SLOT_C):
        s = m.create_box([sc - SLOT_W / 2, 0, 0], [SLOT_W, B, SLOT_H], name=f"slot{i + 1}", material=air_mat)
        slots.append(s)
    tools = [m.create_box([sc - SLOT_W / 2, 0, 0], [SLOT_W, B, SLOT_H], name=f"cut{i + 1}") for i, sc in enumerate(SLOT_C)]
    m.subtract(base, tools, keep_originals=False)
    fins = []
    for k in range(N + 1):
        if k == 0:
            y0, wy = 0.0, EDGE + W_W
        elif k == N:
            y0, wy = EDGE + N * W - W_W, EDGE + W_W
        else:
            y0, wy = EDGE + k * W - W_W, 2 * W_W
        fins.append(m.create_box([0, y0, H_W2], [L, wy, H_CH], name=f"fin{k:02d}", material=cu_mat))
    cover = m.create_box([0, 0, H_W2 + H_CH], [L, B, H_COVER], name="cover", material=pc_mat)

    # fluid region = exactly the heat-sink envelope (zero padding), filled with water
    m.edit_region_dimensions([0, 0, 0, 0, 0, 0])
    region = m["Region"]
    region.material_name = "water_tmean"

    # boundaries
    u_in = re * water_props(15.0)["mu"] / (water_props(15.0)["rho"] * D_H)
    xmin = min(region.faces, key=lambda f: f.center[0])
    xmax = max(region.faces, key=lambda f: f.center[0])
    ipk.assign_velocity_free_opening(xmin.id, boundary_name="inlet", temperature="15cel",
                                     velocity=[f"{u_in}m_per_sec", "0m_per_sec", "0m_per_sec"])
    ipk.assign_pressure_free_opening(xmax.id, boundary_name="outlet", temperature="25cel")
    q_total = q_wcm2 * (B / 10.0) * 4.48                       # W, planform share (B = 10 mm for all 21)
    # Slots split the bottom face into 4 pieces. 'Total Power' on a face list applies the full power
    # to each face, so assign per face, area-weighted: uniform flux P / A_copper (Eq. 14).
    bottoms = [f for f in base.faces if abs(f.center[2]) < 1e-6]
    a_tot = sum(f.area for f in bottoms)
    for i, f in enumerate(sorted(bottoms, key=lambda f: f.center[0]), start=1):
        ipk.assign_source(f.id, "Total Power", f"{q_total * f.area / a_tot}W", boundary_name=f"heater{i}")
    bottom = max(bottoms, key=lambda f: f.center[0])           # downstream piece (hottest) for the monitor
    top = max(cover.faces, key=lambda f: f.center[2])
    ipk.assign_stationary_wall_with_htc(top.id, name="cover_top", htc="10w_per_m2kel", ref_temperature="25cel")

    # monitors: thermocouples at mid-width, z = H_th, and a probe at the heater centre-line outlet end
    for i, x in enumerate(X_TC, start=1):
        ipk.monitor.assign_point_monitor([f"{x}mm", f"{B / 2}mm", f"{SLOT_H}mm"], monitor_name=f"TC{i}")
    # face monitors (area averages): outlet water T, inlet pressure, heater T; plus centre/edge probes
    ipk.monitor.assign_face_monitor(xmax.id, monitor_quantity="Temperature", monitor_name="outlet_T")
    ipk.monitor.assign_face_monitor(xmin.id, monitor_quantity="Pressure", monitor_name="inlet_p")
    ipk.monitor.assign_face_monitor(xmax.id, monitor_quantity="Pressure", monitor_name="outlet_p")
    ipk.monitor.assign_face_monitor(bottom.id, monitor_quantity="Temperature", monitor_name="heater_T_last_section")
    ipk.monitor.assign_face_monitor(xmax.id, monitor_quantity="MassFlowRate", monitor_name="outlet_mdot")
    ipk.monitor.assign_face_monitor(xmin.id, monitor_quantity="MassFlowRate", monitor_name="inlet_mdot")
    for yname, y in (("centre", B / 2), ("edge", EDGE + W / 2)):        # middle channel vs outermost channel
        ipk.monitor.assign_point_monitor([f"{L - 1.0}mm", f"{y}mm", f"{H_W2 + H_CH / 2}mm"],
                                         monitor_name=f"water_{yname}_xL")
        ipk.monitor.assign_point_monitor([f"{L / 2}mm", f"{y}mm", f"{H_W2 + H_CH / 2}mm"],
                                         monitor_quantity="Speed", monitor_name=f"speed_{yname}_mid")
        ipk.monitor.assign_point_monitor([f"{L - 1.0}mm", f"{y}mm", "0mm"], monitor_name=f"heater_{yname}_xL")
    return dict(u_in=u_in, q_total=q_total, water=wp, region=region.name, base=base.name)


def mesh_settings(ipk, fine: float = 1.0, max_x: float | None = None, max_z: float | None = None) -> None:
    """Global max cell sizes, mm. Y kept fine so the mesher resolves the 0.231 mm channels."""
    g = ipk.mesh.global_mesh_region
    g.manual_settings = True
    g.settings["MaxElementSizeX"] = f"{max_x or 0.5 / fine}mm"
    g.settings["MaxElementSizeY"] = f"{0.06 / fine}mm"
    g.settings["MaxElementSizeZ"] = f"{max_z or 0.15 / fine}mm"
    g.settings["MinElementsInGap"] = "4"
    g.settings["MinElementsOnEdge"] = "2"
    g.settings["MaxSizeRatio"] = "2"
    g.settings["MinGapX"] = "0.05mm"
    g.settings["MinGapY"] = "0.05mm"
    g.settings["MinGapZ"] = "0.05mm"
    g.update()


def postprocess(ipk, info: dict) -> dict:
    """Final monitor values and their change over the last 200 iterations."""
    out, hist = {}, {}
    for name, mon in ipk.monitor.all_monitors.items():
        try:
            v = mon.value()                                         # {0: {'Variation':..., qty: {'Value': [..]}}}
            d = v.get(0, v.get("0")) if isinstance(v, dict) else None
            q = next(k for k in d if k != "Variation") if d else None
            out[name] = d[q]["Value"][-1] if q else str(v)
        except Exception as e:                                      # noqa: BLE001
            out[name] = f"error: {e!r}"[:200]
    # monitor histories written by the solver: steady.<n>.<name>.out
    res_dir = next(Path(ipk.results_directory).rglob("*.restart"), None)
    for f in Path(ipk.results_directory).rglob("steady.*.out"):
        try:
            vals = [float(l.split()[-1]) for l in f.read_text(errors="ignore").splitlines()
                    if l.strip() and l.split()[0].isdigit()]
            if len(vals) > 200:
                hist[f.stem.split(".")[-1]] = dict(n_iter=len(vals), last=vals[-1],
                                                   change_last_200=vals[-1] - vals[-201])
        except Exception:                                           # noqa: BLE001
            pass
    out["history"] = hist
    out["restart_dir"] = str(res_dir) if res_dir else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=890)
    ap.add_argument("--q", type=float, default=100)
    ap.add_argument("--t-mean", type=float, default=20.3, help="C, water property temperature")
    ap.add_argument("--fine", type=float, default=1.0)
    ap.add_argument("--max-x", type=float, help="mm, overrides the axial max cell size")
    ap.add_argument("--max-z", type=float, help="mm, overrides the vertical max cell size")
    ap.add_argument("--cover", type=float, default=12.7, help="mm cover thickness (rig: 12.7)")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--channels", type=int, default=21, help="fewer channels for quick checks")
    ap.add_argument("--mesh-only", action="store_true")
    a = ap.parse_args()
    runs = HERE / "runs"
    runs.mkdir(exist_ok=True)
    global N, B, H_COVER
    H_COVER = a.cover
    if a.channels != N:
        B = a.channels * W + 2 * EDGE
        N = a.channels
    tag = (f"fullplate_q{a.q:g}_re{a.re:g}_f{a.fine:g}" + (f"_n{N}" if N != 21 else "")
           + (f"_x{a.max_x:g}" if a.max_x else "") + (f"_z{a.max_z:g}" if a.max_z else "")
           + (f"_c{a.cover:g}" if a.cover != 12.7 else ""))
    for old in runs.glob(f"{tag}.aedt*"):                     # start from a clean project
        if old.is_dir():
            import shutil
            shutil.rmtree(old, ignore_errors=True)
        else:
            old.unlink(missing_ok=True)
    t0 = time.time()
    ipk = pyaedt.Icepak(project=str(runs / f"{tag}.aedt"), design="full_plate", version="2025.2",
                        student_version=True, non_graphical=True, new_desktop=True, close_on_exit=True)
    try:
        info = build(ipk, a.q, a.re, a.t_mean)
        if a.fine > 0:
            mesh_settings(ipk, a.fine, a.max_x, a.max_z)
        setup = ipk.create_setup("steady")
        setup.props["Flow Regime"] = "Laminar"
        setup.props["Include Gravity"] = False
        setup.props["Convergence Criteria - Max Iterations"] = a.iters
        # unreachable residual targets: run a fixed iteration count, judge convergence on monitors
        setup.props["Convergence Criteria - Flow"] = "1e-14"
        setup.props["Convergence Criteria - Energy"] = "1e-20"
        setup.update()
        ipk.save_project()
        vlog = runs / "validation.log"
        valid = ipk.odesign.ValidateDesign(str(vlog))
        print("validation:", valid, vlog.read_text(errors="ignore") if vlog.exists() else "")
        if a.mesh_only:
            try:
                print("mesh generated:", ipk.mesh.generate_mesh("steady"))
            except Exception as e:                                  # noqa: BLE001
                print("mesh generation failed:", repr(e)[:150])
            import re
            for prof in Path(str(runs / f"{tag}.aedtresults")).rglob("*.profile"):
                txt = prof.read_text(errors="ignore")
                cells = re.findall(r"Cells\\', (\d+)", txt)
                print("profile", prof.name, "cells:", cells, "| limit message:" , "greater than the limit" in txt)
            return
        ipk.analyze(cores=4)
        ipk.save_project()
        info.update(results=postprocess(ipk, info), wall_time_s=time.time() - t0)
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / f"{tag}.json").write_text(json.dumps(info, indent=1, default=str))
        print(json.dumps(info, indent=1, default=str))
    finally:
        ipk.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    main()
