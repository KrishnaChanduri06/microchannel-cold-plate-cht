"""All figures for the README and report, drawn from saved results (no solver needed).

    .venv/Scripts/python make_figures.py        -> figures/*.png (300 dpi) and figures/*.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "analytical"), str(ROOT / "cfd")]
from compare_benchmark import sweep  # noqa: E402
from design_analysis import load as load_design, pareto  # noqa: E402
from inputs import Geometry  # noqa: E402
from model import flow_from_re, run  # noqa: E402

DATA = ROOT / "data/digitised"
CFD = ROOT / "cfd/results"
FIGS = ROOT / "figures"
GEO = Geometry()
CASES = {100: (226, 890, 1454), 200: (442, 864, 1220)}

INK, INK2, GRID = "#1a1a1a", "#555555", "#e4e4e4"
RE_RAMP = ("#6da7ec", "#256abf", "#0d366b")                 # ordinal blue, low -> high Re
TC_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")      # ordinal blue, TC1 -> TC4
SOLID_C = {"copper": "#eb6834", "silicon": "#2a78d6"}
RE_MARK = ("o", "s", "^")
QEFF = r"$q^{\prime\prime}_{\mathrm{eff}}$"

mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 9, "axes.linewidth": 0.7, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2, "ytick.minor.size": 2, "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.3, "lines.markersize": 4.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
})


def save(fig, name: str) -> None:
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", dpi=300)
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)


def panel_label(ax, text: str) -> None:
    ax.text(0.02, 0.97, text, transform=ax.transAxes, va="top", ha="left", color=INK)


def cfd_point(q: float, re: float) -> dict:
    return json.loads((CFD / f"unitcell_r1_copper_q{q:g}_re{re:g}.json").read_text())


def tc_line_validation() -> None:
    f7 = pd.read_csv(DATA / "fig7_tc_along_x.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    for ax, q, curves, lab in zip(axes, (100, 200), ("fig7a_num_curves_100", "fig7b_num_curves_200"), "ab"):
        num = pd.read_csv(DATA / f"{curves}.csv")
        for k, re in enumerate(CASES[q]):
            c = RE_RAMP[k]
            r = cfd_point(q, re)
            x, t = np.array(r["x_line"]) * 1e3, np.array(r["t_line_C"])
            ax.plot(x, t, "-", color=c, lw=1.4, zorder=3)
            ax.plot(num.x * 1e3, num[f"re{re}"], ":", color=c, lw=1.1, zorder=2)
            one = run(flow_from_re(re, GEO), q * 1e4, GEO)
            ax.plot(np.array(GEO.x_tc) * 1e3, one.t_tc, "x", color=c, ms=5, mew=1.0, zorder=4)
            e = f7[(f7.q_eff_Wcm2 == q) & (f7.Re == re)]
            ax.plot(e.x_m * 1e3, e.T_th_C, RE_MARK[k], ms=5, mfc="white", mec=c, mew=1.1, ls="none", zorder=5)
            ax.text(45.6, t[-1], f"$Re$ = {re}", va="center", ha="left", fontsize=8, color=INK2)
        ax.set_xlim(0, 44.8)
        ax.set_xlabel(r"$x$ (mm)")
        panel_label(ax, f"({lab}) {QEFF} = {q} W/cm$^2$")
    axes[0].set_ylabel(r"$T_{\mathrm{th}}$ (°C)")
    axes[0].set_ylim(30, 100)
    axes[1].set_ylim(50, 130)
    handles = [Line2D([], [], color=INK, marker="o", mfc="white", ls="none", label="Experiment"),
               Line2D([], [], color=INK, ls="-", label="Present CFD"),
               Line2D([], [], color=INK, ls=":", label="CFD, Qu & Mudawar (2002)"),
               Line2D([], [], color=INK, marker="x", ls="none", label="1-D model")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=4.5)
    save(fig, "tc_line_validation")


def pressure_drop() -> None:
    f4 = pd.read_csv(DATA / "fig4a_dp_vs_re.csv")
    f4b = pd.read_csv(DATA / "fig4b_dp_num_vs_exp.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, q, lab in zip(axes, (100, 200), "ab"):
        e = f4[f4.q_eff_Wcm2 == q]
        line = sweep(q, np.linspace(e.Re.min(), e.Re.max(), 50))
        ax.plot(line.Re, line.dP, "-", color=RE_RAMP[0], lw=1.2)
        ax.plot(line.Re, line.dP_wall, "--", color=RE_RAMP[2], lw=1.2)
        ax.plot(e.Re, e.dP_bar, "o", mfc="white", mec=INK, mew=0.9, ms=4)
        n = f4b[(f4b.q_eff_Wcm2 == q) & (f4b.source == "num")]
        ax.plot(n.Re, n.dP_bar, "D", mfc="white", mec=SOLID_C["copper"], mew=1.0, ms=4.5)
        pts = [cfd_point(q, re) for re in CASES[q]]
        ax.plot([p["Re"] for p in pts], [p["dp_ch"] / 1e5 for p in pts], "s", color=SOLID_C["silicon"], ms=4.5)
        ax.set_xlabel(r"$Re$")
        ax.set_xlim(0, 1800)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(500))
        panel_label(ax, f"({lab}) {QEFF} = {q} W/cm$^2$")
    axes[0].set_ylabel(r"$\Delta P$ (bar)")
    axes[0].set_ylim(0, 1.25)
    handles = [Line2D([], [], color=INK, marker="o", mfc="white", ls="none", label="Experiment"),
               Line2D([], [], color=SOLID_C["silicon"], marker="s", ls="none", label="Present CFD (channel)"),
               Line2D([], [], color=SOLID_C["copper"], marker="D", mfc="white", ls="none",
                      label="CFD, Qu & Mudawar (2002)"),
               Line2D([], [], color=RE_RAMP[0], ls="-", label=r"1-D, bulk $\mu$"),
               Line2D([], [], color=RE_RAMP[2], ls="--", label=r"1-D, $(\mu_w/\mu_b)^{0.58}$ correction")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=7.5,
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    save(fig, "pressure_drop")


def water_temperature_rise() -> None:
    f5 = pd.read_csv(DATA / "fig5_dT_vs_re.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for k, q in enumerate((100, 200)):
        e = f5[f5.q_eff_Wcm2 == q]
        line = sweep(q, np.linspace(e.Re.min(), e.Re.max(), 60))
        c = RE_RAMP[2 * k]
        ax.plot(line.Re, line.dT, "-", color=c, lw=1.1)
        ax.plot(e.Re, e.dT_out_in_C, RE_MARK[k], mfc="white", mec=c, mew=0.9, ms=4, ls="none")
        pts = [cfd_point(q, re) for re in CASES[q]]
        ax.plot([p["Re"] for p in pts], [p["t_out"] - 288.15 for p in pts], "x", color=INK, ms=5, mew=1.0)
        ax.text(line.Re.iloc[0] + 60, line.dT.iloc[0], f"{q} W/cm$^2$", fontsize=7.5, color=INK2, va="center")
    ax.set(xlabel=r"$Re$", ylabel=r"$T_{\mathrm{out}} - T_{\mathrm{in}}$ (K)", xlim=(0, 1800), ylim=(0, 75))
    handles = [Line2D([], [], color=INK, marker="o", mfc="white", ls="none", label="Experiment"),
               Line2D([], [], color=INK, ls="-", label="Energy balance"),
               Line2D([], [], color=INK, marker="x", ls="none", label="Present CFD")]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    save(fig, "water_temperature_rise")


def tc_vs_re() -> None:
    f6 = pd.read_csv(DATA / "fig6_tc_vs_re.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, q, lab in zip(axes, (100, 200), "ab"):
        e6 = f6[f6.q_eff_Wcm2 == q]
        line = sweep(q, np.linspace(e6.Re.min(), e6.Re.max(), 60))
        pts = [cfd_point(q, re) for re in CASES[q]]
        for loc in (1, 2, 3, 4):
            c = TC_RAMP[loc - 1]
            e = e6[e6.location == loc]
            ax.plot(line.Re, line[f"tc{loc}"], "--", color=c, lw=1.0)
            ax.plot(e.Re, e.dT_th_in_C, "o", mfc="white", mec=c, mew=0.9, ms=3.6, ls="none")
            ax.plot([p["Re"] for p in pts], [p["t_tc_C"][loc - 1] - 15.0 for p in pts], "s", color=c, ms=4.2)
        ax.set_xlabel(r"$Re$")
        ax.set_xlim(0, 1800)
        panel_label(ax, f"({lab}) {QEFF} = {q} W/cm$^2$")
    axes[0].set_ylabel(r"$T_{\mathrm{th}} - T_{\mathrm{in}}$ (K)")
    axes[0].set_ylim(10, 110)
    axes[1].set_ylim(30, 135)
    handles = [Line2D([], [], color=TC_RAMP[i], marker="s", ls="none", label=f"TC{i + 1}") for i in range(4)]
    handles += [Line2D([], [], color=INK, marker="o", mfc="white", ls="none", label="Experiment"),
                Line2D([], [], color=INK, marker="s", ls="none", label="Present CFD"),
                Line2D([], [], color=INK, ls="--", label="1-D model")]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 1.0), columnspacing=1.1,
               handletextpad=0.3)
    fig.tight_layout()
    save(fig, "tc_vs_re")


def local_nusselt() -> None:
    d = np.loadtxt(CFD / "local_nu_unitcell_r1_copper_q100_re890.csv", delimiter=",", skiprows=1)
    f = np.load(CFD / "fields_unitcell_r1_copper_q100_re890.npz")
    x, nu = d[:, 0] * 1e3, d[:, 3]
    ok = (x > 0.3) & (x < 43.5)                                   # last cells: outlet boundary
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for xc in (10.794, 22.382, 33.970):
        ax.axvspan(xc - 0.8, xc + 0.8, color="#efefef", lw=0, zorder=0)
    ax.plot(x[ok], nu[ok], "-", color=RE_RAMP[1], lw=1.4, label=r"Present CFD, $Re$ = 890")
    ax.axhline(5.30, color=INK, ls="--", lw=1.0, label=r"Fully developed, 3 walls ($Nu$ = 5.30)")
    ax.set(xlabel=r"$x$ (mm)", ylabel=r"$Nu = h D_h / k$", xlim=(0, 44.8), ylim=(0, 22))
    l_th = 0.05 * float(f["Re"]) * float(f["Pr"]) * float(f["d_h"]) * 1e3
    ax.text(43.8, 20.5, rf"$L_{{\mathrm{{th}}}} \approx$ {l_th:.0f} mm $= {l_th / 44.764:.1f}\,L$", ha="right",
            va="top", fontsize=8, color=INK2)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout()
    save(fig, "local_nusselt")


def field_sections() -> None:
    f = np.load(CFD / "fields_unitcell_r1_copper_q100_re890.npz")
    ys, zs, xs = f["ys"] * 1e6, f["zs"] * 1e3, f["xs"] * 1e3
    kz = zs <= 7.35
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.0), gridspec_kw=dict(width_ratios=[1, 1]))
    im = axes[0].pcolormesh(ys, zs[kz], f["T_cross"][:, kz].T, shading="nearest", cmap="inferno", rasterized=True)
    fig.colorbar(im, ax=axes[0], label=r"$T$ (°C)", pad=0.03)
    axes[0].set(xlabel=r"$y$ (μm)", ylabel=r"$z$ (mm)")
    im2 = axes[1].pcolormesh(f["wy"] * 1e6, f["wz"] * 1e3, f["U_cross"].T, shading="nearest", cmap="viridis",
                             rasterized=True)
    fig.colorbar(im2, ax=axes[1], label=r"$u$ (m/s)", pad=0.03)
    axes[1].set(xlabel=r"$y$ (μm)", ylabel=r"$z$ (mm)")
    for ax in axes:
        ax.grid(False)
        ax.minorticks_off()
    for ax, lab in zip(axes, "ab"):
        ax.set_title(f"({lab})", loc="left", fontsize=9, pad=3)
    fig.tight_layout()
    save(fig, "cross_section")

    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    im = ax.pcolormesh(xs, zs[kz], f["T_mid"][:, kz].T, shading="nearest", cmap="inferno", rasterized=True)
    fig.colorbar(im, ax=ax, label=r"$T$ (°C)", pad=0.015)
    ax.axhline(3.175, color="white", ls=":", lw=0.7)
    ax.text(0.8, 3.33, "thermocouple plane", color="white", fontsize=7.5)
    ax.set(xlabel=r"$x$ (mm)", ylabel=r"$z$ (mm)")
    ax.grid(False)
    ax.minorticks_off()
    fig.tight_layout()
    save(fig, "midplane_temperature")


def design_figures() -> None:
    df = load_design()
    d = df[df.q == 100]
    shapes = {100: "o", 150: "s", 231: "^", 350: "D"}

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    for solid in ("copper", "silicon"):
        c = SOLID_C[solid]
        s = d[d.solid == solid]
        p = pareto(s)
        ax.plot(p.P_pump_W, p.R_th, "-" if solid == "copper" else "--", color=c, lw=1.1, zorder=2)
        for r in s.itertuples():
            filled = r.w_fin == 100
            ax.plot(r.P_pump_W, r.R_th, shapes[r.w_ch], ms=5.5, mew=1.0, mec=c, mfc=c if filled else "white",
                    ls="none", zorder=3)
    ax.set_xscale("log")
    ax.set(xlabel=r"Pumping power, $\dot{V}\,\Delta P$ (W)",
           ylabel=r"$R_{\mathrm{th}} = (T_{\mathrm{base,max}} - T_{\mathrm{in}})/Q$ (K/W)", ylim=(0.025, 0.18))
    handles = [Line2D([], [], color=SOLID_C["copper"], ls="-", label="Copper (Pareto front)"),
               Line2D([], [], color=SOLID_C["silicon"], ls="--", label="Silicon (Pareto front)")]
    handles += [Line2D([], [], color=INK2, marker=m, ls="none", label=rf"$W_{{\mathrm{{ch}}}}$ = {w} μm")
                for w, m in shapes.items()]
    handles += [Line2D([], [], color=INK2, marker="o", ls="none", label="fin 100 μm (filled)"),
                Line2D([], [], color=INK2, marker="o", mfc="white", ls="none", label="fin 236 μm (open)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    save(fig, "pareto")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for solid in ("copper", "silicon"):
        s = df[df.solid == solid]
        axes[0].plot(s.R_th, s.oneD_R_th, "o", color=SOLID_C[solid], ms=4, label=solid.capitalize())
        axes[1].plot(s.dP_bar, s.oneD_dP_bar, "o", color=SOLID_C[solid], ms=4, label=solid.capitalize())
    for ax, lim, lab in ((axes[0], 0.21, r"$R_{\mathrm{th}}$ (K/W)"), (axes[1], 3.6, r"$\Delta P$ (bar)")):
        v = np.array([0, lim])
        ax.plot(v, v, "-", color=INK, lw=0.8)
        ax.plot(v, 1.15 * v, ":", color=INK2, lw=0.8)
        ax.plot(v, 0.85 * v, ":", color=INK2, lw=0.8)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel=f"CFD {lab}", ylabel=f"1-D model {lab}", aspect="equal")
        step = 0.05 if lim < 1 else 1.0
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(step))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step))
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    axes[1].text(0.97, 0.03, r"dotted: $\pm$15 %", transform=axes[1].transAxes, ha="right", fontsize=7.5,
                 color=INK2)
    fig.tight_layout()
    save(fig, "parity_1d_cfd")

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for solid in ("copper", "silicon"):
        s = d[d.solid == solid]
        ax.plot(s.L_th_over_L, s["err_1D_Rth_%"], "o", color=SOLID_C[solid], ms=4, label=solid.capitalize())
    hi = d[d.Re > 1800]
    ax.plot(hi.L_th_over_L, hi["err_1D_Rth_%"], "o", ms=9, mfc="none", mec=INK2, mew=0.8,
            label=r"$Re > 1800$")
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set(xlabel=r"$L_{\mathrm{th}}/L = 0.05\,Re\,Pr\,D_h/L$", ylabel=r"1-D error in $R_{\mathrm{th}}$ (%)",
           ylim=(-5, 68))
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "error_vs_entry_length")


def main() -> None:
    tc_line_validation()
    pressure_drop()
    water_temperature_rise()
    tc_vs_re()
    local_nusselt()
    field_sections()
    design_figures()
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
