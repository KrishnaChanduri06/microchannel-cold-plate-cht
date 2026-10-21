"""1-D analytical model of the Qu & Mudawar (2002) micro-channel heat sink.

Fully developed laminar correlations, fin efficiency and a resistance network.
Assumptions are listed in ASSUMPTIONS.md.  Run:  python model.py
"""
from dataclasses import dataclass

import numpy as np

from inputs import K_COPPER, K_MINOR, T_IN_C, Geometry, water_at

ML_MIN_TO_M3_S = 1e-6 / 60
NU_3WALL = 5.30   # nusselt_fd.py, alpha = 0.324, top wall adiabatic


def f_re_shah_london(a: float) -> float:
    """Fanning f*Re, fully developed laminar, rectangular duct (Shah & London 1978)."""
    return 24 * (1 - 1.3553 * a + 1.9467 * a**2 - 1.7012 * a**3 + 0.9564 * a**4 - 0.2537 * a**5)


def hagenbach_k(a: float) -> float:
    """Incremental pressure defect K(inf) for developing flow (Shah & London 1978)."""
    return 0.6796 + 1.2197 * a + 3.3089 * a**2 - 9.5921 * a**3 + 8.9089 * a**4 - 2.9959 * a**5


@dataclass
class Result:
    flow_ml_min: float
    re: float                 # inlet-property Re (paper's definition)
    u_m: float                # mean channel velocity, m/s
    t_out: float              # outlet water temperature, C
    dp_ch: float              # channel friction + developing-flow defect, Pa
    dp_minor: float           # plenum contraction/expansion, Pa
    dp_ch_local: float        # dp_ch with viscosity integrated along x
    dp_ch_wall: float         # dp_ch with wall viscosity correction (mu_w/mu_b)^0.58
    h: float                  # W/m^2 K
    eta_f: float              # fin efficiency
    r_cond: float             # K/W, full copper base
    r_conv: float             # K/W
    r_cal: float              # K/W, 1/(m_dot cp)
    t_tc: np.ndarray          # thermocouple temperatures at x_tc, C
    t_max: float              # peak base temperature (outlet end), C
    l_h: float                # hydrodynamic entry length, m
    l_th: float               # thermal entry length, m

    @property
    def dp(self) -> float:
        return self.dp_ch + self.dp_minor

    @property
    def pumping_power(self) -> float:
        return self.flow_ml_min * ML_MIN_TO_M3_S * self.dp


def flow_from_re(re: float, geo: Geometry, t_in: float = T_IN_C) -> float:
    """Flow rate (ml/min) that gives inlet Re = re."""
    w = water_at(t_in)
    u = re * w.mu / (w.rho * geo.d_h)
    return u * geo.n_channels * geo.area_ch / ML_MIN_TO_M3_S


def run(flow_ml_min: float, q_eff: float, geo: Geometry = Geometry(), k_s: float = K_COPPER,
        nu: float = NU_3WALL, t_in: float = T_IN_C) -> Result:
    """Solve the 1-D model for one operating point (q_eff in W/m^2)."""
    vdot = flow_ml_min * ML_MIN_TO_M3_S
    q = q_eff * geo.planform_area                       # total heat, W  (Eq. 1)

    w_in = water_at(t_in)
    m_dot = w_in.rho * vdot
    t_out = t_in + q / (m_dot * w_in.cp)                # energy balance (Eq. 23)
    w_m = water_at(0.5 * (t_in + t_out))                # mean-temperature properties

    a = geo.aspect
    u_in = vdot / (geo.n_channels * geo.area_ch)
    re_in = w_in.rho * u_in * geo.d_h / w_in.mu
    u_m = m_dot / (w_m.rho * geo.n_channels * geo.area_ch)
    re_m = w_m.rho * u_m * geo.d_h / w_m.mu

    # pressure drop: friction (mean viscosity) + Hagenbach developing-flow defect
    f = f_re_shah_london(a) / re_m
    dyn = 0.5 * w_m.rho * u_m**2
    dp_ch = (4 * f * geo.length / geo.d_h + hagenbach_k(a)) * dyn
    dp_minor = ((1 + K_MINOR) + (K_MINOR - 1)) * dyn    # plenum losses, Eqs. 19-22

    # friction with viscosity following the linear bulk temperature rise
    xs = np.linspace(0, geo.length, 201)
    mu_x = np.array([water_at(t_in + (t_out - t_in) * x / geo.length).mu for x in xs])
    dpdx = 2 * f_re_shah_london(a) * mu_x * u_m / geo.d_h**2      # = 4 f / D_h * rho u^2 / 2
    dp_ch_local = np.trapezoid(dpdx, xs) + hagenbach_k(a) * dyn

    # heat transfer: 3-wall fully developed h, adiabatic-tip fins (tip at the cover)
    h = nu * w_m.k / geo.d_h
    m = np.sqrt(2 * h / (k_s * 2 * geo.w_w))
    eta_f = np.tanh(m * geo.h_ch) / (m * geo.h_ch)
    a_base = geo.bottom_area_total                      # N * W * L
    r_cond = geo.h_w2 / (k_s * a_base)
    r_conv = 1 / (h * geo.n_channels * (geo.w_ch + 2 * eta_f * geo.h_ch) * geo.length)
    r_cal = 1 / (m_dot * w_in.cp)

    # thermocouples: local bulk T + conduction from TC plane to channel + convection
    x_tc = np.array(geo.x_tc)
    t_f = t_in + q * (x_tc / geo.length) * r_cal
    r_tc = (geo.h_w2 - geo.h_th) / (k_s * a_base)
    t_tc = t_f + q * (r_tc + r_conv)
    t_max = t_in + q * (r_cond + r_conv + r_cal)

    # wall viscosity correction, laminar liquid heating: f/f_iso = (mu_w/mu_b)^0.58
    # (Deissler, in Kakac, Shah & Aung, Handbook of Single-Phase Convective Heat Transfer)
    t_wall = 0.5 * (t_in + t_out) + q * r_conv
    visc_ratio = water_at(t_wall).mu / w_m.mu
    dp_ch_wall = (4 * f * visc_ratio**0.58 * geo.length / geo.d_h + hagenbach_k(a)) * dyn

    return Result(flow_ml_min, re_in, u_m, t_out, dp_ch, dp_minor, dp_ch_local, dp_ch_wall, h, eta_f,
                  r_cond, r_conv, r_cal, t_tc, t_max,
                  l_h=0.05 * re_m * geo.d_h, l_th=0.05 * re_m * w_m.pr * geo.d_h)


@dataclass
class DesignResult:
    re: float
    dp_ch: float           # Pa, channel only (friction + Hagenbach), mean-T viscosity
    t_out: float
    t_base_max: float      # C, heater surface at the outlet end
    r_th: float            # K/W, (T_base,max - T_in) / Q
    pumping_power: float   # W
    eta_f: float
    l_th: float


def run_design(w_ch: float, w_fin: float, h_ch: float, h_base: float, flow_ml_min: float,
               q_eff: float, t_in: float = 30.0, k_s: float = K_COPPER, length: float = 44.764e-3,
               plate_w: float = 1.0e-2, footprint_l: float = 44.8e-3) -> DesignResult:
    """1-D model of a plate with a metal lid; 4-wall Nu, lid at fin-tip T (theta = 1/cosh(mH))."""
    w_cell = w_ch + w_fin
    n = plate_w / w_cell
    vdot = flow_ml_min * ML_MIN_TO_M3_S
    q = q_eff * plate_w * footprint_l
    a = min(w_ch, h_ch) / max(w_ch, h_ch)
    d_h = 2 * w_ch * h_ch / (w_ch + h_ch)
    w_in = water_at(t_in)
    m_dot = w_in.rho * vdot
    t_out = t_in + q / (m_dot * w_in.cp)
    w_m = water_at(0.5 * (t_in + t_out))
    u_m = m_dot / (w_m.rho * n * w_ch * h_ch)
    re_in = w_in.rho * (vdot / (n * w_ch * h_ch)) * d_h / w_in.mu
    re_m = w_m.rho * u_m * d_h / w_m.mu
    dyn = 0.5 * w_m.rho * u_m**2
    dp = (4 * f_re_shah_london(a) / re_m * length / d_h + hagenbach_k(a)) * dyn
    nu4 = 8.235 * (1 - 2.0421 * a + 3.0853 * a**2 - 2.4765 * a**3 + 1.0578 * a**4 - 0.1861 * a**5)
    h = nu4 * w_m.k / d_h
    m = np.sqrt(2 * h / (k_s * w_fin))
    mh = m * h_ch
    eta_f = np.tanh(mh) / mh
    area_eff = n * length * (w_ch + 2 * eta_f * h_ch + w_ch / np.cosh(mh))
    r_conv = 1 / (h * area_eff)
    r_cond = h_base / (k_s * plate_w * footprint_l)
    t_base_max = t_out + q * (r_cond + r_conv)          # outlet end, bulk T = T_out
    return DesignResult(re_in, dp, t_out, t_base_max, (t_base_max - t_in) / q, vdot * dp, eta_f,
                        0.05 * re_m * w_m.pr * d_h)


if __name__ == "__main__":
    geo = Geometry()
    for q_eff, flow in ((100e4, 93.5), (100e4, 1136.4), (200e4, 259.1), (200e4, 859.5)):
        r = run(flow, q_eff, geo)
        print(f"q={q_eff / 1e4:.0f} W/cm2, {flow:7.1f} ml/min: Re={r.re:5.0f}  "
              f"dP={r.dp / 1e5:.3f} bar (ch {r.dp_ch / 1e5:.3f}, minor {r.dp_minor / 1e5:.3f}, "
              f"ch-local-mu {r.dp_ch_local / 1e5:.3f})  dT={r.t_out - T_IN_C:5.1f} K  "
              f"h={r.h:7.0f}  eta_f={r.eta_f:.3f}  R_cond/R_conv/R_cal = "
              f"{r.r_cond * 1e3:.1f}/{r.r_conv * 1e3:.1f}/{r.r_cal * 1e3:.1f} mK/W  "
              f"T_tc-T_in = {np.round(r.t_tc - T_IN_C, 1)}  L_h={r.l_h * 1e3:.1f} mm  L_th={r.l_th * 1e3:.0f} mm")
