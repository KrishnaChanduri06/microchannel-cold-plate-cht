"""Benchmark inputs from Qu & Mudawar (2002), IJHMT 45:2549-2565.

Geometry from Table 1 (unit cell), operating conditions from Sec. 2.3.
"""
from dataclasses import dataclass

from CoolProp.CoolProp import PropsSI

UM = 1e-6  # micrometre -> metre


@dataclass(frozen=True)
class Geometry:
    """Unit cell: one channel plus half a copper fin on each side."""
    n_channels: int = 21
    w_ch: float = 231 * UM        # channel width
    h_ch: float = 713 * UM        # channel height
    w_w: float = 118 * UM         # half fin width (full fin 236 um)
    h_w2: float = 5637 * UM       # copper thickness below the channel
    h_w1: float = 12700 * UM      # polycarbonate cover thickness (Table 1)
    h_th: float = 3175 * UM       # thermocouple plane above unit-cell bottom (Table 1)
    length: float = 44.764e-3     # channel length L
    # thermocouple x positions: L1 = 5 mm, then 11.588 mm pitch (Table 1)
    x_tc: tuple = (5.0e-3, 16.588e-3, 28.176e-3, 39.764e-3)
    planform_area: float = 1.0e-2 * 4.48e-2   # A_t, Eq. (1): 1.0 cm x 4.48 cm

    @property
    def w_cell(self) -> float:
        return self.w_ch + 2 * self.w_w

    @property
    def area_ch(self) -> float:
        """Flow area of one channel."""
        return self.w_ch * self.h_ch

    @property
    def d_h(self) -> float:
        """Hydraulic diameter, 4A/P."""
        return 4 * self.area_ch / (2 * (self.w_ch + self.h_ch))

    @property
    def aspect(self) -> float:
        """Aspect ratio alpha, short side / long side."""
        return min(self.w_ch, self.h_ch) / max(self.w_ch, self.h_ch)

    @property
    def bottom_area_total(self) -> float:
        """N * A_bot for the bottom heat flux, Eq. (14)."""
        return self.n_channels * self.w_cell * self.length


@dataclass(frozen=True)
class Water:
    rho: float   # density, kg/m^3
    mu: float    # dynamic viscosity, Pa s
    cp: float    # specific heat, J/kg K
    k: float     # thermal conductivity, W/m K

    @property
    def pr(self) -> float:
        return self.mu * self.cp / self.k


def water_at(t_celsius: float, p_pa: float = 101325.0) -> Water:
    """Liquid water properties (CoolProp); p is kept above saturation to stay liquid."""
    t = t_celsius + 273.15
    p_pa = max(p_pa, 1.05 * PropsSI("P", "T", t, "Q", 0, "Water"))
    return Water(
        rho=PropsSI("D", "T", t, "P", p_pa, "Water"),
        mu=PropsSI("V", "T", t, "P", p_pa, "Water"),
        cp=PropsSI("C", "T", t, "P", p_pa, "Water"),
        k=PropsSI("L", "T", t, "P", p_pa, "Water"),
    )


K_COPPER = 401.0      # W/m K, oxygen-free copper
K_COVER = 0.2         # W/m K, polycarbonate (Lexan) cover
H_TOP = 10.0          # W/m^2 K, natural convection on the cover top (Sec. 3.1, Eq. 15)
T_AMB_C = 25.0        # deg C, ambient for the cover top
K_MINOR = 1.0         # plenum contraction/expansion loss coefficient (Sec. 3.2)
T_IN_C = 15.0         # inlet temperature, deg C (all tests)
Q_EFF_LEVELS = (100e4, 200e4)   # W/m^2 (100 and 200 W/cm^2)

# flow-rate range (ml/min) and paper Re range
FLOW_RANGES = {
    100e4: {"ml_min": (93.5, 1136.4), "re_paper": (139, 1672)},
    200e4: {"ml_min": (259.1, 859.5), "re_paper": (385, 1289)},
}
