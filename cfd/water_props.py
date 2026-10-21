"""Polynomial fits of liquid-water properties (CoolProp) for Fluent: phi(T) = A1 + A2*T + ..., T in K.
Fit range 288-403 K.  python water_props.py prints the coefficients and fit errors.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytical"))
from inputs import water_at  # noqa: E402

T_FIT = np.linspace(288.15, 403.15, 47)


def fit(prop: str, deg: int) -> tuple[list[float], float]:
    """Return ascending coefficients and the max relative fit error."""
    vals = np.array([getattr(water_at(t - 273.15), prop) for t in T_FIT])
    # fit in scaled tau = (T - t0)/s for conditioning, then expand in powers of T
    t0, s = 345.65, 57.5
    p = np.polynomial.Polynomial.fit((T_FIT - t0) / s, vals, deg, domain=[-1, 1])
    plain = p.convert(domain=[-1, 1], window=[-1, 1])
    tau_of_T = np.polynomial.Polynomial([-t0 / s, 1 / s])
    poly_T = plain.coef[0] + sum(c * tau_of_T**k for k, c in enumerate(plain.coef) if k > 0)
    coef = list(np.asarray(poly_T.coef, dtype=float))
    err = np.max(np.abs(np.polynomial.Polynomial(coef)(T_FIT) / vals - 1))
    return coef, err


VISCOSITY, ERR_MU = fit("mu", 7)
CONDUCTIVITY, ERR_K = fit("k", 3)
SPECIFIC_HEAT, ERR_CP = fit("cp", 3)
RHO_IN = water_at(15.0).rho     # constant density at the 15 C inlet

if __name__ == "__main__":
    for name, c, e in (("mu", VISCOSITY, ERR_MU), ("k", CONDUCTIVITY, ERR_K), ("cp", SPECIFIC_HEAT, ERR_CP)):
        print(f"{name}: max fit error {100 * e:.3f} %   coefficients {['%.6e' % v for v in c]}")
    print(f"rho = {RHO_IN:.2f} kg/m3")
