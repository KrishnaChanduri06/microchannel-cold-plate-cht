"""Fully developed laminar f*Re and Nu (H1) in a rectangular duct, finite volume.

Checked against Shah & London (4 walls heated), then run with the top wall adiabatic.
Run:  python nusselt_fd.py
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve(width: float, height: float, heated: dict, n: int = 240) -> tuple[float, float]:
    """Return (f*Re, Nu_H1); heated = {"bottom", "top", "left", "right": bool}."""
    ny = n
    nz = max(8, int(round(n * height / width)))
    dy, dz = width / ny, height / nz
    N = ny * nz
    idx = lambda j, k: k * ny + j

    def laplacian(dirichlet: dict):
        """5-point Laplacian, wall either Dirichlet (0) or Neumann (no flux)."""
        rows, cols, vals = [], [], []
        diag = np.zeros(N)
        for k in range(nz):
            for j in range(ny):
                p = idx(j, k)
                for nb, d2, wall in ((j - 1, dy, "left"), (j + 1, dy, "right")):
                    if 0 <= nb < ny:
                        rows.append(p); cols.append(idx(nb, k)); vals.append(1 / d2**2)
                        diag[p] -= 1 / d2**2
                    elif dirichlet[wall]:
                        diag[p] -= 2 / d2**2
                for nb, d2, wall in ((k - 1, dz, "bottom"), (k + 1, dz, "top")):
                    if 0 <= nb < nz:
                        rows.append(p); cols.append(idx(j, nb)); vals.append(1 / d2**2)
                        diag[p] -= 1 / d2**2
                    elif dirichlet[wall]:
                        diag[p] -= 2 / d2**2
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N)) + sp.diags(diag)
        return A.tocsc()

    area = width * height
    d_h = 4 * area / (2 * (width + height))

    # momentum: lap(u) = -1, u = 0 on all walls
    u = spla.spsolve(laplacian({w: True for w in heated}), -np.ones(N))
    u_m = u.mean()
    f_re = d_h**2 / (2 * u_m)                      # Fanning, dp/dx = -mu

    # energy: lap(theta) = u / (A u_m), theta = T - T_w, unit q' and k
    theta = spla.spsolve(laplacian(heated), u / (area * u_m))
    theta_b = (u * theta).sum() / u.sum()          # T_b - T_w, magnitude used
    p_heated = sum(L for w, L in (("bottom", width), ("top", width), ("left", height), ("right", height)) if heated[w])
    h = 1.0 / (p_heated * abs(theta_b))           # heated-perimeter basis
    return f_re, h * d_h


ALL4 = {"bottom": True, "top": True, "left": True, "right": True}
TOP_ADIABATIC = {"bottom": True, "top": False, "left": True, "right": True}


def shah_london_4wall(a: float) -> tuple[float, float]:
    f_re = 24 * (1 - 1.3553 * a + 1.9467 * a**2 - 1.7012 * a**3 + 0.9564 * a**4 - 0.2537 * a**5)
    nu = 8.235 * (1 - 2.0421 * a + 3.0853 * a**2 - 2.4765 * a**3 + 1.0578 * a**4 - 0.1861 * a**5)
    return f_re, nu


def main() -> None:
    print("Check vs Shah & London (4 walls heated):")
    print("  alpha   fRe(num)  fRe(S&L)   Nu(num)  Nu(S&L)")
    for a in (1.0, 0.5, 0.25, 0.324):
        fr, nu = solve(1.0, a, ALL4) if a < 1 else solve(1.0, 1.0, ALL4)
        fr_sl, nu_sl = shah_london_4wall(a)
        print(f"  {a:5.3f}   {fr:7.3f}   {fr_sl:7.3f}   {nu:7.3f}  {nu_sl:7.3f}")

    # channel 231 um wide, 713 um deep; top (cover) is a short side
    w, h = 231.0, 713.0
    print("\nQu & Mudawar channel, 231 x 713 um (alpha = 0.324):")
    for n in (120, 240):
        fr4, nu4 = solve(w, h, ALL4, n=n // 3)
        fr3, nu3 = solve(w, h, TOP_ADIABATIC, n=n // 3)
        print(f"  grid {n // 3}x{round(n // 3 * h / w)}: fRe = {fr4:.3f}, Nu_4wall = {nu4:.3f}, "
              f"Nu_3wall(top adiabatic) = {nu3:.3f}")


if __name__ == "__main__":
    main()
