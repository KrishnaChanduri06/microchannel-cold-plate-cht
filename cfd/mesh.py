"""Structured hex mesh of the Qu & Mudawar (2002) unit cell (Fig. 3), written as a Fluent .msh file.
x along the channel, y across, z up. Also writes <mesh>.json with the geometry for case.py.

Usage:  python mesh.py [refinement factor, default 1.0] [output .msh]
        python mesh.py 1.0 out.msh --w-ch 150 --w-fin 100 --h-ch 713 --no-slots   (design study, um)
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

UM = 1e-6
W_W, W_CH = 118 * UM, 231 * UM          # half fin, channel width (Table 1)
H_W2, H_CH, H_W1 = 5637 * UM, 713 * UM, 12700 * UM
H_TH = 3175 * UM                        # thermocouple plane
L = 44.764e-3
X_TC = (5.0e-3, 16.588e-3, 28.176e-3, 39.764e-3)
SLOT_W = 1.6e-3                         # Sec. 2.2: "three 1.6 mm slots"
SLOT_H = H_TH                           # from Fig. 3: slot tops level with the thermocouples (assumption)
SLOT_C = tuple(0.5 * (a + b) for a, b in zip(X_TC[:-1], X_TC[1:]))   # midway between TCs
W = 2 * W_W + W_CH
H = H_W2 + H_CH + H_W1

WATER, COPPER, COVER, AIR = 1, 2, 3, 4
NAMES = {WATER: "water", COPPER: "copper", COVER: "cover", AIR: "air"}


@dataclass
class Geo:
    """Unit-cell geometry; defaults = Qu & Mudawar test section (Table 1)."""
    w_w: float = W_W            # HALF fin width
    w_ch: float = W_CH
    h_ch: float = H_CH
    h_w2: float = H_W2
    h_w1: float = H_W1          # lid thickness
    slots: bool = True          # test-rig air slots (off for the design study)

    @property
    def w(self) -> float:
        return 2 * self.w_w + self.w_ch

    @property
    def h(self) -> float:
        return self.h_w2 + self.h_ch + self.h_w1

    @property
    def a_bot(self) -> float:
        """Copper area of the unit-cell bottom (heated)."""
        return self.w * (L - (3 * SLOT_W if self.slots else 0.0))


def graded(a: float, b: float, n: int, ratio: float = 1.0, both: bool = False) -> np.ndarray:
    """n cells on [a, b]. ratio = last/first cell size (one-sided), or edge/centre if both."""
    n = max(1, int(n))
    if both:
        n1 = n // 2
        left = graded(0, 0.5, n1, 1 / ratio) if n1 else np.array([0.0, 0.5])
        right = graded(0.5, 1.0, n - n1, ratio)
        s = np.concatenate([left, right[1:]])
    elif ratio == 1.0 or n == 1:
        s = np.linspace(0, 1, n + 1)
    else:
        g = ratio ** (1 / (n - 1))
        w = g ** np.arange(n)
        s = np.concatenate([[0], np.cumsum(w) / w.sum()])
    return a + (b - a) * s


def join(*segs) -> np.ndarray:
    out = segs[0]
    for s in segs[1:]:
        out = np.concatenate([out, s[1:]])
    return out


def grid(r: float = 1.0, g: Geo = Geo()):
    """Grid lines; r scales every segment count (r = 1 -> ~100 x 22 x 40 cells)."""
    c = lambda n: max(2, int(round(n * r)))
    W_W, W_CH, H_W2, H_CH, W = g.w_w, g.w_ch, g.h_w2, g.h_ch, g.w
    H = g.h
    # x: fine near the inlet (developing flow), lines at every TC and slot edge
    xs = [0.0]
    bounds = [X_TC[0]]
    for sc, xt in zip(SLOT_C, X_TC[1:]):
        bounds += [sc - SLOT_W / 2, sc + SLOT_W / 2, xt]
    bounds += [L]
    segs = [graded(0.0, X_TC[0], c(20), ratio=12.0)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        n = c(4) if abs(b - a - SLOT_W) < 1e-9 else c(10)
        segs.append(graded(a, b, n))
    x = join(*segs)
    # y: half fin | channel (clustered at both side walls) | half fin
    y = join(graded(0, W_W, c(4)), graded(W_W, W_W + W_CH, c(14), ratio=3.0, both=True),
             graded(W_W + W_CH, W, c(4)))
    # z: base (line at slot top = TC plane) | channel (clustered at bottom/top walls) | cover
    base = (join(graded(0, SLOT_H, c(7)), graded(SLOT_H, H_W2, c(7))) if H_W2 > SLOT_H + 1e-9
            else graded(0, H_W2, c(8)))
    z = join(base, graded(H_W2, H_W2 + H_CH, c(16), ratio=3.0, both=True),
             graded(H_W2 + H_CH, H, c(10), ratio=8.0))
    return x, y, z


def materials(x, y, z, g: Geo = Geo()) -> np.ndarray:
    W_W, W_CH, H_W2, H_CH = g.w_w, g.w_ch, g.h_w2, g.h_ch
    xc, yc, zc = (0.5 * (v[1:] + v[:-1]) for v in (x, y, z))
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    m = np.full(X.shape, COPPER, dtype=np.int8)
    if g.slots:
        in_slot = np.zeros(X.shape, bool)
        for sc in SLOT_C:
            in_slot |= (np.abs(X - sc) < SLOT_W / 2)
        m[(Z < SLOT_H) & in_slot] = AIR
    m[(Z > H_W2) & (Z < H_W2 + H_CH) & (Y > W_W) & (Y < W_W + W_CH)] = WATER
    m[Z > H_W2 + H_CH] = COVER
    return m


def write_msh(path: Path, x, y, z, mat) -> dict:
    nx, ny, nz = mat.shape
    NX, NY = nx + 1, ny + 1
    node = lambda i, j, k: i + NX * (j + NY * k) + 1                # 1-based

    # cells numbered contiguously per material zone
    flat = mat.ravel(order="F")                                     # index = i + nx*(j + ny*k)
    order = np.argsort(flat, kind="stable")
    cell_id = np.empty(flat.size, np.int64)
    cell_id[order] = np.arange(1, flat.size + 1)
    cid = lambda i, j, k: cell_id[i + nx * (j + ny * k)]
    cell_ranges, start = {}, 1
    for m_ in sorted(NAMES):
        n = int((flat == m_).sum())
        if n:
            cell_ranges[m_] = (start, start + n - 1)
            start += n

    # faces: (n1..n4 ordered so the right-hand normal points along +axis), minus cell, plus cell
    faces = {}   # zone name -> list of arrays [n1 n2 n3 n4 c0 c1]

    def add(zone, n4, cm, cp):
        """cm/cp = cell on the -/+ side (0 = none)."""
        faces.setdefault(zone, []).append((n4, cm, cp))

    def zone_for(mm, mp, axis, side):
        if mm and mp:
            if mm == mp:
                return f"interior-{NAMES[mm]}", "interior"
            a, b = sorted((NAMES[mm], NAMES[mp]))
            return f"{a}-{b}", "wall"
        m_ = mm or mp
        if axis == 0:
            if m_ == WATER:
                return ("inlet", "velocity-inlet") if side == 0 else ("outlet", "pressure-outlet")
            return (f"end-{'in' if side == 0 else 'out'}-{NAMES[m_]}", "wall")
        if axis == 1:
            return f"sym-{NAMES[m_]}", "symmetry"
        if side == 0:
            return ("heater", "wall") if m_ == COPPER else ("slot-bottom", "wall")
        return "cover-top", "wall"

    I, J, K = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    big = {}
    for axis, dims in ((0, (nx + 1, ny, nz)), (1, (nx, ny + 1, nz)), (2, (nx, ny, nz + 1))):
        i, j, k = (a.ravel() for a in np.meshgrid(*(np.arange(d) for d in dims), indexing="ij"))
        # node quads: right-hand normal along +axis
        if axis == 0:
            q = np.stack([node(i, j, k), node(i, j + 1, k), node(i, j + 1, k + 1), node(i, j, k + 1)], 1)
        elif axis == 1:
            q = np.stack([node(i, j, k), node(i, j, k + 1), node(i + 1, j, k + 1), node(i + 1, j, k)], 1)
        else:
            q = np.stack([node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k)], 1)
        idx = (i, j, k)
        lo = [a.copy() for a in idx]; lo[axis] = lo[axis] - 1
        n_ax = (nx, ny, nz)[axis]
        has_m = idx[axis] > 0
        has_p = idx[axis] < n_ax
        mm = np.zeros(i.size, np.int64); mp = np.zeros(i.size, np.int64)
        cm = np.zeros(i.size, np.int64); cp = np.zeros(i.size, np.int64)
        ii, jj, kk = (a[has_m] for a in lo)
        mm[has_m] = mat[ii, jj, kk]; cm[has_m] = cell_id[ii + nx * (jj + ny * kk)]
        ii, jj, kk = (a[has_p] for a in idx)
        mp[has_p] = mat[ii, jj, kk]; cp[has_p] = cell_id[ii + nx * (jj + ny * kk)]
        side = np.where(has_m, 1, 0)
        keys = {}
        for t in set(zip(mm.tolist(), mp.tolist(), side.tolist())):
            keys[t] = zone_for(t[0], t[1], axis, t[2])
        labels = np.array([keys[t] [0] for t in zip(mm.tolist(), mp.tolist(), side.tolist())])
        types = {v[0]: v[1] for v in keys.values()}
        for name in np.unique(labels):
            sel = labels == name
            big.setdefault(name, [types[name], []])[1].append((q[sel], cm[sel], cp[sel]))

    bc = {"interior": 2, "wall": 3, "pressure-outlet": 5, "symmetry": 7, "velocity-inlet": 10}
    with open(path, "w") as f:
        f.write('(0 "Qu & Mudawar 2002 unit cell, structured hex, cfd/mesh.py")\n(2 3)\n')
        X, Y, Z = (np.meshgrid(x, y, z, indexing="ij")[d].ravel(order="F") for d in range(3))
        nn = X.size
        f.write(f"(10 (0 1 {nn:x} 0 3))\n(10 (1 1 {nn:x} 1 3)(\n")
        np.savetxt(f, np.column_stack([X, Y, Z]), fmt="%.10e")
        f.write("))\n")
        ncell = flat.size
        f.write(f"(12 (0 1 {ncell:x} 0))\n")
        zone_ids = {}
        zid = 2
        for m_, (a, b) in cell_ranges.items():
            f.write(f"(12 ({zid:x} {a:x} {b:x} 1 4))\n")
            zone_ids[NAMES[m_]] = (zid, "fluid" if m_ == WATER else "solid")
            zid += 1
        total = sum(sum(len(p[1]) for p in parts) for _, parts in big.values())
        f.write(f"(13 (0 1 {total:x} 0))\n")
        fstart = 1
        zid = 10
        face_info = {}
        for name in sorted(big, key=lambda s: (not s.startswith("interior"), s)):
            ztype, parts = big[name]
            q = np.concatenate([p[0] for p in parts]); cm = np.concatenate([p[1] for p in parts])
            cp = np.concatenate([p[2] for p in parts])
            # Fluent: right-hand node order, normal points toward c0. Quads point along +axis, so
            # c0 = plus cell, c1 = minus cell; where only the minus cell exists, reverse the nodes.
            c0 = cp.copy(); c1 = cm.copy()
            flip = c0 == 0
            q[flip] = q[flip][:, ::-1]
            c0[flip] = cm[flip]; c1[flip] = 0
            n = len(q)
            f.write(f"(13 ({zid:x} {fstart:x} {fstart + n - 1:x} {bc[ztype]:x} 4)(\n")
            np.savetxt(f, np.column_stack([q, c0, c1]), fmt="%x")
            f.write("))\n")
            face_info[name] = (zid, ztype, n)
            fstart += n
            zid += 1
        for name, (i_, t) in zone_ids.items():
            f.write(f"(45 ({i_} {t} {name})())\n")
        for name, (i_, t, _) in face_info.items():
            f.write(f"(45 ({i_} {t} {name})())\n")
    return dict(cells=ncell, nodes=nn, faces=total, cell_zones={k: v for k, v in zip(
        [NAMES[m] for m in cell_ranges], [b - a + 1 for a, b in cell_ranges.values()])},
        face_zones={k: v[2] for k, v in face_info.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("r", nargs="?", type=float, default=1.0)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--w-ch", type=float, help="channel width, um")
    ap.add_argument("--w-fin", type=float, help="FULL fin width, um")
    ap.add_argument("--h-ch", type=float, help="channel depth, um")
    ap.add_argument("--h-base", type=float, help="copper below the channel, um")
    ap.add_argument("--h-lid", type=float, help="lid thickness, um")
    ap.add_argument("--no-slots", action="store_true")
    a = ap.parse_args()
    r = a.r
    g = Geo(w_w=(a.w_fin / 2 * UM) if a.w_fin else W_W, w_ch=(a.w_ch * UM) if a.w_ch else W_CH,
            h_ch=(a.h_ch * UM) if a.h_ch else H_CH, h_w2=(a.h_base * UM) if a.h_base else H_W2,
            h_w1=(a.h_lid * UM) if a.h_lid else H_W1, slots=not a.no_slots)
    if g.slots and g.h_w2 < SLOT_H:
        sys.exit("slots need a base thicker than the slot height; use --no-slots")
    out = Path(a.out) if a.out else Path(__file__).parent / "runs" / f"unitcell_r{r:g}.msh"
    out.parent.mkdir(parents=True, exist_ok=True)
    x, y, z = grid(r, g)
    mat = materials(x, y, z, g)
    info = write_msh(out, x, y, z, mat)
    out.with_suffix(".json").write_text(json.dumps(dict(asdict(g), w=g.w, a_bot=g.a_bot, cells=info["cells"],
                                                        L=L, refinement=r), indent=1))
    print(f"{out.name}: {len(x) - 1} x {len(y) - 1} x {len(z) - 1} = {info['cells']} cells")
    print("  cell zones:", info["cell_zones"])
    print("  face zones:", info["face_zones"])
    a_bot = int((mat[:, :, 0] == COPPER).sum())
    print(f"  min dx = {np.diff(x).min() * 1e6:.1f} um, max dx = {np.diff(x).max() * 1e6:.0f} um; "
          f"min dy = {np.diff(y).min() * 1e6:.1f} um; min dz = {np.diff(z).min() * 1e6:.1f} um")


if __name__ == "__main__":
    main()
