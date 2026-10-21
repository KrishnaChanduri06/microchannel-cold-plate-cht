"""Marker digitiser for the Qu & Mudawar (2002) figures (Figs 4a, 5, 6, 7).

Writes one raw CSV per panel and an overlay PNG in data/digitised/check/.
Run:  .venv/Scripts/python digitise/digitise.py
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "data/paper/figs"
OUT = ROOT / "data/digitised"
CHECK = OUT / "check"


@dataclass
class Panel:
    name: str              # output file stem
    image: str             # extracted figure file
    index: int             # 0 = left panel (a), 1 = right panel (b); frames sorted by x
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    series: dict[str, str]  # shape -> series label
    open_kernel: int = 9   # px, opening size (removes thinner features)
    exclude: list[tuple[float, float, float, float]] = field(default_factory=list)  # data-coord boxes to ignore


def ink(path: Path) -> np.ndarray:
    """Binary image, 1 = ink (polarity fixed for inverted 1-bit masks)."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    b = (g < 128).astype(np.uint8)
    return 1 - b if b.mean() > 0.5 else b


def frames(b: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Axes frames as (x0, y0, x1, y1) at line centres, sorted left to right."""
    h_lines = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((1, 400), np.uint8))
    v_lines = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((300, 1), np.uint8))
    lines = cv2.bitwise_or(h_lines, v_lines)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w > 0.3 * b.shape[1] / 2 and h > 0.5 * b.shape[0]:
            sub = lines[y:y + h, x:x + w]
            rows = np.flatnonzero(sub.sum(1) > 0.8 * w)
            cols = np.flatnonzero(sub.sum(0) > 0.8 * h)
            # line centre = mean of the outer rows/cols
            top = rows[rows < h / 2].mean() + y
            bot = rows[rows > h / 2].mean() + y
            left = cols[cols < w / 2].mean() + x
            right = cols[cols > w / 2].mean() + x
            out.append((left, top, right, bot))
    return sorted(out)


def major_ticks(b: np.ndarray, fr, axis: str) -> list[float]:
    """Pixel centres of the major ticks on the bottom or left edge."""
    x0, y0, x1, y1 = (int(round(v)) for v in fr)
    if axis == "x":
        band = b[y1 - 30:y1 - 4, x0 + 6:x1 - 5]       # above the bottom line
        length = band[::-1].cumprod(axis=0).sum(0)    # contiguous ink upward
        pos = np.arange(x0 + 6, x1 - 5)
    else:
        band = b[y0 + 6:y1 - 5, x0 + 4:x0 + 30]       # right of the left line
        length = band.cumprod(axis=1).sum(1)
        pos = np.arange(y0 + 6, y1 - 5)
    if length.max() == 0:
        return []
    is_major = length >= 0.75 * length.max()
    idx = np.flatnonzero(is_major)
    if idx.size == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
    return [float(pos[g].mean()) for g in groups]


def legend_box(b: np.ndarray, fr) -> tuple[int, int, int, int] | None:
    """Legend box (largest rectangle inside the frame), padded by 6 px."""
    x0, y0, x1, y1 = (int(round(v)) for v in fr)
    inner = b[y0 + 35:y1 - 35, x0 + 35:x1 - 35]
    h_l = cv2.morphologyEx(inner, cv2.MORPH_OPEN, np.ones((1, 150), np.uint8))
    v_l = cv2.morphologyEx(inner, cv2.MORPH_OPEN, np.ones((80, 1), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_or(h_l, v_l))
    best = None
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w > 150 and h > 80 and (best is None or w * h > best[2] * best[3]):
            best = (x, y, w, h)
    if best is None:
        return None
    x, y, w, h = best
    return x + x0 + 35 - 6, y + y0 + 35 - 6, x + w + x0 + 35 + 6, y + h + y0 + 35 + 6


def classify(mask: np.ndarray) -> str:
    """Marker shape from fill ratio and widest-row position."""
    h, w = mask.shape
    fill = mask.sum() / (h * w)
    if fill > 0.8:
        return "square"
    widths = mask.sum(1)
    peak = np.flatnonzero(widths >= 0.9 * widths.max()).mean() / (h - 1)
    if peak > 0.7:
        return "up"
    if peak < 0.3:
        return "down"
    return "diamond"


def split_merged(m: np.ndarray, x: int, y: int, size: float) -> list:
    """Split an oversized blob at its narrowest neck."""
    h, w = m.shape
    if size == 0 or (w < 1.4 * size and h < 1.4 * size):
        return [(x, y, m)]
    if w >= h:
        prof = m.sum(0).astype(float)
        lo, hi = int(0.25 * w), int(0.75 * w)
        cut = lo + int(np.argmin(prof[lo:hi]))
        parts = [(x, y, m[:, :cut]), (x + cut, y, m[:, cut:])]
    else:
        prof = m.sum(1).astype(float)
        lo, hi = int(0.25 * h), int(0.75 * h)
        cut = lo + int(np.argmin(prof[lo:hi]))
        parts = [(x, y, m[:cut]), (x, y + cut, m[cut:])]
    out = []
    for px, py, pm in parts:
        rows, cols = np.flatnonzero(pm.any(1)), np.flatnonzero(pm.any(0))
        if rows.size == 0:
            continue
        pm = pm[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        out.extend(split_merged(pm, px + cols[0], py + rows[0], size))
    return out


def join_split(pieces: list, size: float) -> list:
    """Re-join marker pieces split by a curve (combined bbox within one marker size)."""
    pieces = list(pieces)
    merged = True
    while merged:
        merged = False
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                xi, yi, mi = pieces[i]
                xj, yj, mj = pieces[j]
                x0, y0 = min(xi, xj), min(yi, yj)
                x1 = max(xi + mi.shape[1], xj + mj.shape[1])
                y1 = max(yi + mi.shape[0], yj + mj.shape[0])
                small = mi.sum() + mj.sum() < 1.3 * size * size
                if x1 - x0 <= 1.15 * size and y1 - y0 <= 1.15 * size and small:
                    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
                    m[yi - y0:yi - y0 + mi.shape[0], xi - x0:xi - x0 + mi.shape[1]] |= mi
                    m[yj - y0:yj - y0 + mj.shape[0], xj - x0:xj - x0 + mj.shape[1]] |= mj
                    pieces[i] = (x0, y0, m)
                    del pieces[j]
                    merged = True
                    break
            if merged:
                break
    return pieces


def digitise(p: Panel) -> pd.DataFrame:
    b = ink(FIGS / p.image)
    fr = frames(b)[p.index]
    x0, y0, x1, y1 = fr
    to_x = lambda px: p.xlim[0] + (px - x0) / (x1 - x0) * (p.xlim[1] - p.xlim[0])
    to_y = lambda py: p.ylim[0] + (y1 - py) / (y1 - y0) * (p.ylim[1] - p.ylim[0])

    # calibration check, major ticks should fall on round values
    tx = [round(to_x(t), 4) for t in major_ticks(b, fr, "x")]
    ty = [round(to_y(t), 4) for t in major_ticks(b, fr, "y")]

    work = b.copy()
    ix0, iy0, ix1, iy1 = int(x0) + 4, int(y0) + 4, int(x1) - 4, int(y1) - 4
    region = np.zeros_like(work)
    region[iy0:iy1, ix0:ix1] = 1
    work &= region
    # fill hollow markers before the opening
    cnts, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(work)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    k = p.open_kernel
    solid = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    lg = legend_box(b, fr)
    if lg:
        solid[lg[1]:lg[3], lg[0]:lg[2]] = 0
    for ex in p.exclude:
        ex_px = (
            int(x0 + (ex[0] - p.xlim[0]) / (p.xlim[1] - p.xlim[0]) * (x1 - x0)),
            int(y1 - (ex[3] - p.ylim[0]) / (p.ylim[1] - p.ylim[0]) * (y1 - y0)),
            int(x0 + (ex[2] - p.xlim[0]) / (p.xlim[1] - p.xlim[0]) * (x1 - x0)),
            int(y1 - (ex[1] - p.ylim[0]) / (p.ylim[1] - p.ylim[0]) * (y1 - y0)),
        )
        solid[ex_px[1]:ex_px[3], ex_px[0]:ex_px[2]] = 0

    # opening erodes triangle tips: dilate by k and AND with filled ink to restore outline
    solid = cv2.dilate(solid, np.ones((k, k), np.uint8)) & filled

    n, lab, stats, cent = cv2.connectedComponentsWithStats(solid, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    # typical marker area, ignoring crumbs < 150 px
    big_areas = areas[areas > 150]
    typical = np.median(big_areas) if len(big_areas) else 0
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.35 * typical]
    size = np.median([max(stats[i, 2], stats[i, 3]) for i in keep if stats[i, 4] > 150]) if keep else 0

    # (x, y, mask) per marker
    pieces = []
    for i in keep:
        x, y, w, h, a = stats[i]
        m = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        pieces.extend(split_merged(m, x, y, size))
    pieces = join_split(pieces, size)
    pieces = [pc for pc in pieces if pc[2].sum() >= 0.35 * typical]   # drop split crumbs

    rows = []
    for x, y, m in pieces:
        h, w = m.shape
        a = int(m.sum())
        shape = classify(m)
        allowed = {s.removesuffix("_open") for s in p.series}
        if shape not in allowed:
            # fallback on fill ratio (square ~1, triangle ~0.5)
            fill = m.sum() / m.size
            tri = [s for s in ("up", "down") if s in allowed]
            shape = "square" if ("square" in allowed and (fill > 0.68 or not tri)) else tri[0]
        region_ink = b[y:y + h, x:x + w][m > 0]
        hollow = region_ink.mean() < 0.7
        # bbox centre (centroid is biased for triangles)
        cx, cy = x + w / 2, y + h / 2
        rows.append(dict(x=to_x(cx), y=to_y(cy), shape=shape, hollow=bool(hollow),
                         area_ratio=a / typical, px=cx, py=cy, w=w, h=h))
    df = pd.DataFrame(rows).sort_values(["shape", "x"]).reset_index(drop=True)
    df["series"] = [p.series.get(s + ("_open" if hl else ""), p.series.get(s, "?"))
                    for s, hl in zip(df["shape"], df["hollow"])]
    print(f"{p.name}: frame px {tuple(round(v, 1) for v in fr)}  legend {lg}")
    print(f"  x major ticks -> {tx}")
    print(f"  y major ticks -> {ty}")
    print(f"  {len(df)} markers; typical area {typical:.0f}px; counts:",
          df.groupby("series").size().to_dict())
    big = df[(df.area_ratio > 1.5) | (df.area_ratio < 0.6)]
    if len(big):
        print("  CHECK (merged/partial blobs):")
        print(big[["x", "y", "shape", "area_ratio"]].to_string())

    # overlay image
    vis = cv2.cvtColor(((1 - b) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    colours = {"square": (0, 0, 255), "up": (0, 160, 0), "down": (255, 0, 0), "diamond": (0, 140, 255)}
    for r in df.itertuples():
        c = colours[r.shape]
        cv2.circle(vis, (int(r.px), int(r.py)), 22, c, 3)
        cv2.putText(vis, r.shape[0], (int(r.px) + 18, int(r.py) - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
    cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (200, 0, 200), 2)
    if lg:
        cv2.rectangle(vis, lg[:2], lg[2:], (0, 200, 200), 2)
    CHECK.mkdir(parents=True, exist_ok=True)
    cx0, cx1 = max(0, int(x0) - 250), min(b.shape[1], int(x1) + 30)
    cv2.imwrite(str(CHECK / f"{p.name}.png"), vis[:, cx0:cx1])
    return df


PANELS = [
    Panel("fig4a_dp_vs_re", "p07_0.png", 0, (0, 1800), (0, 1.0),
          {"square": "exp_100", "up": "exp_200"}),
    # Fig 4(b) in fig4b_templates.py (overlapping open/filled markers)
    Panel("fig5_dT_vs_re", "p08_0.png", 0, (0, 1800), (0, 80),
          {"square": "exp_100", "up": "exp_200"}),
    Panel("fig6a_tc_vs_re_100", "p09_1.png", 0, (0, 1800), (10, 100),
          {"up": "tc1", "down": "tc2", "square": "tc3", "diamond": "tc4"}),
    Panel("fig6b_tc_vs_re_200", "p09_1.png", 1, (0, 1800), (30, 120),
          {"up": "tc1", "down": "tc2", "square": "tc3", "diamond": "tc4"},
          exclude=[(1150, 100, 1800, 120)]),   # legend title at frame top
    Panel("fig7a_tc_along_x_100", "p09_0.png", 0, (0, 0.045), (0, 140),
          {"up": "re226", "down": "re890", "square": "re1454"}),
    Panel("fig7b_tc_along_x_200", "p09_0.png", 1, (0, 0.045), (0, 180),
          {"up": "re442", "down": "re864", "square": "re1220"}),

]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in PANELS:
        df = digitise(p)
        df.to_csv(OUT / f"{p.name}_raw.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
