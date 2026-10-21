"""Fig 7: digitise the three numerical-prediction curves along the thermocouple line.

Output on a 0.1 mm grid, gaps filled by linear interpolation.
Run:  .venv/Scripts/python digitise/fig7_curves.py
"""
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from digitise import CHECK, FIGS, OUT, frames, ink

XLIM = (0, 0.045)
PANELS = [
    # name, frame index, ylim, curve labels top->bottom, legend mask (data coords x0, y0, x1, y1)
    ("fig7a_num_curves_100", 0, (0, 140), ["re226", "re890", "re1454"], (0.0, 81, 0.0246, 140)),
    ("fig7b_num_curves_200", 1, (0, 180), ["re442", "re864", "re1220"], (0.0, 106, 0.0251, 180)),
]
MIN_LEN = 60   # px (~1.8 mm), minimum segment length


def extract(name, index, ylim, labels, legend) -> pd.DataFrame:
    b = ink(FIGS / "p09_0.png")
    x0, y0, x1, y1 = frames(b)[index]
    to_px = lambda xd, yd: (int(round(x0 + (xd - XLIM[0]) / (XLIM[1] - XLIM[0]) * (x1 - x0))),
                            int(round(y1 - (yd - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0))))
    to_x = lambda px: XLIM[0] + (px - x0) / (x1 - x0) * (XLIM[1] - XLIM[0])
    to_y = lambda py: ylim[0] + (y1 - py) / (y1 - y0) * (ylim[1] - ylim[0])

    work = np.zeros_like(b)
    l, t, r, btm = int(x0) + 25, int(y0) + 30, int(x1) - 25, int(y1) - 30
    work[t:btm, l:r] = b[t:btm, l:r]
    (ax, ay), (bx, by) = to_px(legend[0], legend[3]), to_px(legend[2], legend[1])
    work[ay:by, ax:bx] = 0
    # remove markers and bold labels (9x9 opening, curves are 3-5 px)
    blobs = cv2.morphologyEx(work, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    work[cv2.dilate(blobs, np.ones((11, 11), np.uint8)) > 0] = 0

    n, lab, st, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    segs = [i for i in range(1, n) if st[i, 2] >= MIN_LEN and st[i, 4] / st[i, 2] < 9]

    # segment -> {column: centre row}
    centre = {}
    for i in segs:
        x, y, w, h, _ = st[i]
        m = lab[y:y + h, x:x + w] == i
        cols = {}
        for c in range(w):
            rows = np.flatnonzero(m[:, c])
            if rows.size:
                cols[x + c] = y + rows.mean()
        centre[i] = cols

    # curve index by vertical order (top = lowest Re), voted where three segments overlap
    votes = {i: Counter() for i in segs}
    for px in range(l, r):
        present = sorted((centre[i][px], i) for i in segs if px in centre[i])
        if len(present) == 3:
            for k, (_, i) in enumerate(present):
                votes[i][k] += 1
    assign = {i: votes[i].most_common(1)[0][0] for i in segs if votes[i]}

    def curve_y(k, px):
        pts = sorted((c, yy) for i, kk in assign.items() if kk == k for c, yy in centre[i].items())
        cs, ys = zip(*pts)
        return np.interp(px, cs, ys)

    for i in segs:   # unassigned segments: nearest curve
        if i not in assign:
            mid = sorted(centre[i])[len(centre[i]) // 2]
            assign[i] = int(np.argmin([abs(curve_y(k, mid) - centre[i][mid]) for k in range(3)]))

    grid = np.round(np.arange(0.0, 0.0448 + 1e-9, 0.0001), 6)
    out = pd.DataFrame({"x": grid})
    for k, name_k in enumerate(labels):
        pts = sorted((c, yy) for i, kk in assign.items() if kk == k for c, yy in centre[i].items())
        xs = np.array([to_x(c) for c, _ in pts])
        ys = np.array([to_y(yy) for _, yy in pts])
        out[name_k] = np.interp(grid, xs, ys)
        cover = len(pts) / (r - l)
        print(f"{name} {name_k}: {sum(1 for kk in assign.values() if kk == k)} segments, "
              f"{cover:.0%} of columns traced, T(0)={out[name_k].iloc[0]:.1f} C, T(L)={out[name_k].iloc[-1]:.1f} C")

    vis = cv2.cvtColor(((1 - b) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for name_k, c in zip(labels, [(0, 0, 255), (0, 160, 0), (255, 0, 0)]):
        pts = np.array([to_px(xx, yy) for xx, yy in zip(out.x, out[name_k])], np.int32)
        cv2.polylines(vis, [pts], False, c, 2)
    cv2.imwrite(str(CHECK / f"{name}.png"), vis[:, max(0, int(x0) - 250):int(x1) + 30])
    return out


def main() -> None:
    for p in PANELS:
        extract(*p).to_csv(OUT / f"{p[0]}.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
