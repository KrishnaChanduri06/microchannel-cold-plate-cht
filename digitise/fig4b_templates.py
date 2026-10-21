"""Fig 4(b) digitiser: template matching against the four legend symbols.

Run:  .venv/Scripts/python digitise/fig4b_templates.py
"""
import cv2
import numpy as np
import pandas as pd

from digitise import CHECK, FIGS, OUT, frames, ink, legend_box

XLIM, YLIM = (0, 1800), (0, 1.0)
SERIES = ["num_100", "exp_100", "num_200", "exp_200"]   # legend order, top to bottom


def legend_symbols(b, lg):
    """Legend symbol templates, top to bottom."""
    x0, y0, x1, y1 = lg
    strip = b[y0 + 8:y1 - 8, x0 + 8:x0 + 120]
    n, lab, st, _ = cv2.connectedComponentsWithStats(strip)
    # symbols ~20-35 px, excludes legend border
    comps = sorted((st[i] for i in range(1, n)
                    if st[i, 4] > 100 and 12 < st[i, 2] < 45 and 12 < st[i, 3] < 45), key=lambda s: s[1])
    out = []
    for x, y, w, h, _ in comps:
        out.append(b[y0 + 8 + y - 2:y0 + 8 + y + h + 2, x0 + 8 + x - 2:x0 + 8 + x + w + 2].astype(np.float32))
    return out


def main() -> None:
    b = ink(FIGS / "p07_0.png")
    fr = frames(b)[1]
    x0, y0, x1, y1 = fr
    lg = legend_box(b, fr)
    temps = legend_symbols(b, lg)
    assert len(temps) == 4, f"expected 4 legend symbols, got {len(temps)}"

    plot = b.astype(np.float32).copy()
    plot[lg[1]:lg[3], lg[0]:lg[2]] = 0
    plot[:, :int(x0) + 6] = 0
    plot[int(y1) - 6:, :] = 0

    rows = []
    for name, t in zip(SERIES, temps):
        th, tw = t.shape
        # normalised cross-correlation, peaks above 0.55
        res = cv2.matchTemplate(plot, t, cv2.TM_CCOEFF_NORMED)
        found = []
        r = res.copy()
        while True:
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx < 0.55:
                break
            found.append((mx, loc))
            cv2.rectangle(r, (loc[0] - tw // 2, loc[1] - th // 2), (loc[0] + tw // 2, loc[1] + th // 2), -1, -1)
        for score, (lx, ly) in found:
            cx, cy = lx + tw / 2, ly + th / 2
            rows.append(dict(series=name, score=round(score, 3), px=cx, py=cy,
                             x=XLIM[0] + (cx - x0) / (x1 - x0) * (XLIM[1] - XLIM[0]),
                             y=YLIM[0] + (y1 - cy) / (y1 - y0) * (YLIM[1] - YLIM[0])))
    df = pd.DataFrame(rows)
    # best score within 8 px (open/filled pairs ~11 px apart); drop frame-corner hits
    keep = []
    for i, r in df.iterrows():
        near = df[(abs(df.px - r.px) < 8) & (abs(df.py - r.py) < 8)]
        if r.score >= near.score.max() and r.x < 0.98 * XLIM[1]:
            keep.append(i)
    df = df.loc[keep].sort_values(["series", "x"])
    print(df.to_string(index=False))

    vis = cv2.cvtColor(((1 - b) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    col = {"num_100": (0, 0, 255), "exp_100": (255, 0, 0), "num_200": (0, 160, 0), "exp_200": (0, 140, 255)}
    for r in df.itertuples():
        cv2.circle(vis, (int(r.px), int(r.py)), 4, col[r.series], -1)
        cv2.putText(vis, r.series, (int(r.px) + 15, int(r.py)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col[r.series], 2)
    cv2.imwrite(str(CHECK / "fig4b_templates.png"), vis[:, int(x0) - 250:int(x1) + 30])
    df.to_csv(OUT / "fig4b_dp_num_vs_exp_raw.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
