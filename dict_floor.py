# -*- coding: utf-8 -*-
# Reviewer 3, Comment 6: representation error of the dictionary at the working
# size and its dependence on L and Q (the one-curve-per-depth-function family).
# Run: python extensions/dict_floor.py
from __future__ import annotations
import json
import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auvcurrent.pipeline import Survey
from auvcurrent.model import Dictionary
QSET = [5, 6, 7, 8, 9]
LSET = [3, 4, 5]
def layout(Q, Lx, Ly, margin=0.12):
    R = (1.0 - margin) * min(Lx, Ly) / 2.0
    j = np.arange(Q) + 0.5
    r = R * np.sqrt(j / Q)
    th = j * 2.399963229728653
    cen = np.column_stack([Lx / 2 + r * np.cos(th), Ly / 2 + r * np.sin(th)])
    return cen, np.full(Q, 1.15 * np.sqrt(Lx * Ly / (np.pi * Q)))
def main() -> int:
    s = Survey()
    cfg = s.cfg
    x, y, XG, YG = s.grid(21)
    H = s.bathy(XG.ravel(), YG.ravel())
    rows, b = [], None
    for L in LSET:
        for Q in QSET:
            cen, sca = layout(Q, cfg.site.Lx, cfg.site.Ly)
            d = Dictionary(L, Q, cfg.dictionary.Href, cen, sca,
                           cfg.site.Lx, cfg.site.Ly)
            mats, rhs = [], []
            for zd in cfg.display.depths:
                z = np.minimum(float(zd), H - 0.4)
                Du, Dv = d.velocity_basis(XG.ravel(), YG.ravel(), z, H)
                ut, vt = s.truth.velocity(XG.ravel(), YG.ravel(), z, 0.0)
                mats.append(np.vstack([Du, Dv]))
                rhs.append(np.concatenate([ut, vt]))
            G = np.vstack(mats)
            b = np.concatenate(rhs)
            a, *_ = np.linalg.lstsq(G, b, rcond=1e-10)
            resid = G @ a - b
            floor = float(np.sqrt(np.mean(resid ** 2)))
            rows.append({"L": L, "Q": Q, "coef": int(d.n_coef),
                         "floor_rms": round(floor, 4),
                         "floor_pct": round(100.0 * floor / float(np.sqrt(np.mean(b ** 2))), 1)})
            print("L=%d Q=%d coef=%3d floor=%.4f m/s (%.1f%% of field rms)"
                  % (L, Q, d.n_coef, floor, rows[-1]["floor_pct"]), flush=True)
    p = os.path.join(ROOT, "results", "dict_floor.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print("saved", p)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
