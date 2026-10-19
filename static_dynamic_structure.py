# -*- coding: utf-8 -*-
"""Dynamic versus static structural metrics on the same acoustic records."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auvcurrent.pipeline import Survey  # noqa: E402


def grid(s, depth, truth, n=121):
    x = np.linspace(0, s.cfg.site.Lx, n)
    y = np.linspace(0, s.cfg.site.Ly, n)
    XG, YG = np.meshgrid(x, y)
    H = s.bathy(XG.ravel(), YG.ravel())
    z = np.minimum(depth, H - 0.4)
    if truth:
        u, v = s.truth.velocity(XG.ravel(), YG.ravel(), z, 0.0)
    else:
        u, v = s.estimated_field().velocity(
            XG.ravel(), YG.ravel(), z, H, 0.0)
    return XG, YG, u.reshape(XG.shape), v.reshape(XG.shape)


def peaks(speed, frac=0.55):
    mx = ndimage.maximum_filter(speed, size=15, mode="nearest")
    thr = frac * speed.max()
    cand = (speed >= mx - 1e-9) & (speed >= thr)
    ys, xs = np.where(cand)
    return np.array(xs, float), np.array(ys, float)


def structure(s, depth):
    nx, ny = st_shape = (121, 121)
    dx = s.cfg.site.Lx / (nx - 1)
    dy = s.cfg.site.Ly / (ny - 1)
    wx = np.ones(nx); wx[[0, -1]] = 0.5
    wy = np.ones(ny); wy[[0, -1]] = 0.5
    _, _, ut, vt = grid(s, depth, True)
    _, _, ue, ve = grid(s, depth, False)
    st, se = np.hypot(ut, vt), np.hypot(ue, ve)
    peak_rmse = float(np.sqrt(np.mean((se - st) ** 2)))
    xt, yt = peaks(st)
    xe, ye = peaks(se)
    disp = []
    for a, b in zip(xt, yt):
        if len(xe) == 0:
            break
        j = int(np.argmin((xe - a) ** 2 + (ye - b) ** 2))
        disp.append(dx * np.hypot(xe[j] - a, ye[j] - b))
    def area(sp):
        mask = sp > 0.55 * sp.max()
        return float(np.sum(mask * np.outer(wy, wx)) * dx * dy / 1e6)
    return {
        "depth": depth,
        "peak_rmse": round(peak_rmse, 3),
        "n_vortices_true": int(len(xt)),
        "n_vortices_est": int(len(xe)),
        "vortex_displacement_km": round(float(np.mean(disp)) / 1e3, 3)
        if disp else None,
        "structure_area_true_km2": round(area(st), 2),
        "structure_area_est_km2": round(area(se), 2),
    }


def main() -> int:
    s = Survey()
    s.select("proposed")
    s.simulate()
    s.estimate()
    m_dyn = s.metrics()
    rows_dyn = [structure(s, z) for z in (4.0, 10.0, 18.0)]

    # Static ablation: the rate block is constrained to zero with a very
    # strong prior, while the value block keeps the declared prior.
    d = s.dictionary.n_coef
    prior_static = s.prior.copy()
    prior_static[d:, d:] *= 1e8
    s.prior = prior_static
    s.estimate()
    m_stat = s.metrics()
    rows_stat = [structure(s, z) for z in (4.0, 10.0, 18.0)]

    out = {
        "dynamic": {"rmse": round(m_dyn["rmse"], 4), "structure": rows_dyn},
        "static": {"rmse": round(m_stat["rmse"], 4), "structure": rows_stat},
    }
    p = os.path.join(ROOT, "results", "static_dynamic_structure.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("dynamic RMSE %.4f" % m_dyn["rmse"])
    print("static  RMSE %.4f" % m_stat["rmse"])
    for r in rows_dyn:
        print("dyn ", r)
    for r in rows_stat:
        print("stat", r)
    print("saved", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
