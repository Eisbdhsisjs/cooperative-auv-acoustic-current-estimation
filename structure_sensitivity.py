# -*- coding: utf-8 -*-
# Reviewer 1 C6 / Reviewer 3 C7: structure metrics of the reconstructed field;
# Reviewer 3 C12: sensitivity to whether the 5 m trajectory uncertainty is
# modelled (nuisance terms on/off). Real pipeline, Qinhuangdao inputs.
from __future__ import annotations
import json
import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auvcurrent.pipeline import Survey
from scipy import ndimage

def grid(s, depth, truth, n=121):
    x = np.linspace(0, s.cfg.site.Lx, n)
    y = np.linspace(0, s.cfg.site.Ly, n)
    XG, YG = np.meshgrid(x, y)
    H = s.bathy(XG.ravel(), YG.ravel())
    z = np.minimum(depth, H - 0.4)
    if truth:
        u, v = s.truth.velocity(XG.ravel(), YG.ravel(), z, 0.0)
    else:
        u, v = s.estimated_field().velocity(XG.ravel(), YG.ravel(), z, H, 0.0)
    return XG, YG, u.reshape(XG.shape), v.reshape(XG.shape)

def peaks(speed, frac=0.55):
    mx = ndimage.maximum_filter(speed, size=15, mode="nearest")
    thr = frac * speed.max()
    cand = (speed >= mx - 1e-9) & (speed >= thr)
    ys, xs = np.where(cand)
    return np.array(xs, float), np.array(ys, float)

def structure(s, depth):
    dx = s.cfg.site.Lx / 120.0
    _, _, ut, vt = grid(s, depth, True)
    _, _, ue, ve = grid(s, depth, False)
    st, se = np.hypot(ut, vt), np.hypot(ue, ve)
    peak_err = float(np.sqrt(np.mean((se - st) ** 2)))
    xt, yt = peaks(st)
    xe, ye = peaks(se)
    disp = []
    for a, b in zip(xt, yt):
        if len(xe) == 0:
            break
        j = int(np.argmin((xe - a) ** 2 + (ye - b) ** 2))
        disp.append(dx * np.hypot(xe[j] - a, ye[j] - b))
    area = lambda sp: float(np.sum(sp > 0.55 * sp.max())) * dx * dx / 1e6  # km^2
    return {
        "depth": depth,
        "peak_rmse": round(peak_err, 3),
        "n_vortices_true": int(len(xt)), "n_vortices_est": int(len(xe)),
        "vortex_displacement_km": round(float(np.mean(disp)) / 1e3, 3) if disp else None,
        "structure_area_true_km2": round(area(st), 2),
        "structure_area_est_km2": round(area(se), 2),
    }

def main() -> int:
    s = Survey()
    s.select("proposed")
    s.simulate()
    s.estimate(use_nuisance=True)
    m_full = s.metrics()
    s.estimate(use_nuisance=False)
    m_non = s.metrics()
    print("trajectory nuisance: on %.4f m/s, off %.4f m/s (delta %.3f)"
          % (m_full["rmse"], m_non["rmse"], m_non["rmse"] - m_full["rmse"]), flush=True)
    rows = [structure(s, z) for z in (4.0, 10.0, 18.0)]
    for r in rows:
        print(r, flush=True)
    out = {"nuisance": {"on": round(m_full["rmse"], 4), "off": round(m_non["rmse"], 4)},
           "structure": rows}
    p = os.path.join(ROOT, "results", "structure_sensitivity.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("saved", p)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
