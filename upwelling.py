# -*- coding: utf-8 -*-
# Reviewer 2, Comment 2: quantify the impact of the excluded upwelling and
# vertical-velocity dynamics. A vertical current w(z, x) is imposed on the
# reference field; its travel-time signature is integrated along the solved rays
# and added to the observations, which are then inverted with the divergence-free
# horizontal model. Run: python extensions/upwelling.py
from __future__ import annotations
import json
import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auvcurrent.pipeline import Survey
from auvcurrent.inverse import forward_truth
def vertical_signature(s, sel, w0):
    dt = np.zeros(len(sel))
    Lx = float(s.cfg.site.Lx)
    for j, k in enumerate(sel):
        ray = s.library.ray(int(k), s.ssp)
        r = np.asarray(ray["r"]); z = np.asarray(ray["z"])
        H = float(ray["Hm"]); c = np.asarray(ray["c"])
        if r.size < 3:
            continue
        dr = np.gradient(r); dz = np.gradient(z)
        ds = np.hypot(dr, dz)
        ez = dz / np.maximum(ds, 1e-9)
        xm = 0.5 * (ray["x1"] + ray["x2"])
        w = w0 * np.cos(2 * np.pi * xm / Lx) * np.sin(np.pi * (H - z) / H)
        dt[j] = -float(np.sum(ez * w / c ** 2 * ds))
    return dt
def main() -> int:
    s = Survey()
    s.select("proposed")
    sel = s.selection
    y0 = forward_truth(s.cfg, s.library, s.ssp, s.truth, sel, noise=False)
    s.observations = y0
    s.estimate(link_indices=sel)
    base = s.metrics()
    print("no mismatch:  RMSE %.4f  dir %.1f deg  corr %+.3f"
          % (base["rmse"], base["dir_err"], base["corr"]), flush=True)
    rows = []
    for w0 in (0.01, 0.02, 0.05, 0.10):
        dt = vertical_signature(s, sel, w0)
        s.observations = y0 + dt
        s.estimate(link_indices=sel)
        m = s.metrics()
        rows.append({"w0": w0, "rmse": float(m["rmse"]), "dir_err": float(m["dir_err"]),
                     "corr": float(m["corr"]), "dtau_rms_s": float(np.sqrt(np.mean(dt ** 2)))})
        print("w0=%.2f m/s: RMSE %.4f (+%.1f%%)  dir %.1f deg  corr %+.3f  dtau rms %.2f us"
              % (w0, m["rmse"], 100 * (m["rmse"] / base["rmse"] - 1), m["dir_err"],
                 m["corr"], 1e6 * rows[-1]["dtau_rms_s"]), flush=True)
    out = {"baseline": {"rmse": float(base["rmse"]), "dir_err": float(base["dir_err"]),
                        "corr": float(base["corr"])}, "cases": rows}
    p = os.path.join(ROOT, "results", "upwelling.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("saved", p)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
