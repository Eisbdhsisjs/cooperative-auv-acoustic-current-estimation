# -*- coding: utf-8 -*-
"""Matched-emission and matched-depth-change comparisons from existing runs."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auvcurrent.pipeline import Survey  # noqa: E402


def truncate_emissions(lib, sel, target):
    seen = set()
    out = []
    for k in sel:
        key = (int(lib.ep[k]), int(lib.em[k]), float(lib.de[k]))
        out.append(int(k))
        seen.add(key)
        if len(seen) >= target:
            break
    return np.asarray(out, int)


def depth_cost_m(lib, sel, n_auv):
    depth = np.full(n_auv + 3, 11.0)
    total = 0.0
    for k in sel:
        e, r = int(lib.em[k]), int(lib.rc[k])
        total += abs(float(lib.de[k]) - depth[e]) + abs(float(lib.dr[k]) - depth[r])
        depth[e] = float(lib.de[k])
        depth[r] = float(lib.dr[k])
    return float(total)


def truncate_cost(lib, sel, target, n_auv):
    depth = np.full(n_auv + 3, 11.0)
    total = 0.0
    out = []
    for k in sel:
        e, r = int(lib.em[k]), int(lib.rc[k])
        step = abs(float(lib.de[k]) - depth[e]) + abs(float(lib.dr[k]) - depth[r])
        if total + step > target:
            break
        total += step
        out.append(int(k))
        depth[e] = float(lib.de[k])
        depth[r] = float(lib.dr[k])
    return np.asarray(out, int), total


def main() -> int:
    s = Survey()
    rows = {}
    for pol in ("proposed", "posterior", "uniform", "random"):
        s.select(pol)
        s.simulate()
        s.estimate()
        full = s.metrics()
        sel = s.selection
        t62 = truncate_emissions(s.library, sel, 62)
        s.simulate(t62)
        s.estimate(t62)
        m62 = s.metrics()
        tcost, cost = truncate_cost(s.library, sel, 120.0, s.cfg.auv.N)
        s.simulate(tcost)
        s.estimate(tcost)
        mcost = s.metrics()
        rows[pol] = {
            "full": {"links": int(sel.size), "rmse": round(full["rmse"], 4),
                     "depth_cost_m": round(depth_cost_m(s.library, sel, s.cfg.auv.N), 1)},
            "matched_62_emissions": {
                "links": int(t62.size), "rmse": round(m62["rmse"], 4),
                "depth_cost_m": round(depth_cost_m(s.library, t62, s.cfg.auv.N), 1)},
            "matched_120m_depth": {
                "links": int(tcost.size), "rmse": round(mcost["rmse"], 4),
                "depth_cost_m": round(depth_cost_m(s.library, tcost, s.cfg.auv.N), 1)},
        }
        print(pol, rows[pol], flush=True)
    p = os.path.join(ROOT, "results", "matched_resources.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print("saved", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
