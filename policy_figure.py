# -*- coding: utf-8 -*-
# Corrected Figure 7: five policies plus a D-optimal (experimental-design)
# baseline, all at the same 150-link budget, with policy names on every panel.
# Run: python extensions/policy_figure.py
from __future__ import annotations
import json
import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auvcurrent.pipeline import Survey
from auvcurrent.planner import scaled_columns
def d_optimal(survey, max_sel=150):
    lib, cfg = survey.library, survey.cfg
    C = scaled_columns(cfg, lib)                 # n x 2d
    F = np.asarray(survey.prior, float).copy()
    sel, remaining = [], np.arange(lib.n)
    depth = np.full(cfg.auv.N + cfg.buoy.xy.shape[0], 11.0)
    cost_total, budget = 0.0, cfg.budget.move / 10.0 * (max_sel / 150.0)
    from auvcurrent.planner import _depth_cost
    for _ in range(max_sel):
        if remaining.size == 0:
            break
        Z = C[remaining].T
        Y = np.linalg.solve(F, Z)
        gain = np.sum(Z * Y, axis=0)
        cost = np.array([_depth_cost(cfg, lib, int(k), depth) for k in remaining])
        pick = int(np.argmax(gain / cost))
        k = int(remaining[pick])
        sel.append(k)
        c = C[k]
        F += np.outer(c, c)
        cost_total += cost[pick]
        depth[lib.em[k]] = lib.de[k]
        depth[lib.rc[k]] = lib.dr[k]
        remaining = np.delete(remaining, pick)
        if cost_total >= budget:
            break
    return np.array(sel, int)
def main() -> int:
    s = Survey()
    rows = []
    for pol in ("proposed", "posterior", "uniform", "random", "fixed"):
        s.select(pol); s.simulate(); s.estimate(); m = s.metrics()
        C = scaled_columns(s.cfg, s.library)
        rows.append({"policy": pol, "links": int(s.selection.size),
                     "rmse": float(m["rmse"]), "dir": float(m["dir_err"]),
                     "corr": float(m["corr"]),
                     "rank": int(np.linalg.matrix_rank(C[s.selection].T))})
        print(rows[-1], flush=True)
    sel = d_optimal(s)
    s.simulate(sel); s.estimate(sel); m = s.metrics()
    C = scaled_columns(s.cfg, s.library)
    rows.append({"policy": "D-optimal", "links": int(sel.size),
                 "rmse": float(m["rmse"]), "dir": float(m["dir_err"]),
                 "corr": float(m["corr"]),
                 "rank": int(np.linalg.matrix_rank(C[sel].T))})
    print(rows[-1], flush=True)
    p = os.path.join(ROOT, "results", "policies.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    # figure
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 7,
                         "axes.spines.right": False, "axes.spines.top": False,
                         "axes.linewidth": 0.8, "legend.frameon": False})
    lab = {"proposed": "Information driven", "posterior": "Posterior only",
           "uniform": "Uniform", "random": "Random", "fixed": "Fixed trajectory",
           "D-optimal": "D-optimal (published)"}
    col = {"proposed": "#0F4D92", "posterior": "#3775BA", "uniform": "#8BCF8B",
           "random": "#B64342", "fixed": "#767676", "D-optimal": "#42949E"}
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for a, key, ylab in ((ax[0], "rmse", "RMSE (m s$^{-1}$)"),
                         (ax[1], "dir", "Direction error (deg)"),
                         (ax[2], "rank", "Support rank")):
        a.bar(x, [r[key] for r in rows], width=0.66,
              color=[col[r["policy"]] for r in rows])
        a.set_xticks(x)
        a.set_xticklabels([lab[r["policy"]].replace(" ", "\n") for r in rows],
                          fontsize=5.6)
        a.set_ylabel(ylab)
        for xi, r in zip(x, rows):
            a.annotate("%.2f" % r[key] if key != "rank" else "%d" % r[key],
                       (xi, r[key]), textcoords="offset points", xytext=(0, 2),
                       ha="center", fontsize=5.6)
    out = os.path.join(ROOT, "figures", "nature", "Fig7_policies")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    print("saved", out)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
