# -*- coding: utf-8 -*-
# Reviewer 3, Comment 5: depth-mode response (amplification and leakage),
# same-depth versus cross-depth schedules, and a numeric residual profile.
# Run: python extensions/vertical_resolution.py
from __future__ import annotations
import json
import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auvcurrent.pipeline import Survey
from auvcurrent.model import DictField
from auvcurrent.inverse import forward_truth
def mode_vector(dictionary, l: int):
    a = np.zeros(dictionary.n_coef)
    a[dictionary.idx_U[l]] = 1.0
    return a
def response(survey, lib, sel, a_true):
    truth = DictField(survey.dictionary, a0=a_true)
    y = forward_truth(survey.cfg, lib, survey.ssp, truth, sel, noise=False)
    saved_lib, saved_sel = survey.library, survey.selection
    survey.library, survey.selection = lib, np.asarray(sel, int)
    survey.observations = y
    survey.estimate(link_indices=sel)
    rec = survey.result["a0"]
    survey.library, survey.selection = saved_lib, saved_sel
    return rec
def main() -> int:
    s = Survey()
    lib = s.library
    s.select("proposed")
    sel = s.selection
    L, d = s.dictionary.L, s.dictionary.n_coef
    print("links =", sel.size, " L =", L, flush=True)
    gain = np.zeros((L, L))
    for l in range(L):
        rec = response(s, lib, sel, mode_vector(s.dictionary, l))
        for m in range(L):
            gain[l, m] = rec[s.dictionary.idx_U[m]]
    print("\ndepth-mode gain matrix (rows: injected mode, cols: recovered mode)")
    print("      " + "".join("%9s" % ("B%d" % (m + 1)) for m in range(L)))
    for l in range(L):
        print("  B%-3d" % (l + 1) + "".join("%9.2f" % gain[l, m] for m in range(L)))
    leak = float(np.mean([np.sum(np.abs(gain[l])) - abs(gain[l, l]) for l in range(L)]))
    print("\nmean absolute leakage into the other depth modes: %.2f" % leak)
    # same-depth versus cross-depth schedule
    same = lib.de == lib.dr
    cross = lib.de != lib.dr
    out = {"diagonal_gain": np.diag(gain).round(3).tolist(),
           "gain_matrix": gain.round(3).tolist(), "mean_leakage": round(leak, 3)}
    for name, mask in (("same_depth", same), ("cross_depth", cross)):
        if int(mask.sum()) < 10:
            print(name, "skipped (links:", int(mask.sum()), ")")
            continue
        import copy
        sub = copy.copy(lib)
        for f in ["em", "rc", "ep", "t", "de", "dr", "range", "tau0",
                  "feasible", "Hm", "x1", "y1", "x2", "y2"]:
            setattr(sub, f, getattr(lib, f)[mask])
        sub.A = lib.A[:, mask]
        for f in ("path_r", "path_z", "path_th"):
            setattr(sub, f, getattr(lib, f)[mask])
        sub.e = lib.e[mask]
        from auvcurrent.planner import select_links
        sel2 = select_links(s.cfg, sub, "proposed")
        g = np.zeros((L, L))
        for l in range(L):
            rec = response(s, sub, sel2, mode_vector(s.dictionary, l))
            for m in range(L):
                g[l, m] = rec[s.dictionary.idx_U[m]]
        out[name] = {"links": int(sel2.size), "gain": g.round(3).tolist(),
                     "diagonal": np.diag(g).round(3).tolist()}
        print("%s: %d links, diagonal %s" % (name, sel2.size,
                                             np.diag(g).round(2).tolist()))
    # numeric residual profile: error versus depth (no colour saturation)
    prof = []
    for z in (2.0, 6.0, 10.0, 14.0, 18.0):
        _, _, _, _, Ue, Ve, _ = s.field_on_grid(z, n=21, truth=False)
        _, _, _, _, Ut, Vt, _ = s.field_on_grid(z, n=21, truth=True)
        prof.append((z, float(np.sqrt(np.mean((np.asarray(Ue) - Ut) ** 2 +
                                             (np.asarray(Ve) - Vt) ** 2)))))
    out["residual_profile"] = prof
    print("\nresidual profile (depth m, RMSE m/s):",
          ", ".join("%.0f:%.3f" % p for p in prof))
    p = os.path.join(ROOT, "results", "vertical_resolution.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("saved", p)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
