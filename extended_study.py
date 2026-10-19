# Reviewer-driven extension study on the real pipeline (Qinhuangdao inputs).
# Computes: zero-field and climatological baselines with skill scores,
# matched-budget policy comparison, ablations under a restricted candidate
# library, a prior-free inversion, sensitivity to kappa / rank tolerance, and
# the empirical coverage of the nominal 95% posterior interval.
# Run: python extensions/extended_study.py
from __future__ import annotations
import copy
import json
import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auvcurrent.pipeline import Survey
from auvcurrent.planner import scaled_columns
POLICIES = ["proposed", "posterior", "uniform", "random", "fixed"]
ARR = ["em", "rc", "ep", "t", "de", "dr", "range", "tau0", "feasible",
       "Hm", "x1", "y1", "x2", "y2"]
def subset(lib, mask: np.ndarray):
    # Return a copy of the link library restricted to mask.
    out = copy.copy(lib)
    for name in ARR:
        setattr(out, name, getattr(lib, name)[mask])
    out.A = lib.A[:, mask]
    for name in ("path_r", "path_z", "path_th"):
        setattr(out, name, getattr(lib, name)[mask])
    out.e = lib.e[mask]
    return out
def empirical_coverage(survey: Survey, depth: float = 10.0, n: int = 21):
    # Fraction of grid points whose truth lies inside the nominal 95% interval.
    if survey.result is None:
        return float("nan")
    x, y, XG, YG = survey.grid(n)
    H = survey.bathy(XG.ravel(), YG.ravel())
    z = np.minimum(depth, H - 0.4)
    Du, Dv = survey.dictionary.velocity_basis(XG.ravel(), YG.ravel(), z, H)
    d = survey.library.n_coef
    Pa = survey.result["Pa"][:d, :d]
    su = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Du, Pa, Du), 0.0))
    sv = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Dv, Pa, Dv), 0.0))
    a = survey.result["a0"]
    ue, ve = Du @ a, Dv @ a
    ut, vt = survey.truth.velocity(XG.ravel(), YG.ravel(), z, 0.0)
    ok = np.concatenate([np.abs(ue - ut) <= 1.96 * su, np.abs(ve - vt) <= 1.96 * sv])
    return float(np.mean(ok))
def assess(survey: Survey, policy: str = "proposed", lib=None, prior=None):
    # Select, simulate, invert, score; returns one compact metric row.
    if lib is not None:
        survey.library = lib
    survey.select(policy)
    survey.simulate()
    saved = survey.prior
    if prior is not None:
        survey.prior = prior
    survey.estimate()
    m = survey.metrics()
    C = scaled_columns(survey.cfg, survey.library)
    sel = survey.selection
    rank = int(np.linalg.matrix_rank(C[sel].T)) if sel.size else 0
    cov = empirical_coverage(survey)
    survey.prior = saved
    return {"links": int(sel.size), "rmse": float(m["rmse"]),
            "dir_err": float(m["dir_err"]), "corr": float(m["corr"]),
            "support_rank": rank, "coverage": cov}
def zero_field_rmse(survey: Survey, n: int = 21) -> float:
    _, _, _, _, U, V, _ = survey.field_on_grid(10.0, n=n, truth=True)
    return float(np.mean(np.hypot(np.asarray(U).ravel(), np.asarray(V).ravel())))
def climate_field_rmse(survey: Survey, n: int = 21) -> float:
    # RMSE of the best spatially uniform (climatological) field.
    _, _, _, _, U, V, _ = survey.field_on_grid(10.0, n=n, truth=True)
    u, v = np.asarray(U).ravel(), np.asarray(V).ravel()
    return float(np.sqrt(np.mean((u - u.mean()) ** 2 + (v - v.mean()) ** 2)))
def main() -> int:
    survey = Survey()
    print(survey.summary(), flush=True)
    ref = zero_field_rmse(survey)
    clim = climate_field_rmse(survey)
    print("reference rms (zero-field RMSE) = %.4f m/s" % ref, flush=True)
    print("climatological-baseline RMSE    = %.4f m/s" % clim, flush=True)
    rows = {}
    lib0 = survey.library
    for pol in POLICIES:
        rows[pol] = assess(survey, pol)
        print("[policy] %-10s %s" % (pol, rows[pol]), flush=True)
    n_auv = int(survey.cfg.auv.N)
    masks = {"stationary_UASN": (lib0.em >= n_auv) & (lib0.rc >= n_auv),
             "no_depth_change": lib0.de == lib0.dr,
             "no_inter_AUV": ~((lib0.em < n_auv) & (lib0.rc < n_auv))}
    key = np.minimum(lib0.em, lib0.rc) * 1000 + np.maximum(lib0.em, lib0.rc)
    ep_key = lib0.ep * 1000000 + key
    _, first = np.unique(ep_key, return_index=True)
    masks["no_reciprocal_bundles"] = np.isin(np.arange(lib0.n), first)
    for name, mask in masks.items():
        if int(mask.sum()) < 10:
            print("[ablate] %-20s skipped (only %d links)" % (name, int(mask.sum())), flush=True)
            continue
        try:
            rows[name] = assess(survey, "proposed", lib=subset(lib0, mask))
            print("[ablate] %-20s %s" % (name, rows[name]), flush=True)
        except Exception as exc:
            print("[ablate] %-20s FAILED %s" % (name, exc), flush=True)
    d = int(lib0.n_coef)
    weak = 1e-8 * np.eye(2 * d)
    rows["prior_free"] = assess(survey, "proposed", lib=lib0, prior=weak)
    print("[ctrl  ] %-20s %s" % ("prior_free", rows["prior_free"]), flush=True)
    scan = []
    for kappa in (0.02, 0.05, 0.10, 0.20):
        survey.cfg.design.kappa = float(kappa)
        r = assess(survey, "proposed", lib=lib0)
        scan.append(dict(kappa=kappa, rank_tol=float(survey.cfg.design.rank_tol), **r))
        print("[kappa] %.2f rmse=%.4f rank=%d" % (kappa, r["rmse"], r["support_rank"]), flush=True)
    survey.cfg.design.kappa = 0.05
    for tol in (1e-3, 1e-2, 1e-1):
        survey.cfg.design.rank_tol = float(tol)
        r = assess(survey, "proposed", lib=lib0)
        scan.append(dict(kappa=0.05, rank_tol=tol, **r))
        print("[rankTol] %.0e rmse=%.4f rank=%d" % (tol, r["rmse"], r["support_rank"]), flush=True)
    survey.cfg.design.rank_tol = 1e-2
    out = {"reference_rms": ref, "climate_rms": clim, "configs": rows, "scan": scan}
    path = os.path.join(ROOT, "results", "extended_study.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("saved", path)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
