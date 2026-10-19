# Algorithm change: block-wise (hierarchical) prior.
# Rationale: the error decomposition showed that the horizontally uniform
# (layer-mean) coefficients dominate the error, so the single-sigma prior
# over-constrains them. Here the prior precision of the layer-uniform
# coefficients is scaled by rho^2 (both the field and the rate block) while the
# eddy coefficients keep the published prior. Validated over noise seeds.
# Run: python extensions/blockwise.py
from __future__ import annotations
import json
import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from auvcurrent.pipeline import Survey
RHO = [1.0, 0.2, 0.05, 0.02, 0.01, 0.005]
SEEDS = list(range(1, 9))
def block_prior(dictionary, prior, rho: float):
    P = prior.copy()
    d = int(dictionary.n_coef)
    idx = np.concatenate([dictionary.idx_U, dictionary.idx_V]).astype(int)
    for i in np.concatenate([idx, idx + d]):
        P[i, i] *= rho ** 2
    return P
def decompose(survey, n=21, depth=10.0):
    _, _, _, _, Ue, Ve, _ = survey.field_on_grid(depth, n=n, truth=False)
    _, _, _, _, Ut, Vt, _ = survey.field_on_grid(depth, n=n, truth=True)
    ue, ve = np.asarray(Ue).ravel(), np.asarray(Ve).ravel()
    ut, vt = np.asarray(Ut).ravel(), np.asarray(Vt).ravel()
    uni = float(np.hypot(ue.mean() - ut.mean(), ve.mean() - vt.mean()))
    res = (ue - ue.mean()) - (ut - ut.mean()), (ve - ve.mean()) - (vt - vt.mean())
    return uni, float(np.sqrt(np.mean(res[0] ** 2 + res[1] ** 2)))
def main() -> int:
    survey = Survey()
    base = survey.prior.copy()
    survey.select("proposed")
    print("links =", survey.selection.size, " seeds =", len(SEEDS), flush=True)
    rows = []
    for rho in RHO:
        survey.prior = block_prior(survey.dictionary, base, rho)
        rms, dirs, corrs, uni, stru = [], [], [], [], []
        for seed in SEEDS:
            survey.simulate(seed=seed)
            survey.estimate()
            m = survey.metrics()
            rms.append(m["rmse"]); dirs.append(m["dir_err"]); corrs.append(m["corr"])
            u, s = decompose(survey); uni.append(u); stru.append(s)
        row = {"rho": rho, "eddy": 1.0, "rmse_mean": float(np.mean(rms)),
               "rmse_sem": float(np.std(rms, ddof=1) / np.sqrt(len(rms))),
               "dir_mean": float(np.mean(dirs)), "corr_mean": float(np.mean(corrs)),
               "uniform_err": float(np.mean(uni)), "structure_err": float(np.mean(stru))}
        rows.append(row)
        print("rho=%5.2f  RMSE=%.4f +- %.4f  dir=%5.1f  corr=%+.3f  uniform=%.4f  struct=%.4f"
              % (rho, row["rmse_mean"], row["rmse_sem"], row["dir_mean"],
                 row["corr_mean"], row["uniform_err"], row["structure_err"]), flush=True)
    best = min(rows, key=lambda r: r["rmse_mean"])
    print("\nbest rho=%.2f  RMSE=%.4f (published rho=1: %.4f)  gain %.1f%%"
          % (best["rho"], best["rmse_mean"], rows[0]["rmse_mean"],
             100.0 * (rows[0]["rmse_mean"] - best["rmse_mean"]) / rows[0]["rmse_mean"]))
    path = os.path.join(ROOT, "results", "blockwise.json")
    print("\n--- eddy prior strengthened at rho = %.2f ---" % best["rho"], flush=True)
    for eddy in (2.0, 4.0, 8.0):
        P = block_prior(survey.dictionary, base, best["rho"])
        d = int(survey.dictionary.n_coef)
        keep = np.ones(d, bool)
        keep[np.concatenate([survey.dictionary.idx_U, survey.dictionary.idx_V]).astype(int)] = False
        for i in np.concatenate([np.where(keep)[0], np.where(keep)[0] + d]):
            P[i, i] *= eddy ** 2
        survey.prior = P
        rms = []
        for seed in SEEDS:
            survey.simulate(seed=seed)
            survey.estimate()
            rms.append(survey.metrics()["rmse"])
        rows.append({"rho": best["rho"], "eddy": eddy,
                     "rmse_mean": float(np.mean(rms)),
                     "rmse_sem": float(np.std(rms, ddof=1) / np.sqrt(len(rms)))})
        print("rho=%.3f eddy=%.1f  RMSE=%.4f +- %.4f"
              % (best["rho"], eddy, np.mean(rms), rows[-1]["rmse_sem"]), flush=True)
    best = min(rows, key=lambda r: r["rmse_mean"])
    print("\nFINAL best: rho=%.3f eddy=%.1f RMSE=%.4f (baseline %.4f, gain %.1f%%)"
          % (best["rho"], best["eddy"], best["rmse_mean"], rows[0]["rmse_mean"],
             100.0 * (rows[0]["rmse_mean"] - best["rmse_mean"]) / rows[0]["rmse_mean"]))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print("saved", path)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
