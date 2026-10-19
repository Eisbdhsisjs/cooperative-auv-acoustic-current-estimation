"""Reviewer-driven extension: matched-budget policy comparison, zero-field
baseline and skill score, computed with the existing (real) pipeline.

Run:
    python extensions/baselines.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from auvcurrent.pipeline import Survey  # noqa: E402

POLICIES = ["proposed", "posterior", "uniform", "random", "fixed"]


def zero_field_rmse(survey: Survey, n: int = 21) -> float:
    """RMSE of a zero current field against the reference field."""
    _, _, _, _, U, V, _ = survey.field_on_grid(10.0, n=n, truth=True)
    u = np.asarray(U).ravel()
    v = np.asarray(V).ravel()
    return float(np.sqrt(np.mean(u ** 2 + v ** 2)))


def main() -> int:
    survey = Survey()
    print(survey.summary())

    ref = zero_field_rmse(survey)
    rows = []
    for pol in POLICIES:
        survey.select(pol)
        survey.simulate()
        survey.estimate(np.concatenate(survey.by_epoch) if survey.by_epoch else None)
        m = survey.metrics()
        rows.append({
            "policy": pol,
            "links": int(survey.sel.size) if getattr(survey, "sel", None) is not None else None,
            "rmse": float(m["rmse"]),
            "dir_err": float(m["dir_err"]),
            "corr": float(m["corr"]),
            "skill_vs_zero": float(1.0 - m["rmse"] / ref) if ref > 0 else float("nan"),
        })
        print("%-10s links=%4d  RMSE=%.4f  dir=%5.1f°  corr=%+.3f  skill=%.3f"
              % (pol, rows[-1]["links"] or 0, m["rmse"], m["dir_err"], m["corr"],
                 rows[-1]["skill_vs_zero"]))

    out = {
        "reference_rms": ref,
        "policies": rows,
    }
    path = os.path.join(ROOT, "results", "baselines.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("\nzero-field RMSE = %.4f m/s ; saved %s" % (ref, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
