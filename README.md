# cooperative-auv-acoustic-current-estimation
Simulation code and result files for "A Simulation Study of Cooperative AUV Acoustic Sensing for Ocean Current Field Estimation"# Supplementary Code S1

Minimal code and result package for the manuscript
*A Simulation Study of Cooperative AUV Acoustic Sensing for Ocean Current Field
Estimation*.

The package is intentionally small: it contains the Python reference
implementation, the scripts that generate the numerical results and the result
JSON files used in the manuscript. It does not contain the archived ADCP/CTD
records or the large cached link libraries.

## Contents

- `auvcurrent/`: Python pipeline (configuration, forward model, planner,
  inversion, metrics).
- `extensions/`: scripts for the reviewer-driven analyses:
  - `extended_study.py` — baselines, ablations, prior-free inversion, coverage,
    threshold scans;
  - `policy_figure.py` — five-policy comparison and Figure 7;
  - `matched_resources.py` — matched-emission and 120 m depth-cap comparison
    (Table 2 of the manuscript);
  - `static_dynamic_structure.py` — dynamic/static structural metrics
    (Table 5);
  - `vertical_resolution.py` — depth-mode gain matrix and leakage (Table 3);
  - `upwelling.py` — direct vertical-velocity travel-time sensitivity;
  - `structure_sensitivity.py`, `blockwise.py`, `dict_floor.py`,
    `baselines.py` — supporting analyses.
- `results/`: JSON files used for the tables and figures.
- `requirements.txt`: Python dependencies.

## Requirements

Python 3.12 with `numpy`, `scipy`, `matplotlib` and `h5py`.

```bash
pip install -r requirements.txt
```

## Data

The archived environmental inputs are not redistributed here because the
original ADCP/CTD records are subject to the data provider's terms. The derived
input file `qhd_inputs.mat` can be requested from the corresponding author, or
regenerated from the archived records with the MATLAB preparation script.

Place the file at:

```text
Work3_Fish/Simulation_QHD/data/qhd_inputs.mat
```

and set `MATLAB_SIM_DIR` in `auvcurrent/config.py` accordingly. Without the
file the pipeline falls back to its built-in synthetic scenario; that is useful
for a smoke test but does not reproduce the manuscript numbers.

## Reproduction

From the repository root that contains `auvcurrent/` and `results/`:

```bash
python extensions/extended_study.py
python extensions/policy_figure.py
python extensions/matched_resources.py
python extensions/static_dynamic_structure.py
python extensions/vertical_resolution.py
python extensions/upwelling.py
```

The random seed is `20260813` (`auvcurrent/config.py`). The first run may build
the ray-link library and can take several minutes; later runs reuse the cache.

## Implementation note

The manuscript combines two implementations:

- the Python pipeline is used for the main common-time reconstruction
  (0.248 m/s), the policy comparison (Figure 7), the matched-resource table
  (Table 2), the dynamic/static table (Table 5) and the modal gain matrix
  (Table 3);
- the MATLAB robustness/attribution runs are used for Figure 8 and the
  sensitivity panels of Figure 11. Their zero-perturbation level is 0.31 m/s
  and is not interchangeable with the Python 0.248 m/s result.

## License

No license is asserted in this review package. The authors should choose a
license (for example MIT or Apache-2.0) before public archiving.

[README.md](https://github.com/user-attachments/files/33652966/README.md)
