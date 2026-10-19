"""参考时刻的联合反演（论文 §Spatiotemporally Consistent Reconstruction）。

观测模型（固定参考射线的线性化形式）::

    y = tau0 + A' theta + H_eta eta + eps,   eta 为 AUV 水平位置偏差

求解带场空间先验与轨迹干扰项的正规方程::

    theta_hat = (G' G + P)^-1 G' r
"""

from __future__ import annotations

import numpy as np

from .acoustics import travel_time


def forward_truth(cfg, lib, ssp, truth, sel, noise: bool = True, seed=None):
    """真值走时（沿参考射线积分真实流场）。"""
    m = len(sel)
    tau = np.zeros(m)
    for j, k in enumerate(sel):
        ray = lib.ray(int(k), ssp)
        tau[j] = travel_time(truth, ray, lib.t[int(k)])
    if noise:
        rng = np.random.default_rng(cfg.seed if seed is None else seed + 1000 * m)
        tau = tau + cfg.ac.noise * rng.standard_normal(m)
    return tau


def nuisance_matrix(cfg, lib, sel, cs: float) -> np.ndarray:
    """轨迹干扰项 ``H_eta``（每个 AUV 的常数水平偏移），已按先验尺度缩放。"""
    n_auv = cfg.auv.N
    H = np.zeros((len(sel), 2 * n_auv))
    for j, k in enumerate(sel):
        k = int(k)
        ex, ey = lib.e[k]
        if lib.em[k] < n_auv:
            s = 2 * lib.em[k]
            H[j, s] -= ex / cs
            H[j, s + 1] -= ey / cs
        if lib.rc[k] < n_auv:
            s = 2 * lib.rc[k]
            H[j, s] += ex / cs
            H[j, s + 1] += ey / cs
    return H / cfg.inv.traj_sigma


def invert(cfg, lib, sel, y, prior=None, sigma=None, use_nuisance=None,
           cs: float = 1500.0, cs_prior: float = 0.05):
    """线性化反演，返回系数与后验协方差。

    ``cs`` 为平均声速（用于轨迹干扰项的几何换算），``cs_prior`` 为其先验
    标准差（m/s），用于抑制声速与轨迹的耦合零空间。
    """
    sel = np.asarray(sel, int)
    sigma = float(sigma if sigma is not None else cfg.ac.sigma_t)
    use_nuisance = (cfg.inv.use_nuisance if use_nuisance is None
                    else use_nuisance)
    d = lib.n_coef
    nT = 2 * d
    if prior is None:
        raise ValueError("必须提供场空间先验矩阵 prior")

    Aw = lib.A[:, sel].T / sigma                     # m x 2d
    res = (np.asarray(y, float) - lib.tau0[sel]) / sigma

    if use_nuisance and sel.size:
        Hn = nuisance_matrix(cfg, lib, sel, cs)
        # 声速不参与反演，但其误差通过几何换算耦合，这里以弱先验正则
        G = np.hstack([Aw, Hn])
        nAux = Hn.shape[1]
        P = np.zeros((nT + nAux, nT + nAux))
        P[:nT, :nT] = prior
        P[nT:, nT:] = np.eye(nAux)
    else:
        G = Aw
        nAux = 0
        P = prior

    N = G.T @ G + P
    rhs = G.T @ res
    th = np.linalg.solve(N, rhs)
    try:
        Ninv = np.linalg.inv(N)
    except np.linalg.LinAlgError:                     # pragma: no cover
        Ninv = np.linalg.pinv(N)

    resid = res - G @ th
    return {
        "theta": th[:nT],
        "a0": th[:d].copy(),
        "adot": th[d:nT].copy(),
        "eta": th[nT:] if nAux else np.zeros(0),
        "P": Ninv,
        "Pa": Ninv[:nT, :nT],
        "res": resid,
        "rms_res": float(np.sqrt(np.mean(resid ** 2)) * sigma) if resid.size else 0.0,
        "m": int(sel.size),
    }


def errors(ut, vt, ue, ve):
    """矢量 RMSE、方向误差与空间相关（等价于 ``qhd_errors.m``）。"""
    ut = np.asarray(ut).ravel()
    vt = np.asarray(vt).ravel()
    ue = np.asarray(ue).ravel()
    ve = np.asarray(ve).ravel()
    du, dv = ue - ut, ve - vt
    out = {"rmse": float(np.sqrt(np.mean(du ** 2 + dv ** 2)))}
    s_t = np.hypot(ut, vt)
    s_e = np.hypot(ue, ve)
    ok = s_t > 0.05
    if np.any(ok) and np.all(s_e[ok] > 0):
        cs = (ut[ok] * ue[ok] + vt[ok] * ve[ok]) / (s_t[ok] * s_e[ok])
        cs = np.clip(cs, -1.0, 1.0)
        out["dir_err"] = float(np.mean(np.degrees(np.arccos(cs))))
        out["frac_fail"] = float(np.mean(cs < np.cos(np.deg2rad(30.0))))
    else:
        out["dir_err"] = float("nan")
        out["frac_fail"] = float("nan")
    uc, vc = ut - ut.mean(), vt - vt.mean()
    ue_c, ve_c = ue - ue.mean(), ve - ve.mean()
    num = float(np.sum(uc * ue_c + vc * ve_c))
    den = float(np.sqrt(np.sum(uc ** 2 + vc ** 2) * np.sum(ue_c ** 2 + ve_c ** 2)))
    out["corr"] = num / den if den > 0 else float("nan")
    out["bias_u"] = float(np.mean(du))
    out["bias_v"] = float(np.mean(dv))
    return out
