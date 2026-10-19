"""信息驱动的协同采样（论文 §Cooperative Information-Driven Sampling）。

论文的选择准则是字典序三元组 ``(r_D, -D_kappa, -L)`` 除以移动代价。
离线严格版本见 MATLAB ``qhd_select.m``/``qhd_metric.m``；交互版对同一
准则做了向量化近似（在同一特征基下同时评估三项增益），以保证界面实时响应。
"""

from __future__ import annotations

import numpy as np


def scaled_columns(cfg, lib) -> np.ndarray:
    """把灵敏度按物理尺度与时间戳精度白化：返回 ``(n_link, 2*d)``。"""
    d = lib.n_coef
    sc = np.concatenate([np.full(d, cfg.inv.prior_sigma),
                         np.full(d, cfg.inv.rate_prior_sigma)])
    return (lib.A * sc[:, None]).T / cfg.ac.sigma_t


def _depth_cost(cfg, lib, k: int, depth_now: np.ndarray) -> float:
    dchg = (abs(lib.de[k] - depth_now[lib.em[k]]) +
            abs(lib.dr[k] - depth_now[lib.rc[k]]))
    return 1.0 + 0.5 * dchg / 10.0


def _greedy_scores(cfg, C, F, cand, prior_scale, rank_tol, kappa, ridge=1e-9):
    """在当前支撑矩阵 ``F`` 的特征基下计算三项增益（向量化）。"""
    lam, V = np.linalg.eigh(F + ridge * np.eye(F.shape[0]))
    lam = np.maximum(lam, 0.0)
    Z = V.T @ C[cand].T                     # (2d, ncand)
    sq = Z ** 2
    w_support = (lam <= rank_tol).astype(float)[:, None]
    w_weak = (1.0 / (lam + kappa) ** 2)[:, None]
    w_post = (1.0 / (lam + prior_scale) ** 2)[:, None]
    return (sq * w_support).sum(0), (sq * w_weak).sum(0), (sq * w_post).sum(0)


def select_links(cfg, lib, policy: str = "proposed", seed: int | None = None,
                 progress=None) -> np.ndarray:
    """返回按采集顺序排列的链路索引（含移动/换深代价预算约束）。"""
    n = lib.n
    if n == 0:
        return np.zeros(0, int)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    C = scaled_columns(cfg, lib)
    d = lib.n_coef
    n_plat = cfg.auv.N + cfg.buoy.xy.shape[0]
    depth_now = np.full(n_plat, 11.0)
    budget_cost = cfg.budget.move / 10.0
    max_sel = int(cfg.design.max_sel)
    prior_scale = 1.0 / max(cfg.inv.field_sigma, 1e-6) ** 2

    sel: list[int] = []
    cost_total = 0.0

    if policy in ("proposed", "posterior"):
        F = np.zeros((2 * d, 2 * d))
        remaining = np.arange(n)
        for it in range(max_sel):
            if remaining.size == 0:
                break
            s1, s2, s3 = _greedy_scores(cfg, C, F, remaining, prior_scale,
                                        cfg.design.rank_tol, cfg.design.kappa)
            cost = np.array([_depth_cost(cfg, lib, int(k), depth_now)
                             for k in remaining])
            if policy == "posterior":
                s1 = np.zeros_like(s1)
                s2 = np.zeros_like(s2)
            g1, g2, g3 = s1 / cost, s2 / cost, s3 / cost
            # 字典序（带相对容差）：先数据支撑，再弱模态，最后后验损失
            tol1 = 1e-9 * max(np.abs(g1).max(), 1e-12)
            tol2 = 1e-9 * max(np.abs(g2).max(), 1e-12)
            m1 = g1 >= (g1.max() - tol1)
            cand1 = np.where(m1)[0]
            m2 = g2[cand1] >= (g2[cand1].max() - tol2)
            cand2 = cand1[m2]
            pick = int(cand2[int(np.argmax(g3[cand2]))])
            k = int(remaining[pick])
            sel.append(k)
            cost_total += cost[pick]
            depth_now[lib.em[k]] = lib.de[k]
            depth_now[lib.rc[k]] = lib.dr[k]
            F += np.outer(C[k], C[k])
            remaining = np.delete(remaining, pick)
            if progress and (it % 10 == 0):
                progress(it, max_sel)
            if cost_total >= budget_cost:
                break

    elif policy == "uniform":
        ep_all = np.unique(lib.ep)
        while len(sel) < max_sel:
            added = False
            for e in ep_all:
                cand = np.setdiff1d(np.where(lib.ep == e)[0], np.array(sel, int))
                if cand.size == 0:
                    continue
                k = int(cand[rng.integers(cand.size)])
                cost_total += _depth_cost(cfg, lib, k, depth_now)
                depth_now[lib.em[k]] = lib.de[k]
                depth_now[lib.rc[k]] = lib.dr[k]
                sel.append(k)
                added = True
                if len(sel) >= max_sel or cost_total >= budget_cost:
                    break
            if not added:
                break

    elif policy == "random":
        perm = rng.permutation(n)
        for k in perm:
            k = int(k)
            cost_total += _depth_cost(cfg, lib, k, depth_now)
            depth_now[lib.em[k]] = lib.de[k]
            depth_now[lib.rc[k]] = lib.dr[k]
            sel.append(k)
            if len(sel) >= max_sel or cost_total >= budget_cost:
                break

    elif policy == "fixed":
        # 保留固定模式：每个历元首个 6 m—6 m 链路
        d0 = cfg.auv.depth_set[0]
        for e in np.unique(lib.ep):
            idx = np.where((lib.ep == e) & (lib.de == d0) & (lib.dr == d0))[0]
            if idx.size:
                sel.append(int(idx[0]))
            if len(sel) >= max_sel:
                break
    else:
        raise ValueError(f"未知策略: {policy}")

    if progress:
        progress(1, 1)
    return np.array(sel, int)


def links_by_epoch(cfg, lib, sel: np.ndarray):
    """把选择结果按历元分组，供界面逐步推进。"""
    out = []
    for k in range(cfg.auv.epochs.size):
        out.append(np.array([s for s in sel if lib.ep[s] == k], int))
    return out
