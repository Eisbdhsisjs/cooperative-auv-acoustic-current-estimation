"""移动端点声学模型：射线走时、固定射线灵敏度与候选链路库。

论文式 (13) 的线性化形式为::

    tau_m ~= tau_m^0 - int_{Gamma_m^0} ( e_m . f_h ) / c_0(z)^2 ds

本模块按 MATLAB 版 ``qhd_raycycle.m`` 求解"表面—底床多次反射折射射线"
的参考路径（水平缓变波导中 p = cos(theta)/c 为不变量），得到
``tau0``、沿程 ``(r, z, theta)``、以及灵敏度矩阵 ``A``。
"""

from __future__ import annotations

import os
import hashlib

import numpy as np

from .model import bspline_basis, _rbf_terms

#: NumPy 2.0 起 ``trapz`` 更名为 ``trapezoid``，这里做兼容。
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
# 射线基本算子
# --------------------------------------------------------------------------- #
def _cumtrapz_rows(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """沿最后一个轴做累计梯形积分（首列为 0）。"""
    d = np.diff(x)
    inc = 0.5 * (y[:, 1:] + y[:, :-1]) * d[None, :]
    return np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(inc, axis=1)], axis=1)


def _interp_rows(xq: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """逐行线性插值：``xp`` 每行单调递增，``xq`` 每行一个查询点。"""
    m, n = xp.shape
    idx = np.clip(np.sum(xp < xq[:, None], axis=1), 1, n - 1)
    rows = np.arange(m)
    x0 = xp[rows, idx - 1]
    x1 = xp[rows, idx]
    y0 = fp[rows, idx - 1]
    y1 = fp[rows, idx]
    w = (xq - x0) / np.where(x1 > x0, x1 - x0, 1.0)
    return y0 + w * (y1 - y0)


def _ray_tables(p: np.ndarray, zg: np.ndarray, cg: np.ndarray):
    """给定水平慢度 ``p``（可批量），返回累计水平距离/走时表。"""
    q = np.sqrt(np.maximum(1.0 / cg ** 2 - p[:, None] ** 2, 1e-14))
    IR = _cumtrapz_rows(p[:, None] / q, zg)
    IT = _cumtrapz_rows(1.0 / (cg ** 2 * q), zg)
    return q, IR, IT


def _propagate_many(p, zg, cg, z1, R, max_legs=400):
    """批量求解固定 ``p`` 时到达水平距离 ``R`` 的高度与走时。"""
    q, IR, IT = _ray_tables(p, zg, cg)
    m = p.size
    IR1 = _interp_rows(np.full(m, z1), zg[None, :].repeat(m, 0), IR)
    IT1 = _interp_rows(np.full(m, z1), zg[None, :].repeat(m, 0), IT)
    IRend, ITend = IR[:, -1], IT[:, -1]

    t_ = R - (IRend - IR1)
    k = np.zeros(m, dtype=int)
    rem = t_.copy()
    past = t_ >= 0
    if np.any(past):
        kk = np.floor(t_[past] / IRend[past]).astype(int)
        k[past] = kk
        rem[past] = t_[past] - kk * IRend[past]
    up = np.zeros(m, dtype=bool)
    up[past] = (k[past] % 2) == 1
    IRt = np.where(~past, IR1 + R, np.where(up, rem, IRend - rem))
    IRt = np.where(past, np.clip(IRt, 0.0, IRend), IRt)

    z_end = _interp_rows(IRt, IR, np.repeat(zg[None, :], m, 0))
    ITt = _interp_rows(IRt, IR, IT)
    tau = np.where(~past, ITt - IT1,
                   (ITend - IT1) + k * ITend + np.where(up, ITt, ITend - ITt))
    bad = (k > max_legs) | ~np.isfinite(z_end) | ~np.isfinite(tau)
    z_end[bad] = np.nan
    tau[bad] = np.inf
    return z_end, tau


def _sample_path(p: float, zg, cg, z1: float, R: float, n_step: int):
    """在 ``r in [0, R]`` 上采样射线高度与倾角。"""
    p_arr = np.array([p])
    q, IR, IT = _ray_tables(p_arr, zg, cg)
    IR = IR[0]
    q = q[0]
    IR1 = float(np.interp(z1, zg, IR))
    IRend = IR[-1]
    r = np.linspace(0.0, R, n_step + 1)
    t_ = r - (IRend - IR1)
    past = t_ >= 0
    k = np.zeros(r.size, dtype=int)
    rem = t_.copy()
    k[past] = np.floor(t_[past] / IRend).astype(int)
    rem[past] = t_[past] - k[past] * IRend
    up = np.zeros(r.size, dtype=bool)
    up[past] = (k[past] % 2) == 1
    IRt = np.where(~past, IR1 + r, np.where(up, rem, IRend - rem))
    IRt = np.clip(IRt, 0.0, IRend)
    z = np.interp(IRt, IR, zg)
    qz = np.interp(z, zg, q)
    sign = np.where(~past, 1.0, np.where(up, 1.0, -1.0))
    theta = np.arctan2(sign * qz, p)
    return r, z, theta


def ray_cycle(ssp, Hfun, x1, y1, d1, x2, y2, d2, cfg,
              n_step=None, n_guess=None, Href: float = 22.0):
    """两点射线求解（首个到达分支）。

    ``d1``/``d2`` 为换能器深度（水面下，m）。返回 ``dict`` 或 ``None``。
    """
    ac = cfg.ac
    n_step = int(n_step or ac.n_step)
    n_guess = int(n_guess or ac.n_guess)

    dx, dy = float(x2 - x1), float(y2 - y1)
    R = float(np.hypot(dx, dy))
    if R < 1.0:
        return None
    e = np.array([dx, dy]) / R

    r = np.linspace(0.0, R, n_step + 1)
    Hv = Hfun(x1 + e[0] * r, y1 + e[1] * r)
    Hm = float(np.mean(Hv))
    z1, z2 = Hm - float(d1), Hm - float(d2)
    if not (0.0 < z1 < Hm and 0.0 < z2 < Hm):
        return None

    zg = np.linspace(0.0, Hm, 401)
    cg = ssp.at_height(zg, Hm)
    p_max = 1.0 / float(np.max(cg))
    c1 = float(ssp.at_height(z1, Hm))

    th0 = np.linspace(-ac.theta_max, ac.theta_max, n_guess)
    p_all = np.cos(th0) / c1
    ok = (p_all > 0.0) & (p_all <= p_max * (1.0 + 5e-7))
    if not np.any(ok):
        return None
    p = p_all[ok]
    z_end, tau = _propagate_many(p, zg, cg, z1, R, ac.max_legs)

    f = z_end - z2
    good = np.isfinite(f)
    br = np.where(good[:-1] & good[1:] & (f[:-1] * f[1:] <= 0.0))[0]
    if br.size == 0:
        # 允许"擦边"分支：取最接近接收深度的符号一致区间
        if not np.any(good):
            return None
        i = int(np.nanargmin(np.abs(f)))
        if abs(f[i]) > 0.5 * Hm:
            return None
        br = np.array([max(i - 1, 0)])
    tm = np.minimum(tau[br], tau[br + 1])
    ib = int(br[int(np.argmin(tm))])

    A_, B_ = th0[ok][ib], th0[ok][ib + 1]
    pA = np.cos(A_) / c1
    pB = np.cos(B_) / c1
    Fa = _propagate_many(np.array([pA]), zg, cg, z1, R, ac.max_legs)[0][0] - z2
    Fb = _propagate_many(np.array([pB]), zg, cg, z1, R, ac.max_legs)[0][0] - z2
    if not (np.isfinite(Fa) and np.isfinite(Fb)):
        return None
    for _ in range(24):
        Mv = 0.5 * (A_ + B_)
        pM = np.cos(Mv) / c1
        ze = _propagate_many(np.array([pM]), zg, cg, z1, R, ac.max_legs)[0][0]
        Fm = ze - z2
        if not np.isfinite(Fm):
            break
        if Fa * Fm <= 0.0:
            B_, Fb = Mv, Fm
        else:
            A_, Fa = Mv, Fm
        if abs(Fm) < 1e-7:
            break
    if abs(0.5 * (Fa + Fb)) > 5e-3:
        return None
    th = 0.5 * (A_ + B_)
    p_sel = np.cos(th) / c1

    r, z, theta = _sample_path(p_sel, zg, cg, z1, R, n_step)
    tau0 = _propagate_many(np.array([p_sel]), zg, cg, z1, R, ac.max_legs)[1][0]
    if not np.isfinite(tau0):
        return None
    return {
        "ok": True, "r": r, "z": z, "theta": theta,
        "c": ssp.at_height(z, Hm), "H": np.full(r.size, Hm),
        "e": e, "x1": float(x1), "y1": float(y1),
        "x2": float(x2), "y2": float(y2), "R": R,
        "tau0": float(tau0), "theta0": float(th), "Hm": Hm,
    }


def travel_time(truth, ray, t_s: float) -> float:
    """含流的走时 ``tau = int ds / ((c + u_e cos th) cos th)``（真值场）。"""
    r = ray["r"]
    z = ray["z"]
    th = ray["theta"]
    e = ray["e"]
    x = ray["x1"] + e[0] * r
    y = ray["y1"] + e[1] * r
    Hfun = getattr(truth, "Hfun", None)
    H = Hfun(x, y) if Hfun is not None else ray["H"]
    c = ray["c"]
    u, v = truth.velocity(x, y, z, t_s)
    ue = u * e[0] + v * e[1]
    return float(_trapz(1.0 / ((c + ue * np.cos(th)) * np.cos(th)), r))


def sensitivity(dictionary, ray, Hfun) -> np.ndarray:
    """固定参考射线的走时灵敏度 ``d tau / d a``（论文式 (13)）。"""
    r = ray["r"]
    z = ray["z"]
    e = ray["e"]
    c = ray["c"]
    x = ray["x1"] + e[0] * r
    y = ray["y1"] + e[1] * r
    H = Hfun(x, y)
    zz = np.clip(np.minimum(z, H), 0.0, None)
    B = bspline_basis(zz, dictionary.L, dictionary.Href)
    _, dx_, dy_ = _rbf_terms(x, y, dictionary.centers, dictionary.scales,
                             dictionary.norm)
    n, d = x.size, dictionary.n_coef
    S = np.zeros((n, d))
    for l in range(dictionary.L):
        Bl = B[:, l]
        S[:, dictionary.idx_U[l]] = Bl * e[0]
        S[:, dictionary.idx_V[l]] = Bl * e[1]
        S[:, dictionary.idx_q[l]] = Bl[:, None] * (e[0] * dy_ + e[1] * (-dx_))
    w = 1.0 / np.maximum(c, 1.0) ** 2
    return -_trapz(S * w[:, None], r, axis=0)


# --------------------------------------------------------------------------- #
# 候选链路库
# --------------------------------------------------------------------------- #
class LinkLibrary:
    """所有候选发射块（发射平台、接收平台、历元、两个深度）。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.fingerprint = ""
        self.n_candidates = 0
        self.em = np.zeros(0, int)
        self.rc = np.zeros(0, int)
        self.ep = np.zeros(0, int)
        self.t = np.zeros(0)
        self.de = np.zeros(0)
        self.dr = np.zeros(0)
        self.range = np.zeros(0)
        self.tau0 = np.zeros(0)
        self.feasible = np.zeros(0, bool)
        self.A = np.zeros((0, 0))
        self.path_r = self.path_z = self.path_th = np.zeros((0, 0))
        self.Hm = np.zeros(0)
        self.x1 = np.zeros(0)
        self.y1 = np.zeros(0)
        self.x2 = np.zeros(0)
        self.y2 = np.zeros(0)
        self.e = np.zeros((0, 2))
        self.n_coef = 0

    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return int(self.em.size)

    def epoch_indices(self) -> np.ndarray:
        """每个历元对应的链路序号。"""
        return [np.where(self.ep == k)[0] for k in range(self.cfg.auv.epochs.size)]

    def ray(self, k: int, ssp) -> dict:
        """重建第 ``k`` 条链路的射线字典（供真值正演与绘图使用）。"""
        z = self.path_z[k]
        return {
            "r": self.path_r[k], "z": z, "theta": self.path_th[k],
            "c": ssp.at_height(z, self.Hm[k]),
            "H": np.full(z.size, self.Hm[k]),
            "e": self.e[k], "x1": self.x1[k], "y1": self.y1[k],
            "x2": self.x2[k], "y2": self.y2[k], "R": self.range[k],
            "tau0": self.tau0[k], "Hm": self.Hm[k],
        }


def build_library(cfg, ssp, Hfun, dictionary, progress=None,
                  n_step=None, n_guess=None) -> LinkLibrary:
    """遍历历元 × 平台对 × 换能器深度组合，建立候选链路库。"""
    from .model import platform_positions

    lib = LinkLibrary(cfg)
    lib.fingerprint = library_fingerprint(cfg, ssp, Hfun, dictionary)
    n_auv = cfg.auv.N
    n_plat = n_auv + cfg.buoy.xy.shape[0]
    depth_sets = [cfg.auv.depth_set] * n_auv + [[cfg.buoy.depth]] * cfg.buoy.xy.shape[0]

    # 平台几何：每个历元只算一次
    geom = []
    for k, te in enumerate(cfg.auv.epochs):
        x, y, is_auv = platform_positions(cfg, float(te))
        geom.append((float(te), x, y, is_auv))

    cand = []
    for k, (te, x, y, is_auv) in enumerate(geom):
        for i in range(n_plat):
            for j in range(n_plat):
                if i == j:
                    continue
                if not is_auv[i] and not is_auv[j]:
                    continue                       # 浮标之间不发射
                R = float(np.hypot(x[j] - x[i], y[j] - y[i]))
                if R < cfg.ac.range_min or R > cfg.ac.range_max:
                    continue
                for de in depth_sets[i]:
                    for dr in depth_sets[j]:
                        cand.append((k, i, j, float(de), float(dr), R, te,
                                     x[i], y[i], x[j], y[j]))

    total = len(cand)
    em, rc, ep, de, dr, rng, tt = [], [], [], [], [], [], []
    tau0, feas, paths = [], [], []
    Hm_l, x1_l, y1_l, x2_l, y2_l, e_l = [], [], [], [], [], []
    A_cols = []
    for idx, (k, i, j, d_e, d_r, R, te, x1, y1, x2, y2) in enumerate(cand):
        ray = ray_cycle(ssp, Hfun, x1, y1, d_e, x2, y2, d_r, cfg,
                        n_step=n_step, n_guess=n_guess)
        if ray is None:
            if progress and (idx % 50 == 0):
                progress(idx, total)
            continue
        em.append(i); rc.append(j); ep.append(k); de.append(d_e); dr.append(d_r)
        rng.append(R); tt.append(te); tau0.append(ray["tau0"])
        feas.append(True)
        paths.append((ray["r"], ray["z"], ray["theta"]))
        Hm_l.append(ray["Hm"]); e_l.append(ray["e"])
        x1_l.append(x1); y1_l.append(y1); x2_l.append(x2); y2_l.append(y2)
        # 灵敏度：上块为参考时刻的 d tau / d a0，下块为 d tau / d adot = t * A0
        A0 = sensitivity(dictionary, ray, Hfun)
        A_cols.append(np.concatenate([A0, A0 * float(te)]))
        if progress and (idx % 50 == 0):
            progress(idx, total)
    if progress:
        progress(total, total)

    lib.n_candidates = total
    lib.em = np.array(em, int)
    lib.rc = np.array(rc, int)
    lib.ep = np.array(ep, int)
    lib.de = np.array(de, float)
    lib.dr = np.array(dr, float)
    lib.range = np.array(rng, float)
    lib.t = np.array(tt, float)
    lib.tau0 = np.array(tau0, float)
    lib.feasible = np.array(feas, bool)
    lib.n_coef = dictionary.n_coef
    lib.Hm = np.array(Hm_l, float)
    lib.x1 = np.array(x1_l, float)
    lib.y1 = np.array(y1_l, float)
    lib.x2 = np.array(x2_l, float)
    lib.y2 = np.array(y2_l, float)
    lib.e = np.array(e_l, float).reshape(-1, 2)
    if A_cols:
        lib.A = np.column_stack(A_cols)                  # (d, n)
    if paths:
        lib.path_r = np.vstack([p[0][None, :] for p in paths])
        lib.path_z = np.vstack([p[1][None, :] for p in paths])
        lib.path_th = np.vstack([p[2][None, :] for p in paths])
    return lib


def save_library(lib: LinkLibrary, path: str):
    np.savez_compressed(
        path, em=lib.em, rc=lib.rc, ep=lib.ep, t=lib.t, de=lib.de, dr=lib.dr,
        range=lib.range, tau0=lib.tau0, feasible=lib.feasible, A=lib.A,
        path_r=lib.path_r, path_z=lib.path_z, path_th=lib.path_th,
        Hm=lib.Hm, x1=lib.x1, y1=lib.y1, x2=lib.x2, y2=lib.y2, e=lib.e,
        n_coef=lib.n_coef, n_candidates=lib.n_candidates,
        fingerprint=np.array(lib.fingerprint))


def load_library(cfg, path: str) -> LinkLibrary:
    z = np.load(path)
    lib = LinkLibrary(cfg)
    for k in ("em", "rc", "ep", "t", "de", "dr", "range", "tau0",
              "feasible", "A", "path_r", "path_z", "path_th",
              "Hm", "x1", "y1", "x2", "y2", "e"):
        setattr(lib, k, z[k])
    lib.n_coef = int(z["n_coef"])
    lib.n_candidates = int(z["n_candidates"])
    lib.fingerprint = str(z["fingerprint"]) if "fingerprint" in z else ""
    return lib


def library_fingerprint(cfg, ssp, Hfun, dictionary) -> str:
    """链路库的全部输入参数指纹（字典、声速、地形、平台、射线设置）。

    缓存键必须覆盖所有影响 ``A``/``tau0``/路径的参数，否则改了字典或环境后
    会误用旧库，导致反演精度悄悄变差。
    """
    h = hashlib.sha1()

    def add(arr):
        h.update(np.ascontiguousarray(np.asarray(arr, dtype=float)).tobytes())

    add([cfg.dictionary.L, cfg.dictionary.Q, cfg.dictionary.Href])
    add(dictionary.centers)
    add(dictionary.scales)
    add(dictionary.norm)
    add(ssp.z_depth)
    add(ssp.c)
    add(getattr(Hfun, "x", [0.0]))
    add(getattr(Hfun, "y", [0.0]))
    add(getattr(Hfun, "H", [0.0]))
    add([cfg.ac.n_step, cfg.ac.n_guess, cfg.ac.theta_max, cfg.ac.range_min,
         cfg.ac.range_max, cfg.ac.max_legs, cfg.auv.speed, cfg.buoy.depth])
    add(cfg.auv.epochs)
    add(cfg.auv.depth_set)
    add(cfg.buoy.xy)
    return h.hexdigest()[:16]


def library_cache_path(cfg, cache_dir: str, fingerprint: str = "") -> str:
    tag = (f"L{cfg.dictionary.L}Q{cfg.dictionary.Q}n{cfg.ac.n_step}"
           f"g{cfg.ac.n_guess}r{int(cfg.ac.range_min)}-{int(cfg.ac.range_max)}"
           f"{'_' + fingerprint if fingerprint else ''}")
    return os.path.join(cache_dir, f"link_library_{tag}.npz")
