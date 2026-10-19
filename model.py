"""分层流函数模型、声速剖面、地形与真值场。

流函数（论文式 (12)）::

    psi(p,t) = sum_l B_l(z) [ U_l(t) y - V_l(t) x + sum_q a_lq(t) phi_q(x,y) ]

由此得到的水平流速（论文式 (3)）为 ``f_h = [d_y psi, -d_x psi] = D(p) a(t)``。
``D`` 由 :class:`Dictionary` 给出，``a`` 由 :class:`DictField` 管理。
真值场 :class:`TruthField` 按 ``qhd_eval.m`` 的实测模态构造。
"""

from __future__ import annotations

import numpy as np

try:                                    # 可选：读取 MATLAB v7.3 输入
    import h5py
except Exception:                       # pragma: no cover
    h5py = None


# --------------------------------------------------------------------------- #
# 基函数
# --------------------------------------------------------------------------- #
def bspline_basis(z, L: int, Href: float) -> np.ndarray:
    """端点插值的 3 次 B 样条深度基（等价于 ``bsplineBasis.m``）。

    Parameters
    ----------
    z : array_like
        离底高度 (m)。
    L : int
        基函数个数（>= 4）。
    Href : float
        深度尺度 (m)。

    Returns
    -------
    ndarray, shape (n, L)
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    p = 3
    n = max(int(L), p + 1)
    ni = max(n - p, 2)
    ti = np.arange(ni + 1) / ni
    tt = np.concatenate([np.zeros(p + 1), ti[1:-1], np.ones(p + 1)])
    nt = tt.size
    x = np.clip(z / float(Href), 0.0, 1.0 - 1e-12)

    N = ((x[:, None] >= tt[:-1]) & (x[:, None] < tt[1:])).astype(float)
    for k in range(1, p + 1):
        Nk = np.zeros((x.size, nt - k - 1))
        for i in range(nt - k - 1):
            d1 = tt[i + k] - tt[i]
            d2 = tt[i + k + 1] - tt[i + 1]
            a = (x - tt[i]) / d1 * N[:, i] if d1 > 0 else 0.0
            b = (tt[i + k + 1] - x) / d2 * N[:, i + 1] if d2 > 0 else 0.0
            Nk[:, i] = a + b
        N = Nk
    return N


def rbf_norm(centers: np.ndarray, scales: np.ndarray, Lx: float, Ly: float,
             n: int = 41) -> np.ndarray:
    """高斯 RBF 梯度模的面积均方根（等价于 ``qhd_rbfnorm.m``）。"""
    xg = np.linspace(0.0, Lx, n)
    yg = np.linspace(0.0, Ly, n)
    XG, YG = np.meshgrid(xg, yg)
    out = np.zeros(len(scales))
    for q, s in enumerate(scales):
        dx = XG - centers[q, 0]
        dy = YG - centers[q, 1]
        e = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * s * s))
        gx = -dx / (s * s) * e
        gy = -dy / (s * s) * e
        out[q] = np.sqrt(np.mean(gx ** 2 + gy ** 2))
    return out


def _rbf_terms(x, y, centers, scales, norms):
    """返回 ``(phi, dphi/dx, dphi/dy)``，导数为梯度均方根归一化后的值。"""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    Q = len(scales)
    phi = np.zeros((x.size, Q))
    dx_ = np.zeros((x.size, Q))
    dy_ = np.zeros((x.size, Q))
    for q in range(Q):
        s = scales[q]
        ddx = x - centers[q, 0]
        ddy = y - centers[q, 1]
        e = np.exp(-(ddx ** 2 + ddy ** 2) / (2.0 * s * s))
        phi[:, q] = e
        dx_[:, q] = -ddx / (s * s) * e / norms[q]
        dy_[:, q] = -ddy / (s * s) * e / norms[q]
    return phi, dx_, dy_


# --------------------------------------------------------------------------- #
# 反演字典
# --------------------------------------------------------------------------- #
class Dictionary:
    """固定深度基 + 水平 RBF，给出 ``f_h = D(p) a``。"""

    def __init__(self, L: int, Q: int, Href: float, centers, scales,
                 Lx: float, Ly: float):
        self.L = int(L)
        self.Q = int(Q)
        self.Href = float(Href)
        self.centers = np.asarray(centers, dtype=float)
        self.scales = np.asarray(scales, dtype=float).ravel()
        self.norm = rbf_norm(self.centers, self.scales, Lx, Ly)
        self.n_coef = self.L * (self.Q + 2)
        # 系数排列：[U_1 V_1 a_11..a_1Q | U_2 V_2 a_21.. | ...]
        self.idx_U = np.array([l * (Q + 2) for l in range(self.L)])
        self.idx_V = self.idx_U + 1
        self.idx_q = np.array([[l * (Q + 2) + 2 + q for q in range(Q)]
                               for l in range(self.L)])
        self.x0 = Lx / 2.0
        self.y0 = Ly / 2.0

    def velocity_basis(self, x, y, z, H):
        """返回 ``(Du, Dv)``，形状 ``(n, n_coef)``，满足 ``u = Du @ a``。"""
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        H = np.asarray(H, dtype=float).ravel()
        z = np.clip(np.minimum(z, H), 0.0, None)
        B = bspline_basis(z, self.L, self.Href)                  # (n, L)
        _, dx_, dy_ = _rbf_terms(x, y, self.centers, self.scales, self.norm)
        n = x.size
        Du = np.zeros((n, self.n_coef))
        Dv = np.zeros((n, self.n_coef))
        for l in range(self.L):
            Bl = B[:, l]
            Du[:, self.idx_U[l]] = Bl
            Dv[:, self.idx_V[l]] = Bl
            q = self.idx_q[l]
            Du[:, q] = Bl[:, None] * dy_             #  d psi / d y
            Dv[:, q] = Bl[:, None] * (-dx_)          # -d psi / d x
        return Du, Dv

    def dmat(self, x, y, z, H):
        """单点 ``D``，形状 ``(2, n_coef)``。"""
        Du, Dv = self.velocity_basis([x], [y], [z], [H])
        return np.vstack([Du[0], Dv[0]])

    def map_operator(self, depth_panels, Hfun, Lx: float, Ly: float,
                     n: int = 21, edg: float = 0.3):
        """报图算子 ``G``（``2*nz x d``，按行尺度归一化）。"""
        xg = np.linspace(0.0, Lx, n + 2)[1:-1]
        yg = np.linspace(0.0, Ly, n + 2)[1:-1]
        XG, YG = np.meshgrid(xg, yg)
        nz = XG.size * len(depth_panels)
        G = np.zeros((2 * nz, self.n_coef))
        row = 0
        for zd in depth_panels:
            HG = Hfun(XG.ravel(), YG.ravel())
            zz = np.minimum(float(zd), HG - edg)
            Du, Dv = self.velocity_basis(XG.ravel(), YG.ravel(), zz, HG)
            G[row:row + XG.size, :] = Du
            G[nz + row:nz + row + XG.size, :] = Dv
            row += XG.size
        sc = np.sqrt(np.mean(np.sum(G ** 2, axis=1)))
        return G / max(sc, 1e-300)

    def prior_matrix(self, G, field_sigma: float, rate_field_sigma: float,
                     floor_rel: float = 1e-4):
        """场空间对角先验精度矩阵（论文 §Correlated Field Prior）。"""
        d = self.n_coef
        en = np.sum(G ** 2, axis=0)
        en = np.maximum(en, floor_rel * en.max())
        Pr = np.zeros((2 * d, 2 * d))
        Pr[np.arange(d), np.arange(d)] = en / field_sigma ** 2
        Pr[np.arange(d, 2 * d), np.arange(d, 2 * d)] = en / rate_field_sigma ** 2
        return Pr


class DictField:
    """由系数 ``(a0, adot)`` 描述的估计场：``a(t) = a0 + t * adot``。"""

    def __init__(self, dictionary: Dictionary, a0=None, adot=None, t0: float = 0.0):
        d = dictionary.n_coef
        self.D = dictionary
        self.a0 = np.zeros(d) if a0 is None else np.asarray(a0, dtype=float)
        self.adot = np.zeros(d) if adot is None else np.asarray(adot, dtype=float)
        self.t0 = float(t0)

    def coefficients(self, t_s: float) -> np.ndarray:
        return self.a0 + float(t_s) * self.adot

    def velocity(self, x, y, z, H, t_s: float = 0.0):
        Du, Dv = self.D.velocity_basis(x, y, z, H)
        a = self.coefficients(t_s)
        return Du @ a, Dv @ a


# --------------------------------------------------------------------------- #
# 声速 / 地形
# --------------------------------------------------------------------------- #
class SoundSpeed:
    """一维声速剖面（按深度定义，可换算到离底高度）。"""

    def __init__(self, z_depth, c, Href: float = 22.0):
        self.z_depth = np.asarray(z_depth, dtype=float).ravel()
        self.c = np.asarray(c, dtype=float).ravel()
        self.Href = float(Href)
        order = np.argsort(self.z_depth)
        self.z_depth = self.z_depth[order]
        self.c = self.c[order]

    def at_depth(self, depth):
        return np.interp(depth, self.z_depth, self.c)

    def at_height(self, height, column_depth: float):
        """离底高度 (m) → 声速 (m/s)。"""
        return self.at_depth(column_depth - np.asarray(height, dtype=float))

    @staticmethod
    def synthetic_reference():
        """合成夏季剖面（缺少实测输入时的后备方案）。"""
        depth = np.arange(0.0, 22.0001, 0.05)
        c = 1521.5 + 0.75 * depth - 0.055 * depth ** 2
        return SoundSpeed(depth, c)


class Bathymetry:
    """规则网格水深（正值为水深，单位 m）。"""

    def __init__(self, x, y, H):
        from scipy.interpolate import RegularGridInterpolator
        self.x = np.asarray(x, dtype=float).ravel()
        self.y = np.asarray(y, dtype=float).ravel()
        H = np.asarray(H, dtype=float)
        if H.shape == (self.x.size, self.y.size):
            H = H.T
        self.H = H
        self._f = RegularGridInterpolator((self.y, self.x), self.H,
                                          bounds_error=False, fill_value=None)

    def __call__(self, x, y):
        xx = np.clip(np.asarray(x, dtype=float).ravel(), self.x[0], self.x[-1])
        yy = np.clip(np.asarray(y, dtype=float).ravel(), self.y[0], self.y[-1])
        return self._f(np.column_stack([yy, xx]))

    @staticmethod
    def synthetic_reference(Lx=10000.0, Ly=10000.0, n=201):
        x = np.linspace(0.0, Lx, n)
        y = np.linspace(0.0, Ly, n)
        XG, YG = np.meshgrid(x, y)
        H = 21.2 + 0.55 * np.sin(2 * np.pi * XG / Lx) + 0.35 * np.cos(2 * np.pi * YG / Ly)
        return Bathymetry(x, y, H)


# --------------------------------------------------------------------------- #
# 真值场（实测模态 + 平流涡旋）
# --------------------------------------------------------------------------- #
class TruthField:
    """按 ``qhd_eval.m`` 的 ``truth`` 分支构造的参考场。

    ``W = sum_k mode_k(z) * ( c_k(tau) - gamma_k S_site + gamma_k eddy )``，
    其中 ``u = Re(W)``、``v = Im(W)``，涡旋结构随平均流 ``adv`` 平流。
    """

    def __init__(self, *, z_cell, mode, z_fine, mode_fine, A0, Adot,
                 centers, scales, amp, norms, gamma, adv, site, Hfun):
        self.z_cell = np.asarray(z_cell, dtype=float).ravel()
        mode = np.atleast_2d(np.asarray(mode, dtype=float))
        self.mode = mode if mode.shape[0] == self.z_cell.size else mode.T
        self.z_fine = np.asarray(z_fine, dtype=float).ravel()
        mode_fine = np.atleast_2d(np.asarray(mode_fine, dtype=float))
        self.mode_fine = (mode_fine if mode_fine.shape[0] == self.z_fine.size
                          else mode_fine.T)
        self.A0 = np.asarray(A0, dtype=complex).ravel()
        self.Adot = np.asarray(Adot, dtype=complex).ravel()
        self.centers = np.asarray(centers, dtype=float)
        self.scales = np.asarray(scales, dtype=float).ravel()
        amp = np.atleast_2d(np.asarray(amp, dtype=float))
        self.amp = amp if amp.shape[0] == self.A0.size else amp.T
        self.norms = np.asarray(norms, dtype=float).ravel()
        self.gamma = np.asarray(gamma, dtype=float).ravel()
        self.adv = np.asarray(adv, dtype=float).ravel()
        self.site = np.asarray(site, dtype=float).ravel()
        self.Hfun = Hfun
        self.K = self.A0.size
        self.n_rbf = self.centers.shape[0]

    def _grad(self, q, x, y):
        """复梯度 ``g = dphi/dy - i dphi/dx``。"""
        s = self.scales[q]
        dx = np.asarray(x, dtype=float).ravel() - self.centers[q, 0]
        dy = np.asarray(y, dtype=float).ravel() - self.centers[q, 1]
        e = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * s * s))
        return (-dy / (s * s) * e) - 1j * (-dx / (s * s) * e)

    def _eddy(self, x, y):
        """返回 ``(K, n)`` 的涡旋贡献。"""
        out = np.zeros((self.K, np.size(x)), dtype=complex)
        for q in range(self.n_rbf):
            g = self._grad(q, x, y) / self.norms[q]
            out += self.amp[:, q][:, None] * g[None, :]
        return out

    def velocity(self, x, y, z_height, tau_s=0.0):
        """参考场流速；``tau_s`` 为相对参考时刻的秒数。"""
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        z = np.asarray(z_height, dtype=float).ravel()
        H = self.Hfun(x, y)
        z = np.clip(np.minimum(z, H), 0.0, None)
        tau = float(tau_s)

        xs = np.array([self.site[0] - self.adv[0] * tau])
        ys = np.array([self.site[1] - self.adv[1] * tau])
        S_site = self._eddy(xs, ys)[:, 0]                     # (K,)

        M = np.vstack([np.interp(z, self.z_fine, self.mode_fine[:, k])
                       for k in range(self.K)])               # (K, n)
        ck = self.A0 + tau * self.Adot                         # (K,)
        eddy = self._eddy(x - self.adv[0] * tau, y - self.adv[1] * tau)
        W = np.zeros(x.size, dtype=complex)
        for k in range(self.K):
            W += M[k] * (ck[k] - self.gamma[k] * S_site[k] + self.gamma[k] * eddy[k])
        return W.real, W.imag


# --------------------------------------------------------------------------- #
# 输入装载与场景装配
# --------------------------------------------------------------------------- #
def _mat_array(node, name):
    a = np.array(node[name])
    if a.dtype.names:                    # MATLAB 复数：compound(real, imag)
        a = a["real"] + 1j * a["imag"]
    return a


def load_matlab_inputs(path: str):
    """读取 ``qhd_inputs.mat``（MATLAB v7.3）；不可用时返回 ``None``。"""
    if h5py is None:
        return None
    try:
        f = h5py.File(path, "r")
    except Exception:
        return None
    try:
        ssp, bathy, adcp, tr = f["ssp"], f["bathy"], f["adcp"], f["truth"]
        return {
            "ssp_z": _mat_array(ssp, "z").ravel(),
            "ssp_c": _mat_array(ssp, "c").ravel(),
            "H": _mat_array(bathy, "H"),                   # (ny, nx)
            "H_x": _mat_array(bathy, "x").ravel(),
            "H_y": _mat_array(bathy, "y").ravel(),
            "z_cell": _mat_array(adcp, "zCell").ravel(),
            "mode": _mat_array(adcp, "mode"),              # (19, K)
            "z_fine": _mat_array(adcp, "zFine").ravel(),
            "mode_fine": _mat_array(adcp, "modeFine"),     # (89, K)
            "A0": _mat_array(adcp, "A0").ravel(),
            "Adot": _mat_array(adcp, "Adot").ravel(),
            "t0": float(_mat_array(adcp, "t0").ravel()[0]),
            "Href": float(_mat_array(tr, "Href").ravel()[0]),
            "t_ctr": _mat_array(tr, "ctr"),                # (8, 2)
            "t_scl": _mat_array(tr, "scl").ravel(),
            "t_amp": _mat_array(tr, "Amp"),                # (K, 8)
            "t_norm": _mat_array(tr, "rbfNorm").ravel(),
            "t_gamma": _mat_array(tr, "gamma").ravel(),
            "t_adv": _mat_array(tr, "adv").ravel(),
            "t_umean": float(_mat_array(tr, "umean").ravel()[0]),
            "t_vmean": float(_mat_array(tr, "vmean").ravel()[0]),
        }
    finally:
        f.close()


def _synthetic_truth(cfg, bathy) -> TruthField:
    """缺少实测输入时的合成参考场（结构与实测版本一致）。"""
    rng = np.random.default_rng(cfg.seed)
    z_cell = np.arange(3.23, 21.24, 1.0)
    Href = float(cfg.dictionary.Href)
    z_fine = np.arange(0.0, Href + 1e-9, 0.25)
    B = bspline_basis(z_cell, cfg.truth.L, Href)
    Bf = bspline_basis(z_fine, cfg.truth.L, Href)
    coef = np.array([[1.0, 0.6, -0.4, 0.3, -0.2, 0.1, -0.1, 0.05],
                     [0.2, -0.8, 0.9, -0.5, 0.3, -0.2, 0.1, -0.05],
                     [0.1, 0.3, -0.7, 0.8, -0.4, 0.2, -0.1, 0.05]])
    mode_fine = Bf @ coef.T
    mode_fine /= np.linalg.norm(mode_fine, axis=0)
    mode_cell = B @ coef.T
    mode_cell /= np.linalg.norm(mode_cell, axis=0)
    A0 = np.array([1.6 + 0.55j, 0.30 + 0.28j, 0.10 - 0.12j])
    Adot = np.array([6e-6 - 1.6e-5j, -2e-6 + 1.4e-5j, 7e-6 - 2e-6j])
    centers = np.vstack([cfg.dictionary.centers,
                         np.array([[4200.0, 4600.0], [1600.0, 5400.0],
                                   [8600.0, 5200.0]])])
    scales = np.concatenate([cfg.dictionary.scales, [1500.0, 4500.0, 2500.0]])
    amp = np.cos(2 * np.pi * rng.random((3, centers.shape[0])))
    norms = rbf_norm(centers, scales, cfg.site.Lx, cfg.site.Ly)
    return TruthField(z_cell=z_cell, mode=mode_cell, z_fine=z_fine,
                      mode_fine=mode_fine, A0=A0, Adot=Adot, centers=centers,
                      scales=scales, amp=amp, norms=norms,
                      gamma=np.array([0.35, 0.35, 0.35]),
                      adv=np.array([0.21, 0.08]),
                      site=(cfg.site.Lx / 2, cfg.site.Ly / 2), Hfun=bathy)


def build_environment(cfg, inputs=None):
    """返回 ``(ssp, bathy, truth)``。"""
    if inputs is None:
        ssp = SoundSpeed.synthetic_reference()
        bathy = Bathymetry.synthetic_reference(cfg.site.Lx, cfg.site.Ly, cfg.site.nx)
        return ssp, bathy, _synthetic_truth(cfg, bathy)
    ssp = SoundSpeed(inputs["ssp_z"], inputs["ssp_c"], inputs["Href"])
    bathy = Bathymetry(inputs["H_x"], inputs["H_y"], inputs["H"])
    truth = TruthField(
        z_cell=inputs["z_cell"], mode=inputs["mode"],
        z_fine=inputs["z_fine"], mode_fine=inputs["mode_fine"],
        A0=inputs["A0"], Adot=inputs["Adot"],
        centers=inputs["t_ctr"], scales=inputs["t_scl"], amp=inputs["t_amp"],
        norms=inputs["t_norm"], gamma=inputs["t_gamma"], adv=inputs["t_adv"],
        site=(cfg.site.Lx / 2, cfg.site.Ly / 2), Hfun=bathy)
    return ssp, bathy, truth


# --------------------------------------------------------------------------- #
# 平台运动
# --------------------------------------------------------------------------- #
_AUV_P0 = np.array([[1500.0, 2000.0], [8000.0, 1500.0], [8500.0, 8000.0],
                    [2000.0, 8000.0], [5000.0, 1500.0], [5000.0, 8500.0],
                    [1500.0, 5000.0]])
_AUV_HDG = np.deg2rad([20.0, 130.0, 200.0, 320.0, 90.0, 270.0, 0.0])


def platform_positions(cfg, t: float):
    """返回 ``(x, y, is_auv)``：前 ``N`` 个为 AUV，其余为浮标。"""
    x = _AUV_P0[:, 0] + cfg.auv.speed * np.cos(_AUV_HDG) * float(t)
    y = _AUV_P0[:, 1] + cfg.auv.speed * np.sin(_AUV_HDG) * float(t)
    x = np.concatenate([x, cfg.buoy.xy[:, 0]])
    y = np.concatenate([y, cfg.buoy.xy[:, 1]])
    is_auv = np.concatenate([np.ones(cfg.auv.N, bool),
                             np.zeros(cfg.buoy.xy.shape[0], bool)])
    return x, y, is_auv


def auv_track(cfg, i: int, n: int = 60):
    """第 ``i`` 个 AUV 的整段航迹（用于界面绘制）。"""
    tt = np.linspace(0.0, float(cfg.auv.epochs.max()), n)
    hdg = _AUV_HDG[i]
    return (_AUV_P0[i, 0] + cfg.auv.speed * np.cos(hdg) * tt,
            _AUV_P0[i, 1] + cfg.auv.speed * np.sin(hdg) * tt)


def enu_to_lonlat(cfg, x, y):
    """局地 ENU (m) → (lon, lat)。"""
    mx = 111320.0 * np.cos(np.deg2rad(cfg.site.lat))
    my = 110574.0
    lon = cfg.site.lon + (np.asarray(x, dtype=float) - cfg.site.Lx / 2) / mx
    lat = cfg.site.lat + (np.asarray(y, dtype=float) - cfg.site.Ly / 2) / my
    return lon, lat
