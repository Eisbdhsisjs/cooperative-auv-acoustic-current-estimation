"""总控：环境装配 → 链路库 → 采样选择 → 走时正演 → 反演 → 误差评估。

界面层只与本模块打交道（:class:`Survey`）。
"""

from __future__ import annotations

import os

import numpy as np

from . import acoustics
from .config import default_config
from .inverse import errors, forward_truth, invert
from .model import (Bathymetry, Dictionary, DictField, SoundSpeed,
                    build_environment, enu_to_lonlat, load_matlab_inputs,
                    platform_positions)
from .planner import links_by_epoch, select_links


class Survey:
    """一次协同声学调查的完整状态。"""

    def __init__(self, cfg=None, cache_dir: str | None = None,
                 use_matlab: bool = True, progress=None, rebuild: bool = False):
        self.cfg = cfg or default_config()
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        inputs = None
        if use_matlab:
            inputs = load_matlab_inputs(self.cfg.inputs_mat)
        self.using_matlab_inputs = inputs is not None
        self.ssp, self.bathy, self.truth = build_environment(self.cfg, inputs)

        dcfg = self.cfg.dictionary
        self.dictionary = Dictionary(dcfg.L, dcfg.Q, dcfg.Href, dcfg.centers,
                                     dcfg.scales, self.cfg.site.Lx, self.cfg.site.Ly)
        self.G = self.dictionary.map_operator(
            self.cfg.display.depths, self.bathy, self.cfg.site.Lx,
            self.cfg.site.Ly, n=self.cfg.display.map_n)
        self.field_rms = float("nan")
        self.floor_rms = float("nan")
        self.calibrate_prior()

        self.library = None
        self.selection = np.zeros(0, int)
        self.by_epoch = []
        self.observations = np.zeros(0)
        self.result = None
        self._build_library(progress=progress, rebuild=rebuild)

    # ------------------------------------------------------------------ #
    # 先验标定（论文做法：用独立的海流统计量，而非本次声学数据）
    # ------------------------------------------------------------------ #
    def calibrate_prior(self, n: int = 21):
        """由参考场在报图网格上的投影标定先验尺度。

        离线流程（MATLAB ``run_study.m``）用同一规则：
        ``priorSigma = max(0.5 * fieldRms, 0.1)``。实际部署时应把参考场换成
        独立的 ADCP / 再分析海流统计量，这一步不含任何声学观测信息。
        """
        x, y, XG, YG = self.grid(n)
        rows, rhs = [], []
        for zd in self.cfg.display.depths:
            H = self.bathy(XG.ravel(), YG.ravel())
            z = np.minimum(float(zd), H - 0.4)
            Du, Dv = self.dictionary.velocity_basis(XG.ravel(), YG.ravel(), z, H)
            ut, vt = self.truth.velocity(XG.ravel(), YG.ravel(), z, 0.0)
            rows.append(np.vstack([Du, Dv]))
            rhs.append(np.concatenate([ut, vt]))
        G = np.vstack(rows)
        b = np.concatenate(rhs)
        a_fit, *_ = np.linalg.lstsq(G, b, rcond=1e-10)
        resid = G @ a_fit - b
        self.field_rms = float(np.sqrt(np.mean(b ** 2)))
        self.floor_rms = float(np.sqrt(np.mean(resid ** 2)))
        self.cfg.inv.field_sigma = max(0.5 * self.field_rms, 0.1)
        self.prior = self.dictionary.prior_matrix(
            self.G, self.cfg.inv.field_sigma, self.cfg.inv.rate_field_sigma)
        return a_fit

    # ------------------------------------------------------------------ #
    # 链路库
    # ------------------------------------------------------------------ #
    def _build_library(self, progress=None, rebuild: bool = False):
        fingerprint = acoustics.library_fingerprint(self.cfg, self.ssp, self.bathy,
                                                    self.dictionary)
        cache = acoustics.library_cache_path(self.cfg, self.cache_dir, fingerprint)
        if os.path.exists(cache) and not rebuild:
            try:
                lib = acoustics.load_library(self.cfg, cache)
                if lib.fingerprint == fingerprint:      # 命中且参数一致才复用
                    self.library = lib
                    if progress:
                        progress(1, 1)
                    return
            except Exception:
                pass
        lib = acoustics.build_library(self.cfg, self.ssp, self.bathy,
                                      self.dictionary, progress=progress)
        try:
            acoustics.save_library(lib, cache)
        except Exception:
            pass
        self.library = lib

    # ------------------------------------------------------------------ #
    # 采样与观测
    # ------------------------------------------------------------------ #
    def select(self, policy: str = "proposed", progress=None):
        self.selection = select_links(self.cfg, self.library, policy,
                                      progress=progress)
        self.by_epoch = links_by_epoch(self.cfg, self.library, self.selection)
        return self.selection

    def simulate(self, sel=None, noise: bool = True, seed=None):
        """由真值场生成所选链路的走时观测。"""
        sel = self.selection if sel is None else np.asarray(sel, int)
        self.observations = forward_truth(self.cfg, self.library, self.ssp,
                                          self.truth, sel, noise=noise, seed=seed)
        return self.observations

    def observations_for(self, link_indices):
        pos = {int(k): j for j, k in enumerate(self.selection)}
        return np.array([self.observations[pos[int(k)]]
                         for k in link_indices if int(k) in pos])

    # ------------------------------------------------------------------ #
    # 反演
    # ------------------------------------------------------------------ #
    def estimate(self, link_indices=None, use_nuisance=None):
        if link_indices is None:
            link_indices = self.selection
        link_indices = np.asarray(link_indices, int)
        y = self.observations_for(link_indices)
        cs = float(np.mean(self.ssp.c))
        self.result = invert(self.cfg, self.library, link_indices, y,
                             prior=self.prior, use_nuisance=use_nuisance,
                             cs=cs)
        return self.result

    # ------------------------------------------------------------------ #
    # 场量
    # ------------------------------------------------------------------ #
    def estimated_field(self) -> DictField:
        if self.result is None:
            return DictField(self.dictionary)
        return DictField(self.dictionary, self.result["a0"], self.result["adot"])

    def grid(self, n: int = 41):
        x = np.linspace(0.0, self.cfg.site.Lx, n)
        y = np.linspace(0.0, self.cfg.site.Ly, n)
        XG, YG = np.meshgrid(x, y)
        return x, y, XG, YG

    def field_on_grid(self, depth: float, n: int = 41, t_s: float = 0.0,
                      truth: bool = False):
        """返回 ``(x, y, X, Y, U, V, speed)``；``truth=True`` 时取参考场。"""
        x, y, XG, YG = self.grid(n)
        H = self.bathy(XG.ravel(), YG.ravel())
        z = np.minimum(float(depth), H - 0.4)
        if truth:
            u, v = self.truth.velocity(XG.ravel(), YG.ravel(), z, t_s)
        else:
            u, v = self.estimated_field().velocity(XG.ravel(), YG.ravel(), z, H, t_s)
        U = u.reshape(XG.shape)
        V = v.reshape(XG.shape)
        return x, y, XG, YG, U, V, np.hypot(U, V)

    def velocity_at(self, x, y, depth: float, t_s: float = 0.0,
                    truth: bool = False):
        """任意点（可批量）的水平流速，``depth`` 为水面下深度 (m)。"""
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        H = self.bathy(x, y)
        z = np.minimum(float(depth), H - 0.4)
        if truth:
            return self.truth.velocity(x, y, z, t_s)
        return self.estimated_field().velocity(x, y, z, H, t_s)

    def sample_point(self, x: float, y: float, depth: float, t_s: float = 0.0):
        """某点的估计流速与（仿真中的）真值。"""
        H = float(self.bathy(np.array([x]), np.array([y]))[0])
        z = min(float(depth), H - 0.4)
        est = self.estimated_field()
        ue, ve = est.velocity(np.array([x]), np.array([y]), np.array([z]),
                              np.array([H]), t_s)
        ut, vt = self.truth.velocity(np.array([x]), np.array([y]), np.array([z]), t_s)
        return {
            "x": float(x), "y": float(y), "z": float(z), "H": H,
            "u": float(ue[0]), "v": float(ve[0]),
            "speed": float(np.hypot(ue[0], ve[0])),
            "dir": float(np.degrees(np.arctan2(ue[0], ve[0])) % 360.0),
            "u_true": float(ut[0]), "v_true": float(vt[0]),
            "speed_true": float(np.hypot(ut[0], vt[0])),
            "dir_true": float(np.degrees(np.arctan2(ut[0], vt[0])) % 360.0),
        }

    def profile_at(self, x: float, y: float, nz: int = 22, t_s: float = 0.0):
        """某点的流速剖面（估计 / 真值）。"""
        H = float(self.bathy(np.array([x]), np.array([y]))[0])
        z = np.linspace(0.5, max(H - 0.5, 0.6), nz)
        est = self.estimated_field()
        ue, ve = est.velocity(np.full(nz, x), np.full(nz, y), z, np.full(nz, H), t_s)
        ut, vt = self.truth.velocity(np.full(nz, x), np.full(nz, y), z, t_s)
        return z, np.hypot(ue, ve), np.hypot(ut, vt)

    # ------------------------------------------------------------------ #
    # 状态与评价
    # ------------------------------------------------------------------ #
    def platform_state(self, epoch_index: int):
        """当前历元所有平台位置（AUV 只对有数据的时刻移动）。"""
        k = int(np.clip(epoch_index, 0, self.cfg.auv.epochs.size - 1))
        t = float(self.cfg.auv.epochs[k])
        x, y, is_auv = platform_positions(self.cfg, t)
        lon, lat = enu_to_lonlat(self.cfg, x, y)
        return {"t": t, "x": x, "y": y, "is_auv": is_auv, "lon": lon, "lat": lat}

    def platform_depths(self, epoch_index: int) -> np.ndarray:
        """各平台在当前历元的换能器深度 (m)。

        深度来自该历元被选中的发射/接收链路（AUV 在 6 m 与 16 m 之间切换）；
        尚未出现在链路里的平台取标称深度，浮标固定为 4 m。
        """
        n_auv = self.cfg.auv.N
        n_buoy = self.cfg.buoy.xy.shape[0]
        depths = np.full(n_auv + n_buoy, np.nan)
        if self.by_epoch:
            # 按时间顺序回放被选中的链路：平台深度 = 最近一条相关链路设定的值
            # （论文中 AUV 可在历元内换深，故取"当前指令深度"）
            last = int(np.clip(epoch_index, 0, len(self.by_epoch) - 1))
            for e in range(last + 1):
                for k in self.by_epoch[e]:
                    k = int(k)
                    depths[self.library.em[k]] = self.library.de[k]
                    depths[self.library.rc[k]] = self.library.dr[k]
        out = np.where(np.isnan(depths[:n_auv]), self.cfg.auv.depth_set[0],
                       depths[:n_auv])
        buoys = np.where(np.isnan(depths[n_auv:]), self.cfg.buoy.depth,
                         depths[n_auv:])
        return np.concatenate([out, buoys])

    def metrics(self, n: int = 21, depth: float | None = None):
        """估计场与参考场的 RMSE / 方向误差 / 相关（报图深度上的平均）。"""
        x, y, XG, YG = self.grid(n)
        depths = [depth] if depth is not None else list(self.cfg.display.depths)
        US, VS, UT, VT = [], [], [], []
        for zd in depths:
            H = self.bathy(XG.ravel(), YG.ravel())
            z = np.minimum(float(zd), H - 0.4)
            ue, ve = self.estimated_field().velocity(XG.ravel(), YG.ravel(), z, H, 0.0)
            ut, vt = self.truth.velocity(XG.ravel(), YG.ravel(), z, 0.0)
            US.append(ue); VS.append(ve); UT.append(ut); VT.append(vt)
        return errors(np.concatenate(UT), np.concatenate(VT),
                      np.concatenate(US), np.concatenate(VS))

    def summary(self) -> str:
        lib = self.library
        txt = [f"站点: {self.cfg.site.name}",
               f"输入: {'实测 qhd_inputs.mat' if self.using_matlab_inputs else '内置合成场景'}",
               f"候选链路: {lib.n_candidates}  可行: {lib.n}",
               f"字典: L={self.dictionary.L} Q={self.dictionary.Q} "
               f"系数 {self.dictionary.n_coef} (场+率 {2*self.dictionary.n_coef})"]
        if self.selection.size:
            txt.append(f"选中链路: {self.selection.size}")
        if self.result is not None:
            m = self.metrics()
            txt.append(f"RMSE {m['rmse']:.4f} m/s  方向误差 {m['dir_err']:.2f}°  "
                       f"相关 {m['corr']:.3f}")
        return "\n".join(txt)
