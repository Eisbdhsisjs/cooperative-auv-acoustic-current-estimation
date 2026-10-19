"""仿真配置：与 MATLAB 版 ``qhd_config.m`` 一一对应。"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

#: MATLAB 仿真目录（可选）。若存在 ``data/qhd_inputs.mat`` 则读取实测温盐
#: 声速剖面、GEBCO 地形与 ADCP 真值模态；否则使用内置合成场景。
MATLAB_SIM_DIR = r"C:\Users\asus\Desktop\5 Work4-Remote Sensing\Work3_Fish\Simulation_QHD"


def default_config(matlab_dir: str | None = None) -> SimpleNamespace:
    """返回配置树（属性式访问，如 ``cfg.auv.speed``）。"""
    d = matlab_dir or MATLAB_SIM_DIR
    cfg = SimpleNamespace()
    cfg.matlab_dir = d
    cfg.inputs_mat = os.path.join(d, "data", "qhd_inputs.mat")

    cfg.site = SimpleNamespace(
        name="Qinhuangdao shelf, Bohai Sea",
        lon=120.05,          # deg E
        lat=39.80,           # deg N
        Lx=10000.0,          # m
        Ly=10000.0,          # m
        nx=201, ny=201,
    )
    cfg.grid = SimpleNamespace(
        dx=cfg.site.Lx / (cfg.site.nx - 1),
        dy=cfg.site.Ly / (cfg.site.ny - 1),
    )

    cfg.auv = SimpleNamespace(
        N=7,
        speed=0.901,                                     # m/s
        epochs=np.array([250, 1000, 1750, 2500, 3250, 4000, 4750], float),
        depth_set=np.array([6.0, 16.0]),                 # 可选换能器深度 (m)
    )
    cfg.buoy = SimpleNamespace(
        xy=np.array([[1.0, 9400.0], [5000.0, 300.0], [9800.0, 5200.0]]),
        depth=4.0,
    )

    cfg.dictionary = SimpleNamespace(
        L=4, Q=5, Href=22.0,
        centers=np.array([[2500.0, 2500.0], [7400.0, 2600.0], [5200.0, 6000.0],
                          [2600.0, 7800.0], [7900.0, 7600.0]]),
        scales=np.array([1800.0, 2600.0, 3400.0, 2200.0, 3000.0]),
    )
    cfg.truth = SimpleNamespace(n_rbf=5, model_error=0.0, L=8, n_mode=3,
                                rbf_amp=0.40, adv_fac=0.55)

    cfg.ac = SimpleNamespace(
        freq=4000.0,          # Hz
        sigma_t=25e-6,        # s, 接收时刻精度
        noise=25e-6,          # s, 到达拾取标准差
        jitter=10e-6,         # s, 发射时间戳抖动
        range_min=800.0, range_max=10000.0,
        n_step=120,           # 每条参考射线的采样点数
        n_guess=180,          # 发射角搜索点数
        theta_max=float(np.deg2rad(70.0)),
        max_legs=400,
        depth_tol=1.0,
    )

    cfg.budget = SimpleNamespace(pings=120, move=2600.0, duration=5000.0)
    cfg.design = SimpleNamespace(kappa=0.30, rank_tol=1.0, max_sel=150)

    cfg.inv = SimpleNamespace(
        field_sigma=0.35,        # m/s, 场先验尺度
        rate_field_sigma=2e-4,   # m/s^2
        prior_sigma=0.35,
        rate_prior_sigma=2e-5,
        traj_sigma=5.0,          # m, AUV 位置不确定度
        use_nuisance=True,
    )
    cfg.display = SimpleNamespace(
        depths=np.array([4.0, 10.0, 18.0]),   # m, 报图深度
        grid_n=6,                             # 报图粗网格
        map_n=21,                             # 先验/损失评估网格
    )
    cfg.seed = 20260813
    return cfg
