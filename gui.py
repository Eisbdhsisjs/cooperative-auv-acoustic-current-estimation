"""Tkinter 显示界面（v2）：小蓝 AUV 位置 + 海流流速估计。

界面结构::

    顶部标题栏（数据源 / 精度状态）
    左栏控制区 ｜ 中部流速场地图 ｜ 右侧信息面板（小蓝 / 流速 / 剖面 / 质量）
    底部状态栏

右侧"流速"卡片是视觉重点：大字 m/s 读数、流向罗盘、u/v 分量、真值对比，
以及随流速变色的横向色条。地图上以蓝色高亮并标注"小蓝"。
快捷键：空格=开始/暂停，→=单步，←=回退，R=重新选择链路。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

try:                                              # 可选：用 Pillow 渲染底图
    from PIL import Image, ImageTk
    _HAS_PIL = True
except Exception:                                 # pragma: no cover
    _HAS_PIL = False

from .model import _AUV_HDG, auv_track
from .pipeline import Survey

FONT = "Microsoft YaHei UI"
FONT_NUM = "Segoe UI"
BG = "#eef2f7"
CARD = "#ffffff"
INK = "#16283a"
MUTED = "#6b7c8f"
LINE = "#d7e0ea"
BLUE = "#1f5fbf"
BLUE_DK = "#123a72"
BLUE_LT = "#e8f1fd"
ORANGE = "#ff8a3d"
RED = "#c0392b"
ANIM_MS = 70          # 动画帧间隔 (ms)，约 14 fps

_CMAP = np.array([
    [0.03, 0.09, 0.30], [0.05, 0.32, 0.62], [0.05, 0.60, 0.75],
    [0.10, 0.72, 0.50], [0.65, 0.83, 0.20], [0.98, 0.75, 0.10],
    [0.90, 0.25, 0.15], [0.62, 0.05, 0.10],
])


def speed_to_rgb(speed, vmax):
    """流速 → 0-255 RGB（深蓝慢 → 红快）。"""
    t = np.clip(np.asarray(speed, float) / max(float(vmax), 1e-6), 0.0, 1.0)
    pos = t * (_CMAP.shape[0] - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, _CMAP.shape[0] - 1)
    w = (pos - i0)[..., None]
    return ((_CMAP[i0] * (1 - w) + _CMAP[i1] * w) * 255).astype(np.uint8)


def rgb_to_photo(rgb: np.ndarray, side: int):
    """``(h, w, 3)`` uint8 → PhotoImage（需要 Pillow，缺失时返回 None）。"""
    if not _HAS_PIL:
        return None
    return ImageTk.PhotoImage(Image.fromarray(rgb).resize((side, side), Image.BILINEAR))


def color_of(speed, vmax) -> str:
    return "#%02x%02x%02x" % tuple(speed_to_rgb(speed, vmax))


def dir_name(deg: float) -> str:
    names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    return names[int((deg % 360) / 45.0 + 0.5) % 8]


class Card(tk.Frame):
    """带标题的白色卡片容器。"""

    def __init__(self, master, title: str, accent: str = BLUE):
        super().__init__(master, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        head = tk.Frame(self, bg=CARD)
        head.pack(fill=tk.X, padx=12, pady=(9, 0))
        tk.Frame(head, bg=accent, width=4, height=15).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(head, text=title, bg=CARD, fg=INK,
                 font=(FONT, 11, "bold")).pack(side=tk.LEFT)
        self.body = tk.Frame(self, bg=CARD)
        self.body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 10))


class ScrollColumn(tk.Frame):
    """纵向可滚动容器（右栏用，窗口变矮时内容不会被截断）。"""

    def __init__(self, master, width: int):
        super().__init__(master, bg=BG, width=width)
        self.user_scrolled = False
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0,
                                width=width - 14)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self._on_scroll)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = tk.Frame(self.canvas, bg=BG)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._win, width=e.width))
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, event):
        self.user_scrolled = True
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_scroll(self, *args):
        self.user_scrolled = True
        self.canvas.yview(*args)

    def reset_if_idle(self):
        """未被用户手动滚动时保持顶部（Tk 有时会把可视区自动滚走）。"""
        if not self.user_scrolled:
            self.canvas.yview_moveto(0.0)

    def scroll_top(self):
        self.canvas.yview_moveto(0.0)


class FlatButton(tk.Button):
    """扁平化按钮。"""

    def __init__(self, master, text, command=None, primary=False, width=11):
        super().__init__(
            master, text=text, command=command, width=width,
            bg=BLUE if primary else "#e6edf6", fg="#ffffff" if primary else INK,
            activebackground=BLUE_DK if primary else "#d5e2f2",
            activeforeground="#ffffff" if primary else INK,
            relief=tk.FLAT, bd=0, font=(FONT, 10), cursor="hand2", pady=5)


class Compass(tk.Canvas):
    """小蓝航向（灰）与海流流向（彩色）罗盘。"""

    def __init__(self, master, size=124):
        super().__init__(master, width=size, height=size, bg=CARD,
                         highlightthickness=0)
        self.size = size

    def draw(self, heading_deg, flow_deg, speed, vmax):
        c, s = self, self.size
        c.delete("all")
        cx = cy = s / 2 - 4
        r = s / 2 - 16
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#b9c9da", width=2,
                      fill="#f7fbff")
        for ang, txt in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            rad = np.deg2rad(90 - ang)
            x, y = cx + r * np.cos(rad), cy - r * np.sin(rad)
            c.create_text(x, y, text=txt, fill=MUTED, font=(FONT_NUM, 8))
        rad = np.deg2rad(90 - heading_deg)
        c.create_line(cx, cy, cx + 0.70 * r * np.cos(rad), cy - 0.70 * r * np.sin(rad),
                      fill="#9aa9b8", width=3, arrow=tk.LAST)
        rad = np.deg2rad(90 - flow_deg)
        c.create_line(cx, cy, cx + 0.94 * r * np.cos(rad), cy - 0.94 * r * np.sin(rad),
                      fill=color_of(speed, vmax), width=5, arrow=tk.LAST)
        c.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=INK, outline="")
        c.create_text(cx, s - 6, text="灰=航向   彩=流向", fill=MUTED, font=(FONT, 7))


class SpeedBar(tk.Canvas):
    """横向流速色条，带当前值游标。"""

    def __init__(self, master, width=372, height=36):
        super().__init__(master, width=width, height=height, bg=CARD,
                         highlightthickness=0)
        self.w, self.h = width, height

    def draw(self, speed, vmax):
        c = self
        c.delete("all")
        x0, x1 = 4, self.w - 42
        y0, y1 = 8, 22
        for i in range(int(x1 - x0)):
            f = i / max(x1 - x0 - 1, 1)
            c.create_line(x0 + i, y0, x0 + i, y1, fill=color_of(f * vmax, vmax))
        c.create_rectangle(x0, y0, x1, y1, outline=LINE)
        x = x0 + (x1 - x0) * min(speed / max(vmax, 1e-6), 1.0)
        c.create_polygon(x, y0 - 3, x - 5, y0 - 9, x + 5, y0 - 9, fill=INK, outline="")
        c.create_text(x0, y1 + 9, text="0", anchor="w", fill=MUTED, font=(FONT_NUM, 8))
        c.create_text(x1, y1 + 9, text=f"{vmax:.2f} m/s", anchor="e", fill=MUTED,
                      font=(FONT_NUM, 8))


class ProfilePlot(tk.Canvas):
    """该点流速剖面：估计（蓝实线）与真值（红虚线）。"""

    def __init__(self, master, width=372, height=180):
        super().__init__(master, width=width, height=height, bg="#fbfdff",
                         highlightthickness=1, highlightbackground=LINE)
        self.w, self.h = width, height

    def draw(self, z, est, truth):
        c = self
        c.delete("all")
        ml, mr, mt, mb = 40, 14, 16, 26
        zmax = max(float(z.max()), 1e-6)
        vmax = max(0.15, float(max(est.max(), truth.max())) * 1.15)
        for fi in np.linspace(0, 1, 5):
            xx = ml + (self.w - ml - mr) * fi
            c.create_line(xx, mt, xx, self.h - mb, fill="#e8eef5")
            c.create_text(xx, self.h - mb + 9, text=f"{fi * vmax:.2f}",
                          font=(FONT_NUM, 7), fill=MUTED)
        for fj in np.linspace(0, 1, 5):
            yy = mt + (self.h - mt - mb) * fj
            c.create_line(ml, yy, self.w - mr, yy, fill="#eef3f8")
        c.create_line(ml, mt, ml, self.h - mb, fill="#9aa9b8")
        c.create_line(ml, self.h - mb, self.w - mr, self.h - mb, fill="#9aa9b8")
        c.create_text(ml - 6, mt + 4, text="水面", anchor="e", font=(FONT, 7), fill=MUTED)
        c.create_text(ml - 6, self.h - mb - 4, text="底床", anchor="e", font=(FONT, 7),
                      fill=MUTED)
        c.create_text((ml + self.w) / 2, self.h - 7, text="流速 (m/s)", font=(FONT, 8),
                      fill=INK)
        for series, dash, col in ((est, None, BLUE), (truth, (5, 3), RED)):
            pts = []
            for zz, vv in zip(z, series):
                yy = self.h - mb - (self.h - mt - mb) * zz / zmax
                xx = ml + (self.w - ml - mr) * vv / vmax
                pts += [xx, yy]
            c.create_line(*pts, fill=col, width=2, dash=dash, smooth=True)
        c.create_text(self.w - mr, mt - 2, text="蓝=估计   红虚线=真值", anchor="e",
                      font=(FONT, 7), fill=MUTED)


class Sparkline(tk.Canvas):
    """小蓝处流速随历元的演变（蓝=估计，红虚线=真值）。"""

    def __init__(self, master, width=372, height=96):
        super().__init__(master, width=width, height=height, bg="#fbfdff",
                         highlightthickness=1, highlightbackground=LINE)
        self.w, self.h = width, height

    def draw(self, est_series, truth_series):
        c = self
        c.delete("all")
        ml, mr, mt, mb = 34, 12, 10, 20
        if not est_series:
            c.create_text(self.w / 2, self.h / 2, text="（推进历元后显示流速变化）",
                          fill=MUTED, font=(FONT, 8))
            return
        vmax = max(0.2, max(list(est_series) + list(truth_series)) * 1.15)
        n = len(est_series)
        for fi in np.linspace(0, 1, 4):
            yy = self.h - mb - (self.h - mt - mb) * fi
            c.create_line(ml, yy, self.w - mr, yy, fill="#eef3f8")
            c.create_text(ml - 5, yy, text=f"{fi * vmax:.1f}", anchor="e",
                          font=(FONT_NUM, 7), fill=MUTED)
        for series, dash, col, dots in ((truth_series, (4, 3), RED, False),
                                        (est_series, None, BLUE, True)):
            pts = []
            for i, v in enumerate(series):
                xx = ml + (self.w - ml - mr) * (i / max(n - 1, 1))
                yy = self.h - mb - (self.h - mt - mb) * min(v / vmax, 1.0)
                pts += [xx, yy]
            if len(pts) >= 4:
                c.create_line(*pts, fill=col, width=2, dash=dash)
            if dots:
                for i in range(0, len(pts), 2):
                    c.create_oval(pts[i] - 2, pts[i + 1] - 2, pts[i] + 2, pts[i + 1] + 2,
                                  fill=col, outline="")
        c.create_text(ml, self.h - 7, text="历元 →", font=(FONT, 7), fill=MUTED,
                      anchor="w")
        c.create_text(self.w - mr, self.h - 7, text="m/s", font=(FONT, 7), fill=MUTED,
                      anchor="e")


class DepthSection(tk.Canvas):
    """水柱剖面：各平台的换能器深度（AUV 三角 / 浮标圆点），底床随位置起伏。"""

    def __init__(self, master, width=376, height=170):
        super().__init__(master, width=width, height=height, bg="#fbfdff",
                         highlightthickness=1, highlightbackground=LINE)
        self.w, self.h = width, height

    def draw(self, is_auv, depths, water_depth):
        c = self
        c.delete("all")
        ml, mr, mt, mb = 40, 12, 18, 26
        n = len(depths)
        zmax = float(max(np.max(water_depth) * 1.06, np.max(depths) * 1.25, 20.0))
        x0, x1 = ml, self.w - mr

        def px(i):
            return x0 + (x1 - x0) * (i / max(n - 1, 1))

        def py(depth):
            return mt + (self.h - mt - mb) * min(depth / zmax, 1.0)

        # 水体与底床
        bed = [v for pair in zip([px(i) for i in range(n)], [py(water_depth[i])
                                                            for i in range(n)]) for v in pair]
        poly = [x0, mt] + bed + [x1, py(water_depth[-1]), x1, mt]
        c.create_polygon(*poly, fill="#e5f1fb", outline="")
        c.create_line(*bed, fill="#8a7a5f", width=2)
        c.create_line(x0, mt, x1, mt, fill="#7fb3dd", width=2)
        for d in (0, 5, 10, 15, 20):
            if d > zmax:
                break
            yy = py(d)
            c.create_line(x0, yy, x1, yy, fill="#eef3f8")
            c.create_text(ml - 6, yy, text=f"{d}", anchor="e", font=(FONT_NUM, 7),
                          fill=MUTED)
        c.create_text(ml - 6, mt - 8, text="深度", anchor="e", font=(FONT, 7), fill=MUTED)
        c.create_text((x0 + x1) / 2, self.h - 8, text="平台（左→右：小蓝、其它 AUV、浮标）",
                      font=(FONT, 7), fill=MUTED)

        for i in range(n):
            xx, yy = px(i), py(depths[i])
            if is_auv[i]:
                col = BLUE if i == 0 else "#33475b"
                c.create_polygon(xx, yy - 6, xx - 6, yy + 5, xx + 6, yy + 5,
                                 fill=col, outline="#ffffff")
                c.create_text(xx, yy + 15, text=f"{depths[i]:.0f}", font=(FONT_NUM, 7),
                              fill=col)
                c.create_text(xx, mt - 8, text=("小蓝" if i == 0 else f"A{i+1}"),
                              font=(FONT, 7), fill=col)
            else:
                c.create_oval(xx - 4, yy - 4, xx + 4, yy + 4, fill=RED, outline="#ffffff")
                c.create_text(xx, yy + 14, text=f"{depths[i]:.0f}", font=(FONT_NUM, 7),
                              fill=RED)
                c.create_text(xx, mt - 8, text=f"浮{i - sum(1 for v in is_auv if v) + 1}",
                              font=(FONT, 7), fill=RED)


class App(tk.Tk):
    """主窗口。"""

    def __init__(self, survey: Survey | None = None):
        super().__init__()
        self.title("小蓝 AUV · 海流估计与显示系统")
        # 依据 Tk 报告的可视区域自适应（高 DPI 下 Tk 会看到被虚拟化的小屏）
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1440, max(1180, sw - 40))}x{min(950, max(700, sh - 70))}")
        self.minsize(1120, 660)
        self.configure(bg=BG)

        self.survey = survey
        self.epoch = 0
        self.running = False
        self.show_links = tk.BooleanVar(value=True)
        self.show_tracks = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)
        self.show_particles = tk.BooleanVar(value=True)
        self.animate_time = tk.BooleanVar(value=True)
        self.field_truth = tk.BooleanVar(value=False)
        self.display_depth = tk.DoubleVar(value=6.0)
        self.policy = tk.StringVar(value="proposed")
        self.pick_xy: tuple[float, float] | None = None
        self._bg_photo = None
        self._vmax = 0.5
        self._pad = 58
        self._scale = 1.0
        self.clock_t = 0.0                 # 显示时刻（相对参考时刻，s）
        self._tick_n = 0
        self.particles = None
        self.particles_prev = None
        self._prng = np.random.default_rng(7)
        self._speed_hist: list[float] = []
        self._truth_hist: list[float] = []

        self._build_theme()
        self._build_header()
        self._build_body()
        self._build_status()
        self._bind_keys()
        self.after(60, self._initialise)

    # ------------------------------------------------------------------ #
    def _build_theme(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except Exception:
            pass
        st.configure("TCombobox", fieldbackground="#ffffff", background="#ffffff",
                     bordercolor=LINE, arrowcolor=INK)
        st.configure("TProgressbar", troughcolor="#dfe7f0", background=BLUE,
                     bordercolor="#dfe7f0", lightcolor=BLUE, darkcolor=BLUE)
        st.configure("Depth.Horizontal.TScale", background=CARD, troughcolor="#dfe7f0")

    def _build_header(self):
        head = tk.Frame(self, bg=BLUE_DK, height=56)
        head.pack(side=tk.TOP, fill=tk.X)
        head.pack_propagate(False)
        tk.Label(head, text="●", bg=BLUE_DK, fg="#5ea8ff",
                 font=(FONT, 14)).pack(side=tk.LEFT, padx=(16, 6))
        tk.Label(head, text="小蓝 AUV 海流估计与显示系统", bg=BLUE_DK, fg="#ffffff",
                 font=(FONT, 15, "bold")).pack(side=tk.LEFT)
        tk.Label(head, text="多 AUV 协同声学反演 · 分层流函数 + 移动端点走时",
                 bg=BLUE_DK, fg="#a9c8ee", font=(FONT, 9)).pack(side=tk.LEFT, padx=16)
        self.chip_rmse = tk.Label(head, text="场 RMSE —", bg="#1b4b86", fg="#dce9fb",
                                  font=(FONT, 9), padx=10, pady=4)
        self.chip_rmse.pack(side=tk.RIGHT, padx=(4, 14))
        self.chip_src = tk.Label(head, text="数据源 —", bg="#1b4b86", fg="#dce9fb",
                                 font=(FONT, 9), padx=10, pady=4)
        self.chip_src.pack(side=tk.RIGHT, padx=4)

    def _build_body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg=BG, width=230)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 6), pady=8)
        left.pack_propagate(False)
        c1 = Card(left, "运行控制")
        c1.pack(fill=tk.X, pady=(0, 8))
        self.btn_run = FlatButton(c1.body, "▶  开始演示", self.toggle_run, primary=True,
                                  width=14)
        self.btn_run.pack(fill=tk.X, pady=2)
        row = tk.Frame(c1.body, bg=CARD)
        row.pack(fill=tk.X, pady=2)
        FlatButton(row, "⏭ 单步", self.step, width=6).pack(side=tk.LEFT, padx=(0, 4))
        FlatButton(row, "↻ 重算", self.reselect, width=6).pack(side=tk.LEFT)
        c2 = Card(left, "采样策略")
        c2.pack(fill=tk.X, pady=(0, 8))
        ttk.Combobox(c2.body, textvariable=self.policy, state="readonly",
                     values=["proposed", "posterior", "uniform", "random", "fixed"]
                     ).pack(fill=tk.X)
        tk.Label(c2.body, text="proposed = 论文信息驱动准则", bg=CARD, fg=MUTED,
                 font=(FONT, 8)).pack(anchor="w", pady=(4, 0))
        c3 = Card(left, "显示深度")
        c3.pack(fill=tk.X, pady=(0, 8))
        self.lbl_depth = tk.Label(c3.body, text="6 m", bg=CARD, fg=BLUE,
                                  font=(FONT_NUM, 15, "bold"))
        self.lbl_depth.pack(anchor="w")
        ttk.Scale(c3.body, from_=1.0, to=21.0, variable=self.display_depth,
                  command=self._on_depth, style="Depth.Horizontal.TScale"
                  ).pack(fill=tk.X, pady=(2, 0))
        c4 = Card(left, "图层")
        c4.pack(fill=tk.BOTH, expand=True)
        for text, var in (("声学链路", self.show_links), ("AUV 航迹", self.show_tracks),
                          ("公里网格", self.show_grid),
                          ("流动粒子", self.show_particles),
                          ("时间推进（流场随 t 演化）", self.animate_time)):
            tk.Checkbutton(c4.body, text=text, variable=var, bg=CARD, fg=INK,
                           activebackground=CARD, font=(FONT, 9),
                           command=self.redraw).pack(anchor="w")
        tk.Frame(c4.body, bg=LINE, height=1).pack(fill=tk.X, pady=6)
        tk.Radiobutton(c4.body, text="显示估计场", variable=self.field_truth, value=False,
                       bg=CARD, fg=INK, activebackground=CARD, font=(FONT, 9),
                       command=self.redraw).pack(anchor="w")
        tk.Radiobutton(c4.body, text="显示真值场", variable=self.field_truth, value=True,
                       bg=CARD, fg=INK, activebackground=CARD, font=(FONT, 9),
                       command=self.redraw).pack(anchor="w")

        mid = tk.Frame(body, bg=BG)
        mid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=8)
        wrap = tk.Frame(mid, bg=BG, highlightbackground=LINE, highlightthickness=1)
        wrap.pack(fill=tk.BOTH, expand=True)
        self.map = tk.Canvas(wrap, bg="#dfe9f4", highlightthickness=0)
        self.map.pack(fill=tk.BOTH, expand=True)
        self.map.bind("<Button-1>", self.on_map_click)
        self.map.bind("<Configure>", lambda *_: self.redraw_map())

        right = tk.Frame(body, bg=BG, width=440)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 10), pady=8)
        right.pack_propagate(False)
        self.right_col = ScrollColumn(right, 440)
        self.right_col.pack(fill=tk.BOTH, expand=True)
        right = self.right_col.inner

        c5 = Card(right, "小蓝", accent="#5ea8ff")
        c5.pack(fill=tk.X, pady=(0, 8))
        top = tk.Frame(c5.body, bg=CARD)
        top.pack(fill=tk.X)
        info = tk.Frame(top, bg=CARD)
        info.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_pos = tk.Label(info, text="位置 —", bg=CARD, fg=INK, anchor="w",
                                justify="left", font=(FONT, 9))
        self.lbl_pos.pack(anchor="w")
        self.lbl_geo = tk.Label(info, text="经纬度 —", bg=CARD, fg=MUTED, anchor="w",
                                justify="left", font=(FONT, 9))
        self.lbl_geo.pack(anchor="w")
        self.lbl_state = tk.Label(info, text="深度 —   航速 —   航向 —", bg=CARD,
                                  fg=MUTED, anchor="w", justify="left", font=(FONT, 9))
        self.lbl_state.pack(anchor="w", pady=(3, 0))
        self.compass = Compass(top, 104)
        self.compass.pack(side=tk.RIGHT)

        c6 = Card(right, "流速  ·  小蓝所在处", accent=ORANGE)
        c6.pack(fill=tk.X, pady=(0, 8))
        spd = tk.Frame(c6.body, bg=BLUE_LT, highlightbackground="#c9ddf6",
                       highlightthickness=1)
        spd.pack(fill=tk.X)
        inner = tk.Frame(spd, bg=BLUE_LT)
        inner.pack(fill=tk.X, padx=12, pady=8)
        tk.Label(inner, text="估计流速", bg=BLUE_LT, fg="#33506e",
                 font=(FONT, 9)).pack(anchor="w")
        self.lbl_speed = tk.Label(inner, text="—  m/s", bg=BLUE_LT, fg=BLUE_DK,
                                  font=(FONT_NUM, 31, "bold"))
        self.lbl_speed.pack(anchor="w")
        self.lbl_dir = tk.Label(inner, text="流向 —", bg=BLUE_LT, fg=INK,
                                font=(FONT, 12))
        self.lbl_dir.pack(anchor="w")
        self.lbl_uv = tk.Label(inner, text="u = —    v = —", bg=BLUE_LT, fg="#41576e",
                               font=(FONT, 9))
        self.lbl_uv.pack(anchor="w")
        self.lbl_cmp = tk.Label(inner, text="真值 —", bg=BLUE_LT, fg=MUTED,
                                font=(FONT, 9))
        self.lbl_cmp.pack(anchor="w", pady=(3, 0))
        self.speedbar = SpeedBar(c6.body, width=376)
        self.speedbar.pack(anchor="w", pady=(8, 0))
        self.lbl_hint = tk.Label(c6.body, text="点击地图任意点可读取该处流速", bg=CARD,
                                 fg=MUTED, font=(FONT, 8))
        self.lbl_hint.pack(anchor="w", pady=(3, 0))

        c7 = Card(right, "流速剖面")
        c7.pack(fill=tk.X, pady=(0, 8))
        self.profile = ProfilePlot(c7.body, width=376, height=152)
        self.profile.pack(anchor="w")
        tk.Label(c7.body, text="估计受场空间先验约束：该场景可辨识方向有限，"
                               "估计曲线通常比真值平滑",
                 bg=CARD, fg=MUTED, font=(FONT, 8), wraplength=372,
                 justify="left").pack(anchor="w", pady=(4, 0))
        c7b = Card(right, "各平台所在深度")
        c7b.pack(fill=tk.X, pady=(0, 8))
        self.depthsec = DepthSection(c7b.body, width=376, height=168)
        self.depthsec.pack(anchor="w")
        tk.Label(c7b.body, text="AUV 在 6 m 与 16 m 之间切换换能器深度，"
                                "浮标固定 4 m；数字为当前深度 (m)",
                 bg=CARD, fg=MUTED, font=(FONT, 8), wraplength=372,
                 justify="left").pack(anchor="w", pady=(4, 0))

        c8 = Card(right, "估计质量")
        c8.pack(fill=tk.BOTH, expand=True)
        self.lbl_quality = tk.Label(c8.body, text="—", bg=CARD, fg=INK, anchor="w",
                                    justify="left", font=(FONT, 9))
        self.lbl_quality.pack(anchor="w")
        self.prog = ttk.Progressbar(c8.body, style="TProgressbar", maximum=1.0)
        self.prog.pack(fill=tk.X, pady=(6, 2))
        self.spark = Sparkline(c8.body, width=376, height=82)
        self.spark.pack(anchor="w", pady=(4, 0))

    def _build_status(self):
        bar = tk.Frame(self, bg="#e3eaf3", height=30)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.pack_propagate(False)
        self.status = tk.Label(bar, text="正在初始化…", bg="#e3eaf3", fg=INK,
                               anchor="w", font=(FONT, 9))
        self.status.pack(side=tk.LEFT, padx=12)
        self.lbl_clock = tk.Label(bar, text="t = 0 s", bg="#e3eaf3", fg=MUTED,
                                  font=(FONT_NUM, 9))
        self.lbl_clock.pack(side=tk.RIGHT, padx=12)

    def _bind_keys(self):
        self.bind("<space>", lambda e: self.toggle_run())
        self.bind("<Right>", lambda e: self.step())
        self.bind("<Left>", lambda e: self.goto(self.epoch - 1))
        self.bind("<r>", lambda e: self.reselect())

    # ------------------------------------------------------------------ #
    # 初始化与交互
    # ------------------------------------------------------------------ #
    def _initialise(self):
        if self.survey is None:
            self.status.config(text="正在构建候选声学链路库…")
            self.update_idletasks()

            def prog(i, n):
                self.status.config(text=f"正在构建候选声学链路库… {i}/{n}")
                self.update_idletasks()

            self.survey = Survey(progress=prog)
        s = self.survey
        self.status.config(text="选择声学链路…")
        self.update_idletasks()
        s.select(self.policy.get())
        s.simulate()
        s.estimate(self._links_upto(self.epoch))
        self.chip_src.config(text="实测输入 qhd_inputs.mat" if s.using_matlab_inputs
                             else "内置合成场景")
        self.clock_t = float(s.cfg.auv.epochs[0])
        self._refresh_histories()
        self.redraw()
        self.after(150, self._animate)          # 流场动画主循环
        self.after(400, self.toggle_run)

    def _links_upto(self, epoch: int) -> np.ndarray:
        if not self.survey.by_epoch:
            return np.zeros(0, int)
        last = min(max(epoch, 0), len(self.survey.by_epoch) - 1)
        parts = [self.survey.by_epoch[i] for i in range(last + 1)]
        return np.concatenate(parts) if parts else np.zeros(0, int)

    def toggle_run(self):
        self.running = not self.running
        self.btn_run.config(text="⏸  暂停" if self.running else "▶  开始演示",
                            bg=ORANGE if self.running else BLUE,
                            activebackground="#e07a2e" if self.running else BLUE_DK)
        if self.running:
            if self.epoch >= self.survey.cfg.auv.epochs.size - 1 and self.survey.by_epoch:
                self.goto(0)
            self._tick()

    def _tick(self):
        if not self.running:
            return
        if self.epoch >= self.survey.cfg.auv.epochs.size - 1:
            self.running = False
            self.btn_run.config(text="▶  开始演示", bg=BLUE, activebackground=BLUE_DK)
            self.status.config(text="演示完成：已使用全部历元的声学链路")
        else:
            self.step()
        if self.running:
            self.after(1500, self._tick)

    def step(self):
        if self.epoch < self.survey.cfg.auv.epochs.size - 1:
            self.goto(self.epoch + 1)

    def goto(self, epoch: int):
        n = self.survey.cfg.auv.epochs.size
        self.epoch = int(np.clip(epoch, 0, n - 1))
        self.clock_t = float(self.survey.cfg.auv.epochs[self.epoch])
        self.survey.estimate(self._links_upto(self.epoch))
        self._refresh_histories()
        self.redraw()

    def reselect(self):
        self.status.config(text="重新选择链路…")
        self.update_idletasks()
        self.survey.select(self.policy.get())
        self.survey.simulate()
        self.survey.estimate(self._links_upto(self.epoch))
        self._refresh_histories()
        self.redraw()

    def _on_depth(self, _=None):
        self.lbl_depth.config(text=f"{float(self.display_depth.get()):.0f} m")
        self.redraw_map()

    def on_map_click(self, event):
        x = (event.x - self._pad) / self._scale
        y = (self.map.winfo_height() - self._pad - event.y) / self._scale
        if 0 <= x <= self.survey.cfg.site.Lx and 0 <= y <= self.survey.cfg.site.Ly:
            self.pick_xy = (float(x), float(y))
            self.redraw_map()
            self._draw_panel()

    def _blue_depth(self, epoch: int) -> float:
        """小蓝在当前历元的换能器深度（与地图标注同源）。"""
        return float(self.survey.platform_depths(epoch)[0])

    def _refresh_histories(self):
        """记录小蓝处流速随历元的演变（不改动当前 result）。"""
        s = self.survey
        keep = s.result
        est, tru = [], []
        for e in range(self.epoch + 1):
            s.estimate(self._links_upto(e))
            st = s.platform_state(e)
            pt = s.sample_point(float(st["x"][0]), float(st["y"][0]),
                                self._blue_depth(e), st["t"])
            est.append(pt["speed"])
            tru.append(pt["speed_true"])
        s.result = keep
        self._speed_hist, self._truth_hist = est, tru

    # ------------------------------------------------------------------ #
    # 绘制
    # ------------------------------------------------------------------ #
    def redraw(self):
        if self.survey is None:
            return
        self.redraw_map()
        self._draw_panel()
        self.right_col.reset_if_idle()

    def redraw_map(self):
        if self.survey is None:
            return
        s = self.survey
        c = self.map
        c.delete("all")
        W = c.winfo_width() or 900
        H = c.winfo_height() or 800
        Lx = s.cfg.site.Lx
        pad = 58
        self._pad = pad
        self._W, self._H = W, H
        self._scale = scale = min((W - 2 * pad) / Lx, (H - 2 * pad) / Lx)

        def sx(x):
            return pad + x * scale

        def sy(y):
            return H - pad - y * scale

        depth = float(self.display_depth.get())
        truth = self.field_truth.get()
        x, y, XG, YG, U, V, S = s.field_on_grid(depth, n=121, t_s=self.clock_t,
                                                truth=truth)
        vmax = max(0.5, float(np.nanpercentile(S, 98)))
        self._vmax = vmax

        side = max(int(scale * Lx), 10)
        rgb = speed_to_rgb(S, vmax).astype(float)
        rgb = 0.74 * rgb + 0.26 * np.array([238.0, 243.0, 249.0])
        photo = rgb_to_photo(np.flipud(rgb.astype(np.uint8)), side)
        if photo is not None:
            self._bg_photo = photo
            # 图像第 0 行 = 北 = 屏幕顶部，故锚在域左上角
            c.create_image(sx(0), sy(Lx), image=photo, anchor="nw")

        if self.show_grid.get():
            for i in range(11):
                xx = i * 1000.0
                c.create_line(sx(xx), sy(0), sx(xx), sy(Lx), fill="#ffffff", dash=(2, 7))
                c.create_line(sx(0), sy(xx), sx(Lx), sy(xx), fill="#ffffff", dash=(2, 7))
                c.create_text(sx(xx), sy(0) + 13, text=f"{xx/1000:.0f}",
                              fill="#33475b", font=(FONT_NUM, 8))
                c.create_text(sx(0) - 15, sy(xx), text=f"{xx/1000:.0f}",
                              fill="#33475b", font=(FONT_NUM, 8))
        c.create_text(sx(Lx / 2), sy(0) + 30, text="东向 (km)", fill=INK, font=(FONT, 9))
        c.create_text(sx(0) - 34, sy(Lx / 2), text="北向 (km)", fill=INK, angle=90,
                      font=(FONT, 9))
        c.create_rectangle(sx(0), sy(Lx), sx(Lx), sy(0), outline="#8fa4b8")

        step = max(1, S.shape[0] // 16)
        for iy in range(0, S.shape[0], step):
            for ix in range(0, S.shape[1], step):
                uu, vv = U[iy, ix], V[iy, ix]
                if not (np.isfinite(uu) and np.isfinite(vv)):
                    continue
                sp = float(np.hypot(uu, vv))
                if sp < 1e-3:
                    continue
                px, py = sx(x[ix]), sy(y[iy])
                ln = 32.0 * min(sp / vmax, 1.0)
                dx, dy = uu / sp * ln, -vv / sp * ln
                c.create_line(px, py, px + dx, py + dy, fill="#152534", width=2)
                for sgn in (1.0, -1.0):
                    c.create_line(px + dx, py + dy,
                                  px + dx - 0.42 * dx + sgn * 0.30 * dy,
                                  py + dy - 0.42 * dy - sgn * 0.30 * dx,
                                  fill="#152534", width=2)

        if self.show_links.get() and s.by_epoch:
            for e_ in range(min(self.epoch, len(s.by_epoch) - 1) + 1):
                current = (e_ == self.epoch)
                col = ORANGE if current else "#f6cda9"
                wid = 2 if current else 1
                st_e = s.platform_state(e_)
                for k in s.by_epoch[e_]:
                    k = int(k)
                    c.create_line(sx(st_e["x"][s.library.em[k]]),
                                  sy(st_e["y"][s.library.em[k]]),
                                  sx(st_e["x"][s.library.rc[k]]),
                                  sy(st_e["y"][s.library.rc[k]]),
                                  fill=col, width=wid)

        st = s.platform_state(self.epoch)
        if self.show_tracks.get():
            for i in range(s.cfg.auv.N):
                tx, ty = auv_track(s.cfg, i, 40)
                pts = [v for pair in zip(map(sx, tx), map(sy, ty)) for v in pair]
                c.create_line(*pts, fill=BLUE if i == 0 else "#b3c2d1",
                              width=2 if i == 0 else 1,
                              dash=None if i == 0 else (3, 3))

        depths = s.platform_depths(self.epoch)
        for i in range(s.cfg.auv.N):
            px, py = sx(st["x"][i]), sy(st["y"][i])
            if i == 0:
                c.create_oval(px - 22, py - 22, px + 22, py + 22, outline="#cfe2fb")
                c.create_oval(px - 16, py - 16, px + 16, py + 16, outline="#8fc0ff",
                              width=2)
                c.create_polygon(px, py - 12, px - 10, py + 10, px + 10, py + 10,
                                 fill=BLUE, outline="#ffffff", width=2)
                c.create_text(px, py - 32, text=f"小蓝 · {depths[i]:.0f} m",
                              fill=BLUE_DK, font=(FONT, 11, "bold"))
            else:
                c.create_polygon(px, py - 8, px - 7, py + 7, px + 7, py + 7,
                                 fill="#33475b", outline="#ffffff")
                c.create_text(px, py + 18, text=f"{depths[i]:.0f}m", fill="#33475b",
                              font=(FONT_NUM, 8))
        for j in range(s.cfg.buoy.xy.shape[0]):
            i = s.cfg.auv.N + j
            px, py = sx(st["x"][i]), sy(st["y"][i])
            c.create_oval(px - 7, py - 7, px + 7, py + 7, fill=RED, outline="#ffffff")
            c.create_text(px, py + 17, text=f"{depths[i]:.0f}m", fill=RED,
                          font=(FONT_NUM, 8))

        if self.pick_xy is not None:
            px, py = sx(self.pick_xy[0]), sy(self.pick_xy[1])
            c.create_line(px - 11, py, px + 11, py, fill="#111111", width=2)
            c.create_line(px, py - 11, px, py + 11, fill="#111111", width=2)
            c.create_oval(px - 4, py - 4, px + 4, py + 4, outline="#111111")

        bx, by = sx(0) + 10, sy(0) - 22
        c.create_line(bx, by, bx + 2000 * scale, by, fill=INK, width=3)
        for xedge in (bx, bx + 2000 * scale):
            c.create_line(xedge, by - 5, xedge, by + 5, fill=INK, width=2)
        c.create_text(bx + 1000 * scale, by - 10, text="2 km", fill=INK,
                      font=(FONT_NUM, 8))
        nx, ny = sx(Lx) - 26, sy(Lx) + 34
        c.create_line(nx, ny + 16, nx, ny - 8, fill=INK, width=2, arrow=tk.LAST)
        c.create_text(nx, ny - 18, text="N", fill=INK, font=(FONT_NUM, 9, "bold"))

        bx0, bx1 = W - pad + 14, W - pad + 32
        ytop, ybot = pad, H - pad - 30
        for i in range(64):
            f = 1.0 - i / 63.0
            y0 = ytop + (ybot - ytop) * i / 64.0
            c.create_rectangle(bx0, y0, bx1, y0 + (ybot - ytop) / 64.0 + 1,
                               fill=color_of(f * vmax, vmax), outline="")
        c.create_rectangle(bx0, ytop, bx1, ybot, outline=LINE)
        for f in (0.0, 0.5, 1.0):
            yy = ybot - (ybot - ytop) * f
            c.create_line(bx1, yy, bx1 + 4, yy, fill=MUTED)
            c.create_text(bx1 + 7, yy, text=f"{f * vmax:.2f}", anchor="w",
                          fill=MUTED, font=(FONT_NUM, 8))
        c.create_text((bx0 + bx1) / 2, ytop - 12, text="流速 m/s", font=(FONT, 9),
                      fill=INK)
        c.create_text(pad, pad - 30, anchor="w", fill=INK, font=(FONT, 10, "bold"),
                      text=f"{'参考真值场' if truth else '声学估计场'} · "
                           f"{depth:.0f} m · 历元 {self.epoch + 1}/"
                           f"{s.cfg.auv.epochs.size} · t = {self.clock_t:.0f} s")

        lgx, lgy = pad + 6, pad + 6
        items = [("▲", BLUE, "小蓝（AUV-1）"), ("▲", "#33475b", "其它 AUV"),
                 ("●", RED, "浮标"), ("—", ORANGE, "本历元链路"),
                 ("—", "#f6cda9", "历史链路")]
        c.create_rectangle(lgx, lgy, lgx + 150, lgy + 18 * len(items) + 10,
                           fill="#ffffff", outline=LINE)
        for i, (sym, col, txt) in enumerate(items):
            yy = lgy + 16 + 18 * i
            c.create_text(lgx + 14, yy, text=sym, fill=col, font=(FONT, 9, "bold"))
            c.create_text(lgx + 28, yy, text=txt, anchor="w", fill=INK, font=(FONT, 8))
        self._draw_particles()

    # ------------------------------------------------------------------ #
    # 流场动画：时间推进 + 流动粒子
    # ------------------------------------------------------------------ #
    def _sx(self, x):
        return self._pad + np.asarray(x, dtype=float) * self._scale

    def _sy(self, y):
        return self._H - self._pad - np.asarray(y, dtype=float) * self._scale

    def _animate(self):
        """约 14 fps 的动画主循环。"""
        try:
            if self.survey is not None:
                if self.animate_time.get():
                    # 视觉时间：每真实秒推进 120 s 模型时间，在调查窗内循环
                    self.clock_t += 120.0 * ANIM_MS / 1000.0
                    tmax = float(self.survey.cfg.auv.epochs.max())
                    if self.clock_t > tmax:
                        self.clock_t = 0.0
                    self._tick_n += 1
                    if self._tick_n >= max(int(1000 / ANIM_MS), 1):
                        self._tick_n = 0
                        self.redraw()            # 流场与读数随 t 刷新
                if self.show_particles.get():
                    self._step_particles()
                self._draw_particles()
        except Exception:
            pass
        finally:
            self.after(ANIM_MS, self._animate)

    def _step_particles(self, n: int = 230, dt_vis: float = 150.0):
        """粒子按流速平流（dt_vis 为每帧等效模型时间，用于视觉放大）。"""
        s = self.survey
        Lx, Ly = s.cfg.site.Lx, s.cfg.site.Ly
        if self.particles is None or self.particles.shape[0] != n:
            self.particles = np.column_stack([self._prng.uniform(0, Lx, n),
                                              self._prng.uniform(0, Ly, n)])
            self.particles_prev = self.particles.copy()
        depth = float(self.display_depth.get())
        u, v = s.velocity_at(self.particles[:, 0], self.particles[:, 1], depth,
                             self.clock_t, truth=self.field_truth.get())
        nxt = self.particles + np.column_stack([u, v]) * dt_vis
        outside = ((nxt[:, 0] < 0) | (nxt[:, 0] > Lx) |
                   (nxt[:, 1] < 0) | (nxt[:, 1] > Ly) |
                   (self._prng.random(n) < 0.004))
        k = int(outside.sum())
        if k:
            nxt[outside] = np.column_stack([self._prng.uniform(0, Lx, k),
                                            self._prng.uniform(0, Ly, k)])
        self.particles_prev = self.particles
        self.particles = nxt

    def _draw_particles(self):
        c = self.map
        c.delete("particles")
        if not self.show_particles.get() or self.particles is None:
            return
        px0 = self._sx(self.particles_prev[:, 0])
        py0 = self._sy(self.particles_prev[:, 1])
        px1 = self._sx(self.particles[:, 0])
        py1 = self._sy(self.particles[:, 1])
        for i in range(self.particles.shape[0]):
            if abs(px1[i] - px0[i]) > 160 or abs(py1[i] - py0[i]) > 160:
                continue                        # 重生造成的跳变不画
            c.create_line(px0[i], py0[i], px1[i], py1[i], fill="#1d3a55",
                          width=3, tags="particles")
            c.create_line(px0[i], py0[i], px1[i], py1[i], fill="#ffffff",
                          width=1, tags="particles")

    def _draw_panel(self):
        s = self.survey
        st = s.platform_state(self.epoch)
        x0, y0 = float(st["x"][0]), float(st["y"][0])
        depth = self._blue_depth(self.epoch)
        pt = s.sample_point(x0, y0, depth, self.clock_t)
        vmax = self._vmax

        self.lbl_pos.config(text=f"位置  x = {x0/1000:.2f} km,  y = {y0/1000:.2f} km")
        self.lbl_geo.config(text=f"经纬度  {st['lon'][0]:.4f}°E, {st['lat'][0]:.4f}°N")
        hdg = float(np.degrees(np.arctan2(np.sin(_AUV_HDG[0]),
                                          np.cos(_AUV_HDG[0]))) % 360)
        self.lbl_state.config(
            text=f"换能器深度 {depth:.1f} m    航速 {s.cfg.auv.speed:.2f} m/s    "
                 f"航向 {hdg:.0f}°    水深 {pt['H']:.1f} m")
        self.compass.draw(hdg, pt["dir"], pt["speed"], vmax)

        self.lbl_speed.config(text=f"{pt['speed']:.3f} m/s")
        self.lbl_dir.config(text=f"流向 {pt['dir']:.0f}°（{dir_name(pt['dir'])}）")
        self.lbl_uv.config(text=f"u = {pt['u']:+.3f} m/s        v = {pt['v']:+.3f} m/s")
        self.lbl_cmp.config(
            text=f"真值 {pt['speed_true']:.3f} m/s（{dir_name(pt['dir_true'])}）    "
                 f"偏差 {pt['speed'] - pt['speed_true']:+.3f} m/s")
        self.speedbar.draw(pt["speed"], max(vmax, pt["speed"] * 1.1))

        z, se, stv = s.profile_at(x0, y0, nz=24, t_s=self.clock_t)
        self.profile.draw(z, se, stv)
        depths = s.platform_depths(self.epoch)
        self.depthsec.draw(st["is_auv"], depths, s.bathy(st["x"], st["y"]))

        if s.result is not None:
            met = s.metrics(n=15)
            self.chip_rmse.config(text=f"场 RMSE {met['rmse']:.3f} m/s")
            self.lbl_quality.config(
                text=f"使用链路 {s.result['m']} 条（第 {self.epoch + 1}/"
                     f"{s.cfg.auv.epochs.size} 历元）    候选链路 {s.library.n} 条\n"
                     f"场 RMSE {met['rmse']:.3f} m/s    平均方向误差 {met['dir_err']:.1f}°\n"
                     f"空间相关系数 {met['corr']:+.2f}    数据残差 "
                     f"{s.result['rms_res']*1e6:.1f} µs\n"
                     f"字典表示下限 {s.floor_rms:.3f} m/s（场 rms {s.field_rms:.3f}）")
            self.prog["value"] = (self.epoch + 1) / s.cfg.auv.epochs.size
        self.spark.draw(self._speed_hist, self._truth_hist)

        if self.pick_xy is not None:
            q = s.sample_point(self.pick_xy[0], self.pick_xy[1],
                               float(self.display_depth.get()), self.clock_t)
            self.lbl_hint.config(
                text=f"取值点 ({q['x']/1000:.2f}, {q['y']/1000:.2f}) km：估计 "
                     f"{q['speed']:.3f} m/s / {q['dir']:.0f}°，真值 {q['speed_true']:.3f} m/s")
        else:
            self.lbl_hint.config(text="点击地图任意点可读取该处流速")

        self.lbl_clock.config(text=f"t = {st['t']:.0f} s")
        self.status.config(
            text=f"历元 {self.epoch + 1}/{s.cfg.auv.epochs.size}    "
                 f"累计链路 {self._links_upto(self.epoch).size}    "
                 f"候选链路 {s.library.n}    策略 {self.policy.get()}    "
                 f"{'运行中' if self.running else '就绪'}"
                 f"        快捷键：空格=开始/暂停   →=单步   ←=回退   R=重算")


def run(survey: Survey | None = None, truth: bool = False):
    """启动界面。"""
    app = App(survey)
    if truth:
        app.field_truth.set(True)
    app.mainloop()
    return app
