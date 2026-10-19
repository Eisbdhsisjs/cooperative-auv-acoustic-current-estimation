"""AUV 协同声学海流估计与"小蓝"显示界面.

本包实现论文 *Cooperative acoustic remote sensing of depth-dependent
horizontal currents using multiple AUVs* 的方法：

* 分层流函数表示（``model.Dictionary``）；
* 移动端点声学走时模型（``acoustics.ray_cycle`` / ``LinkLibrary``）；
* 信息驱动的链路选择（``planner.select_links``）；
* 参考时刻的场空间先验反演（``inverse.invert``）；
* 供界面调用的总控（``pipeline.Survey``）。
"""

__all__ = ["config", "model", "acoustics", "planner", "inverse", "pipeline"]
__version__ = "1.0.0"
