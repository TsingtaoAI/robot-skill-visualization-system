"""轻量 Viser 机器人视图：网格 + 相机跟随（供技能页 / 导航页复用）。"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


class RobotViserView:
    """在浏览器中显示 Go2，并用仿真连杆位姿驱动网格。"""

    def __init__(self, server, body_meshes: Optional[Dict] = None, *, add_camera_gui: bool = True):
        self.server = server
        self._body_handles: Dict[str, object] = {}
        self._link_names = None
        self._robot_xy: Optional[Tuple[float, float]] = None
        self._camera_tracking = True
        self._camera_offset = np.array([2.2, 2.2, 1.4], dtype=np.float64)
        self._camera_look_offset = np.array([0.0, 0.0, 0.35], dtype=np.float64)

        self.server.scene.add_grid(
            "/ground",
            infinite_grid=True,
            fade_distance=40.0,
            plane_opacity=0.35,
        )

        if not body_meshes:
            raise ValueError(
                "RobotViserView 需要真实 Go2 连杆网格；禁止方块占位。"
                "请调用 newtest.common.go2_meshes.load_go2_body_meshes()"
            )
        for name, mesh in body_meshes.items():
            self._body_handles[name] = self.server.scene.add_mesh_trimesh(
                f"/robot/{name}",
                mesh,
                cast_shadow=True,
                receive_shadow=True,
            )

        if add_camera_gui:
            with self.server.gui.add_folder("相机"):
                cb = self.server.gui.add_checkbox("跟随机器人", initial_value=True)

                @cb.on_update
                def _(_e) -> None:
                    self._camera_tracking = bool(cb.value)

    @property
    def robot_xy(self) -> Optional[Tuple[float, float]]:
        return self._robot_xy

    def update_from_robot(self, robot, env_idx: int = 0) -> None:
        """用 Genesis robot 实体更新网格。"""
        try:
            base = robot.get_pos()
            if hasattr(base, "ndim") and base.ndim == 2:
                base = base[env_idx]
            base = base.detach().cpu().numpy()
            self._robot_xy = (float(base[0]), float(base[1]))
        except Exception:
            base = None

        try:
            links = robot.links
            if self._link_names is None:
                self._link_names = [lnk.name for lnk in links]
            link_pos = robot.get_links_pos()
            link_quat = robot.get_links_quat()
            if link_pos.ndim == 3:
                link_pos = link_pos[env_idx]
                link_quat = link_quat[env_idx]
            link_pos = link_pos.detach().cpu().numpy()
            link_quat = link_quat.detach().cpu().numpy()
        except Exception:
            return

        with self.server.atomic():
            for i, name in enumerate(self._link_names):
                handle = self._body_handles.get(name)
                if handle is None:
                    continue
                p = link_pos[i]
                q = link_quat[i]
                handle.position = (float(p[0]), float(p[1]), float(p[2]))
                handle.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

        self._maybe_track_camera(base)

    def update_from_legged_env(self, env, robot_index: int = 0) -> None:
        """用 LeggedGym-Ex env（含 simulator）更新：优先 FK 网格，否则基座。"""
        sim = env.simulator
        base_pos = sim.base_pos[robot_index].detach().cpu().numpy()
        self._robot_xy = (float(base_pos[0]), float(base_pos[1]))

        # 若底层有 genesis robot，用连杆位姿
        robot = getattr(sim, "_robot", None)
        if robot is not None and self._body_handles:
            self.update_from_robot(robot, env_idx=robot_index)
            return

        self._maybe_track_camera(base_pos)

    def _maybe_track_camera(self, base) -> None:
        if not self._camera_tracking or base is None:
            return
        base = np.asarray(base, dtype=np.float64).reshape(3)
        for client in self.server.get_clients().values():
            client.camera.position = base + self._camera_offset
            client.camera.look_at = base + self._camera_look_offset


def _yaw_from_xyzw(q: np.ndarray) -> float:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
