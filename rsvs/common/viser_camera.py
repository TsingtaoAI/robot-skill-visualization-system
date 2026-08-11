"""相机跟随 / 视角预设 / FOV 公共控件。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# 预设：(camera_offset, look_offset) 相对机器人基座
CAMERA_PRESETS = {
    "第三人称": (
        np.array([2.4, 2.4, 1.5], dtype=np.float64),
        np.array([0.0, 0.0, 0.35], dtype=np.float64),
    ),
    "俯视": (
        np.array([0.15, 0.15, 4.5], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
    ),
    "侧面": (
        np.array([0.2, 3.2, 1.1], dtype=np.float64),
        np.array([0.0, 0.0, 0.3], dtype=np.float64),
    ),
}


class CameraController:
    """可挂到任意持有 server 的视图：跟随、FOV、预设、复位。"""

    def __init__(
        self,
        server,
        *,
        folder_label: str = "相机",
        tracking: bool = True,
        offset: Optional[np.ndarray] = None,
        look_offset: Optional[np.ndarray] = None,
        fov_deg: float = 55.0,
        add_gui: bool = True,
    ):
        self.server = server
        self.tracking = bool(tracking)
        self.offset = (
            np.asarray(offset, dtype=np.float64)
            if offset is not None
            else CAMERA_PRESETS["第三人称"][0].copy()
        )
        self.look_offset = (
            np.asarray(look_offset, dtype=np.float64)
            if look_offset is not None
            else CAMERA_PRESETS["第三人称"][1].copy()
        )
        self.fov_deg = float(fov_deg)
        self._base: Optional[np.ndarray] = None
        self._gui_built = False

        if add_gui:
            self.build_gui(folder_label=folder_label)

    def build_gui(self, *, folder_label: str = "相机") -> None:
        if self._gui_built:
            return
        self._gui_built = True
        with self.server.gui.add_folder(folder_label, expand_by_default=False):
            cb = self.server.gui.add_checkbox("跟随机器人", initial_value=self.tracking)

            @cb.on_update
            def _(_e) -> None:
                self.tracking = bool(cb.value)

            fov = self.server.gui.add_slider(
                "视场角 (°)",
                min=30.0,
                max=110.0,
                step=1.0,
                initial_value=self.fov_deg,
            )

            @fov.on_update
            def _(_e) -> None:
                self.fov_deg = float(fov.value)
                self._apply_fov()

            preset = self.server.gui.add_dropdown(
                "视角预设",
                options=tuple(CAMERA_PRESETS.keys()),
                initial_value="第三人称",
            )

            @preset.on_update
            def _(_e) -> None:
                self.apply_preset(str(preset.value))

            btn = self.server.gui.add_button("复位视角")

            @btn.on_click
            def _(_e) -> None:
                self.apply_preset("第三人称")
                self.tracking = True
                cb.value = True

    def apply_preset(self, name: str) -> None:
        pair = CAMERA_PRESETS.get(name)
        if pair is None:
            return
        self.offset = pair[0].copy()
        self.look_offset = pair[1].copy()
        if self._base is not None:
            self._push_camera(self._base)

    def set_base(self, base_xyz) -> None:
        if base_xyz is None:
            return
        self._base = np.asarray(base_xyz, dtype=np.float64).reshape(3)
        if self.tracking:
            self._push_camera(self._base)

    def tick(self, base_xyz) -> None:
        """主循环调用：更新基座位姿并在跟随时推相机。"""
        self.set_base(base_xyz)

    def _apply_fov(self) -> None:
        rad = float(np.radians(self.fov_deg))
        for client in self.server.get_clients().values():
            try:
                client.camera.fov = rad
            except Exception:
                pass

    def _push_camera(self, base: np.ndarray) -> None:
        pos = base + self.offset
        look = base + self.look_offset
        for client in self.server.get_clients().values():
            try:
                client.camera.position = pos
                client.camera.look_at = look
                client.camera.fov = float(np.radians(self.fov_deg))
            except Exception:
                pass


def attach_camera_to_robot_view(view, *, add_gui: bool = False) -> CameraController:
    """把 CameraController 接到 RobotViserView（可选跳过重复 GUI）。"""
    cam = CameraController(
        view.server,
        tracking=getattr(view, "_camera_tracking", True),
        offset=getattr(view, "_camera_offset", None),
        look_offset=getattr(view, "_camera_look_offset", None),
        add_gui=add_gui,
    )

    def _maybe_track(base) -> None:
        if base is None:
            return
        cam.tick(base)
        view._camera_tracking = cam.tracking
        view._camera_offset = cam.offset
        view._camera_look_offset = cam.look_offset

    view._maybe_track_camera = _maybe_track  # type: ignore[method-assign]
    view._camera_controller = cam  # type: ignore[attr-defined]
    return cam
