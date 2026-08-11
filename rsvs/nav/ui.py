"""导航演示自研 Viser 交互层（软著交互主体）。

从 vendor NavViserUI 迁出并增强：主题、热键、相机、Lidar/栅格可视化。
仿真/规划算法仍由 nav_mod（vendor）提供。
"""

from __future__ import annotations

import math
import threading
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from newtest.common.viser_camera import CameraController
from newtest.common.viser_hotkeys import HotkeySpec, hotkey_help_markdown, register_hotkeys
from newtest.common.viser_theme import apply_product_theme


class NavViserUI:
    """Viser 网页交互：点地设终点、热键控流程、传感器可视化。"""

    SCENE_KEYS = ("corridor", "fence", "open8")

    def __init__(self, nav_mod, port=8082, scene_name="corridor", nav_speed=0.55):
        try:
            import viser
        except ImportError as exc:
            raise ImportError("需要安装 viser：pip install viser") from exc

        self._m = nav_mod
        self._viser = viser
        self.server = viser.ViserServer(port=port)
        try:
            from newtest.common.viser_lifecycle import attach_exit_when_browser_closed

            attach_exit_when_browser_closed(self.server, grace_sec=8.0, label="nav")
        except Exception as e:
            print(f"  [viser] 未能挂载关页退出: {e}", flush=True)

        scene_label = nav_mod.NAV_SCENES.get(scene_name, {}).get("label", scene_name)
        apply_product_theme(
            self.server,
            module_label="导航演示",
            module_id="nav",
            dark_mode=True,
            scene_info=f"{scene_name}- ({scene_label})",
            status_text="STATUS: IDLE [待机]",
        )

        self._lock = threading.Lock()
        self.goal_xy = None
        self.navigating = False
        self.goal_revision = 0
        self.stop_requested = False
        self.restart_requested = False
        self.reset_requested = False
        self.pending_scene = None
        self.nav_speed = float(nav_speed)
        self._status = "请点击地面或输入坐标设置终点，再按「开始导航」/空格"
        self._path_line = None
        self._robot_handle = None
        self._body_handles = {}
        self._link_name_to_idx = None
        self._goal_handle = None
        self._goal_halo = None
        self._goal_arrow = None
        self._obstacle_handles = []
        self._lidar_cloud = None
        self._occ_cloud = None
        self._show_lidar = True
        self._show_occupancy = False
        self._show_path = True
        self._robot_xy = nav_mod.NAV_SCENES[scene_name]["start"]
        self._distance_m = None
        self._scene_label = scene_name
        self._auto_start_on_click = False
        self._bookmarked_goal = None
        self._path_len_m = None

        self._camera = CameraController(
            self.server,
            folder_label="VIEW",
            tracking=False,
            offset=np.array([-2.2, -3.0, 3.2], dtype=np.float64),
            look_offset=np.array([1.5, 1.5, 0.2], dtype=np.float64),
            fov_deg=55.0,
            add_gui=False,
        )

        self._build_scene(scene_name)
        self._build_gui(scene_name)
        self._bind_click()
        self._bind_hotkeys()

        host = self.server.get_host()
        print(f"  [viser] 导航交互（自研层）：http://{host}:{port}")
        print("  [viser] 点地设终点 · 空格开停 · C 清除 · R 重置 · ↑↓ 调速")

    # ── 场景 ──────────────────────────────────────────────
    def _build_scene(self, scene_name):
        m = self._m
        cfg = m.NAV_SCENES[scene_name]
        xmin, xmax, ymin, ymax = m.NAV_GRID_BOUNDS
        gx0, gx1, gy0, gy1 = m.NAV_GOAL_BOUNDS
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        span = max(xmax - xmin, ymax - ymin)

        self.server.scene.add_grid(
            "/ground",
            width=span + 1.0,
            height=span + 1.0,
            position=(cx, cy, 0.0),
            cell_color=(40, 120, 160),
            section_color=(0, 200, 230),
            cell_thickness=0.8,
            section_thickness=1.6,
            plane_color=(8, 14, 22),
            plane_opacity=0.35,
        )
        self.server.scene.add_box(
            "/goal_region",
            color=(40, 255, 120),
            dimensions=(max(0.2, gx1 - gx0), max(0.2, gy1 - gy0), 0.01),
            position=(0.5 * (gx0 + gx1), 0.5 * (gy0 + gy1), 0.005),
            opacity=0.18,
            wireframe=True,
        )

        z = m.NAV_OBSTACLE_H * 0.5
        self._obstacle_handles = []
        for i, wall in enumerate(cfg["walls"]):
            self._obstacle_handles.append(
                self.server.scene.add_box(
                    f"/obstacles/wall_{i}",
                    color=(120, 120, 130),
                    dimensions=tuple(wall["size"]),
                    position=(wall["pos"][0], wall["pos"][1], z),
                )
            )
        for i, (px, py) in enumerate(cfg["pillars"]):
            self._obstacle_handles.append(
                self.server.scene.add_box(
                    f"/obstacles/pillar_{i}",
                    color=(90, 90, 100),
                    dimensions=(m.NAV_PILLAR_SIZE, m.NAV_PILLAR_SIZE, m.NAV_OBSTACLE_H),
                    position=(px, py, z),
                )
            )

        sx, sy = cfg["start"]
        self.server.scene.add_icosphere(
            "/start_marker",
            radius=0.08,
            color=(80, 180, 255),
            position=(sx, sy, 0.08),
        )
        self._add_robot_visuals(sx, sy)

        self._goal_halo = self.server.scene.add_icosphere(
            "/goal_halo",
            radius=0.22,
            color=(255, 220, 40),
            opacity=0.35,
            position=(0.0, 0.0, 0.10),
            visible=False,
        )
        self._goal_handle = self.server.scene.add_icosphere(
            "/goal_marker",
            radius=0.10,
            color=(40, 255, 80),
            position=(0.0, 0.0, 0.12),
            visible=False,
        )
        self._goal_arrow = self.server.scene.add_box(
            "/goal_stem",
            color=(40, 255, 80),
            dimensions=(0.04, 0.04, 0.35),
            position=(0.0, 0.0, 0.28),
            visible=False,
        )

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (sx - 2.2, sy - 3.0, 3.2)
            client.camera.look_at = (sx + 1.5, sy + 1.5, 0.2)

    def _add_robot_visuals(self, sx, sy):
        body_meshes = self._m._load_go2_body_meshes()
        self._body_handles = {}
        self._robot_handle = None
        if body_meshes:
            for body_name, mesh in body_meshes.items():
                handle = self.server.scene.add_mesh_trimesh(
                    f"/robot/{body_name}",
                    mesh,
                    cast_shadow=True,
                    receive_shadow=True,
                )
                handle.position = (sx, sy, 0.35)
                self._body_handles[body_name] = handle
            return
        self._robot_handle = self.server.scene.add_box(
            "/robot",
            color=(40, 160, 255),
            dimensions=(0.35, 0.22, 0.28),
            position=(sx, sy, 0.20),
        )

    @staticmethod
    def _scene_option_label(key):
        return f"{key} — "

    def _scene_option_label_full(self, key):
        return f"{key} — {self._m.NAV_SCENES[key]['label']}"

    @staticmethod
    def _scene_key_from_option(option):
        return option.split(" — ", 1)[0].strip()

    # ── GUI ───────────────────────────────────────────────
    def _build_gui(self, scene_name):
        from newtest.common.viser_hud import card_head, gauge_svg, log_box, meter_html

        m = self._m
        cfg = m.NAV_SCENES[scene_name]
        gx, gy = cfg["goal"]
        xmin, xmax, ymin, ymax = m.NAV_GOAL_BOUNDS

        self._scene_label = scene_name
        self.reset_requested = False

        # 五卡底坞（对齐参考图模块拆分）
        with self.server.gui.add_folder("SCENE SELECTION", expand_by_default=True):
            scene_opts = tuple(self._scene_option_label_full(k) for k in self.SCENE_KEYS)
            self._scene_dd = self.server.gui.add_dropdown(
                "场景",
                options=scene_opts,
                initial_value=self._scene_option_label_full(scene_name),
            )
            # Hidden GUI labels for custom-dock proxy
            for key in self.SCENE_KEYS:
                btn_sc = self.server.gui.add_button(f"SCENE {key}", color="#3aa0a8")

                def _make_scene(k=key, button=btn_sc):
                    @button.on_click
                    def _(_e, scene_key=k):
                        try:
                            self._scene_dd.value = self._scene_option_label_full(scene_key)
                        except Exception:
                            pass
                        self.pending_scene = scene_key
                        self._set_status(f"已选择场景 {scene_key}，可 LOAD SCENE")

                _make_scene()

            btn_load_scene = self.server.gui.add_button("LOAD SCENE", color="#7c5cff")
            btn_reset = self.server.gui.add_button("RESET", color="orange")

            @btn_load_scene.on_click
            def _(_e):
                self._request_scene_load()

            @btn_reset.on_click
            def _(_e):
                self._request_reset()

        with self.server.gui.add_folder("VELOCITY CONTROL", expand_by_default=True):
            self.server.gui.add_html(card_head("Velocity Control"))
            self._speed_num = self.server.gui.add_number(
                "导航速度 (m/s)",
                initial_value=float(self.nav_speed),
                min=0.10,
                max=1.00,
                step=0.05,
            )
            self._speed_slider = self.server.gui.add_slider(
                "速度",
                min=0.10,
                max=1.00,
                step=0.05,
                initial_value=float(self.nav_speed),
            )
            lvl = int(round((float(self.nav_speed) - 0.1) / 0.9 * 15)) + 1
            self._meter_html = self.server.gui.add_html(meter_html(level=lvl))
            btn_slow = self.server.gui.add_button("SLOW", color="#3aa0a8")
            btn_mid = self.server.gui.add_button("NORMAL", color="cyan")
            btn_fast = self.server.gui.add_button("FAST", color="green")

            @btn_slow.on_click
            def _(_e):
                self._apply_speed_preset(0.30)

            @btn_mid.on_click
            def _(_e):
                self._apply_speed_preset(0.65)

            @btn_fast.on_click
            def _(_e):
                self._apply_speed_preset(1.00)

            @self._speed_num.on_update
            def _(_e):
                self._sync_speed(float(self._speed_num.value), from_num=True)

            @self._speed_slider.on_update
            def _(_e):
                self._sync_speed(float(self._speed_slider.value), from_num=False)

        with self.server.gui.add_folder("TARGET COORDINATES", expand_by_default=True):
            self.server.gui.add_html(card_head("Target Coordinates"))
            self._goal_vec = self.server.gui.add_vector2(
                "坐标 (x, y)",
                initial_value=(float(gx), float(gy)),
                min=(float(xmin), float(ymin)),
                max=(float(xmax), float(ymax)),
                step=0.05,
            )
            self._gauge_html = self.server.gui.add_html(
                gauge_svg(value=0.2, label="DIST", accent="#00d2e6")
            )
            self._preset = None
            if scene_name == "corridor":
                presets = ["(自定义)"] + [g["id"] for g in m.NAV_CORRIDOR_GOALS]
                self._preset = self.server.gui.add_dropdown(
                    "corridor 预设",
                    options=tuple(presets),
                    initial_value="(自定义)",
                )

                @self._preset.on_update
                def _(_e):
                    if self._preset.value == "(自定义)":
                        return
                    g = m._corridor_goal_cfg(self._preset.value)["goal"]
                    self._goal_vec.value = (float(g[0]), float(g[1]))
                    self.preview_goal(g, source=f"预设:{self._preset.value}")

            self._cb_auto = self.server.gui.add_checkbox("点地后自动开始", initial_value=False)

            @self._cb_auto.on_update
            def _(_e):
                self._auto_start_on_click = bool(self._cb_auto.value)

        with self.server.gui.add_folder("ACTION COMMANDS", expand_by_default=True):
            self.server.gui.add_html(card_head("Action Commands"))
            btn_preview = self.server.gui.add_button("PREVIEW ENDPOINT", color="cyan")
            btn_go = self.server.gui.add_button("START NAV", color="green")
            btn_stop = self.server.gui.add_button("STOP", color="red")
            btn_clear = self.server.gui.add_button("CLEAR ENDPOINT", color="#3d7dff")
            btn_bookmark = self.server.gui.add_button("BOOKMARK")
            btn_goto_bm = self.server.gui.add_button("GOTO BOOKMARK", color="#7c5cff")

            @btn_preview.on_click
            def _(_e):
                self.preview_goal(self._goal_vec.value, source="输入坐标")

            @btn_go.on_click
            def _(_e):
                self.start_navigation(self._goal_vec.value)

            @btn_stop.on_click
            def _(_e):
                self._stop_nav()

            @btn_clear.on_click
            def _(_e):
                self._clear_goal()

            @btn_bookmark.on_click
            def _(_e):
                self._bookmark_goal()

            @btn_goto_bm.on_click
            def _(_e):
                self._goto_bookmark()

        with self.server.gui.add_folder("SYSTEM STATUS LOG", expand_by_default=True):
            self.server.gui.add_html(card_head("System Status Log"))
            self._status_md = self.server.gui.add_markdown(self._status_block())
            self._log_html = self.server.gui.add_html(
                log_box(
                    [
                        "> link established",
                        "> lidar stream online",
                        "> awaiting target",
                        "> Space start/stop · T/Y speed",
                    ]
                )
            )
            cb_lidar = self.server.gui.add_checkbox("Lidar", initial_value=True)
            cb_occ = self.server.gui.add_checkbox("Occupancy", initial_value=False)
            cb_path = self.server.gui.add_checkbox("Path", initial_value=True)

            @cb_lidar.on_update
            def _(_e):
                self._show_lidar = bool(cb_lidar.value)
                if not self._show_lidar and self._lidar_cloud is not None:
                    try:
                        self._lidar_cloud.visible = False
                    except Exception:
                        pass

            @cb_occ.on_update
            def _(_e):
                self._show_occupancy = bool(cb_occ.value)
                if not self._show_occupancy and self._occ_cloud is not None:
                    try:
                        self._occ_cloud.visible = False
                    except Exception:
                        pass

            @cb_path.on_update
            def _(_e):
                self._show_path = bool(cb_path.value)
                if self._path_line is not None:
                    try:
                        self._path_line.visible = self._show_path
                    except Exception:
                        pass
            self._cb_lidar = cb_lidar
            btn_lidar = self.server.gui.add_button("TOGGLE LIDAR", color="#3aa0a8")

            @btn_lidar.on_click
            def _(_e):
                self._toggle_lidar_vis()

    def _apply_speed_preset(self, v: float) -> None:
        self._speed_num.value = float(v)
        self._speed_slider.value = float(v)
        self._sync_speed(float(v), from_num=True)

    def _bookmark_goal(self) -> None:
        if self.goal_xy is None:
            self._set_status("没有可收藏的终点")
            return
        self._bookmarked_goal = tuple(self.goal_xy)
        self._set_status(
            f"已收藏终点 ({self._bookmarked_goal[0]:.2f}, {self._bookmarked_goal[1]:.2f})"
        )

    def _goto_bookmark(self) -> None:
        if self._bookmarked_goal is None:
            self._set_status("尚未收藏终点")
            return
        self.preview_goal(self._bookmarked_goal, source="收藏终点")
        self.start_navigation()

    def _toggle_lidar_vis(self) -> None:
        self._show_lidar = not self._show_lidar
        try:
            self._cb_lidar.value = self._show_lidar
        except Exception:
            pass
        if self._lidar_cloud is not None:
            try:
                self._lidar_cloud.visible = self._show_lidar
            except Exception:
                pass
        self._set_status(f"Lidar 点云：{'开' if self._show_lidar else '关'}")

    def _bind_hotkeys(self):
        specs = [
            HotkeySpec("开始或停止导航", self._toggle_nav, "space", description="Space"),
            HotkeySpec("清除终点", self._clear_goal, "C", description="清除终点"),
            HotkeySpec("重置位姿与建图", self._request_reset, "R", description="重置"),
            HotkeySpec("提高导航速度", lambda: self._nudge_speed(0.05), "Y", description="速度 +（Y）"),
            HotkeySpec("降低导航速度", lambda: self._nudge_speed(-0.05), "T", description="速度 -（T）"),
            HotkeySpec("开关 Lidar 显示", self._toggle_lidar_vis, "V", description="Lidar"),
            HotkeySpec("预设终点 1", lambda: self._apply_corridor_preset(0), "1", description="corridor 预设"),
            HotkeySpec("预设终点 2", lambda: self._apply_corridor_preset(1), "2", description="corridor 预设"),
            HotkeySpec("预设终点 3", lambda: self._apply_corridor_preset(2), "3", description="corridor 预设"),
            HotkeySpec("预设终点 4", lambda: self._apply_corridor_preset(3), "4", description="corridor 预设"),
        ]
        register_hotkeys(self.server, specs)

    def _bind_click(self):
        @self.server.scene.on_click()
        def _(event):
            hit = self._m._ray_hit_ground(event.ray_origin, event.ray_direction, z_plane=0.0)
            if hit is None:
                return
            self.preview_goal(hit, source="点击地面", robot_xy=self._robot_xy)
            if self._auto_start_on_click:
                self.start_navigation()

    # ── 动作 ──────────────────────────────────────────────
    def _sync_speed(self, v: float, *, from_num: bool) -> None:
        self.nav_speed = float(v)
        try:
            if from_num and abs(float(self._speed_slider.value) - v) > 1e-6:
                self._speed_slider.value = v
            if not from_num and abs(float(self._speed_num.value) - v) > 1e-6:
                self._speed_num.value = v
        except Exception:
            pass
        try:
            from newtest.common.viser_hud import meter_html

            lvl = int(round((float(v) - 0.1) / 0.9 * 15)) + 1
            self._meter_html.content = meter_html(level=max(1, min(16, lvl)))
        except Exception:
            pass
        self._set_status(f"导航速度已设为 {v:.2f} m/s")

    def _set_status(self, text):
        from newtest.common.viser_hud import gauge_svg, log_box

        self._status = text
        if self.goal_xy is not None and self._robot_xy is not None:
            dx = self.goal_xy[0] - self._robot_xy[0]
            dy = self.goal_xy[1] - self._robot_xy[1]
            self._distance_m = math.hypot(dx, dy)
        try:
            self._status_md.content = self._status_block()
        except Exception:
            pass
        # 更新仪表与日志
        try:
            dist_n = 0.0
            if self._distance_m is not None:
                dist_n = max(0.0, min(1.0, 1.0 - self._distance_m / 8.0))
            self._gauge_html.content = gauge_svg(value=dist_n, label="DIST", accent="#00d2e6")
        except Exception:
            pass
        try:
            lines = [
                f"> {text}",
                f"> speed {self.nav_speed:.2f} m/s",
                f"> nav={'RUN' if self.navigating else 'IDLE'}",
                "> camera: WASD / arrows reserved",
            ]
            self._log_html.content = log_box(lines)
        except Exception:
            pass
        # 顶栏状态灯
        try:
            # 无法直接改 DOM，状态已在面板日志体现
            pass
        except Exception:
            pass

    def _nudge_speed(self, delta: float) -> None:
        v = max(0.10, min(1.00, float(self.nav_speed) + delta))
        self._speed_num.value = v
        self._speed_slider.value = v
        self._sync_speed(v, from_num=True)

    def _request_scene_load(self) -> None:
        key = self._scene_key_from_option(self._scene_dd.value)
        if key not in self._m.NAV_SCENES:
            self._set_status(f"未知场景：{self._scene_dd.value}")
            return
        if key == self._scene_label:
            self._set_status(f"当前已是场景 {key}，无需重新加载")
            return
        with self._lock:
            self.pending_scene = key
            self.restart_requested = True
            self.navigating = False
        self._set_status(f"正在切换到 {key}，请稍候…")

    def _request_reset(self) -> None:
        with self._lock:
            self.reset_requested = True
            self.navigating = False
            self.stop_requested = True
            self.goal_xy = None
            self.goal_revision += 1
        self._hide_goal()
        self.update_path(None)
        self._set_status("已请求重置：机器狗将回到起点")

    def _stop_nav(self) -> None:
        with self._lock:
            self.navigating = False
            self.stop_requested = True
        self._set_status("已停止，可重新设置终点")

    def _toggle_nav(self) -> None:
        if self.navigating:
            self._stop_nav()
        else:
            self.start_navigation()

    def _clear_goal(self) -> None:
        with self._lock:
            self.goal_xy = None
            self.navigating = False
            self.stop_requested = True
            self.goal_revision += 1
        self._hide_goal()
        self._set_status("终点已清除")

    def _apply_corridor_preset(self, index: int) -> None:
        if self._scene_label != "corridor":
            self._set_status("预设终点仅在 corridor 场景可用")
            return
        goals = self._m.NAV_CORRIDOR_GOALS
        if index < 0 or index >= len(goals):
            return
        g = goals[index]
        self.preview_goal(g["goal"], source=f"预设:{g['id']}")
        if self._preset is not None:
            try:
                self._preset.value = g["id"]
            except Exception:
                pass

    def _hide_goal(self) -> None:
        self._goal_handle.visible = False
        self._goal_halo.visible = False
        self._goal_arrow.visible = False

    def _status_block(self) -> str:
        m = self._m
        label = m.NAV_SCENES.get(self._scene_label, {}).get("label", getattr(self, "_scene_label", ""))
        goal = self.goal_xy
        goal_txt = "无" if goal is None else f"({goal[0]:.2f}, {goal[1]:.2f})"
        nav = "导航中" if self.navigating else "待机"
        dist = "—" if self._distance_m is None else f"{self._distance_m:.2f} m"
        bm = "无" if self._bookmarked_goal is None else (
            f"({self._bookmarked_goal[0]:.2f}, {self._bookmarked_goal[1]:.2f})"
        )
        return (
            f"**场景**：{getattr(self, '_scene_label', '')}（{label}）  |  **状态**：{nav}  |  "
            f"**速度**：{self.nav_speed:.2f} m/s\n\n"
            f"**当前终点**：{goal_txt}  |  **距离**：{dist}  |  **收藏**：{bm}\n\n{self._status}"
        )

    def _set_status(self, text):
        self._status = text
        if self.goal_xy is not None and self._robot_xy is not None:
            dx = self.goal_xy[0] - self._robot_xy[0]
            dy = self.goal_xy[1] - self._robot_xy[1]
            self._distance_m = math.hypot(dx, dy)
        self._status_md.content = self._status_block()

    def preview_goal(self, xy, source="坐标", robot_xy=None):
        raw = (float(xy[0]), float(xy[1]))
        ref = robot_xy if robot_xy is not None else self._robot_xy
        projected, clamped, raw_xy = self._m._project_goal_into_world(raw, robot_xy=ref)
        x, y = projected
        with self._lock:
            self.goal_xy = (x, y)
            self.goal_revision += 1
        try:
            self._goal_vec.value = (float(x), float(y))
        except Exception:
            pass
        pos = (x, y, 0.12)
        self._goal_handle.position = pos
        self._goal_handle.visible = True
        self._goal_halo.position = (x, y, 0.08)
        self._goal_halo.visible = True
        self._goal_arrow.position = (x, y, 0.28)
        self._goal_arrow.visible = True
        if clamped:
            self._set_status(
                f"终点在地图外，已投影："
                f"({raw_xy[0]:.2f},{raw_xy[1]:.2f}) → ({x:.2f},{y:.2f})，按空格开始"
            )
        else:
            self._set_status(f"已设置终点（{source}）：({x:.2f}, {y:.2f})，按空格开始")
        return projected, clamped

    def start_navigation(self, xy=None):
        if xy is not None:
            self.preview_goal(xy, source="开始导航", robot_xy=self._robot_xy)
        with self._lock:
            if self.goal_xy is None:
                self._set_status("请先设置终点（点击或输入）")
                return
            self.navigating = True
            self.stop_requested = False
            self.goal_revision += 1
            g = self.goal_xy
        self._set_status(f"开始前往 ({g[0]:.2f}, {g[1]:.2f})")

    def poll(self):
        try:
            v_num = float(self._speed_num.value)
            v_sld = float(self._speed_slider.value)
            if abs(v_num - v_sld) < 1e-3:
                self.nav_speed = v_num
            else:
                self.nav_speed = (
                    v_sld if abs(v_sld - self.nav_speed) >= abs(v_num - self.nav_speed) else v_num
                )
                self._speed_num.value = self.nav_speed
                self._speed_slider.value = self.nav_speed
        except Exception:
            pass
        with self._lock:
            return {
                "goal_xy": self.goal_xy,
                "navigating": self.navigating,
                "goal_revision": self.goal_revision,
                "stop_requested": self.stop_requested,
                "nav_speed": float(self.nav_speed),
                "restart_requested": self.restart_requested,
                "pending_scene": self.pending_scene,
                "reset_requested": self.reset_requested,
            }

    def consume_reset(self):
        with self._lock:
            self.reset_requested = False

    def mark_arrived(self):
        with self._lock:
            self.navigating = False
        g = self.goal_xy
        if g is None:
            self._set_status("已到达")
        else:
            self._set_status(f"已到达 ({g[0]:.2f}, {g[1]:.2f})，可设新终点继续")

    def update_robot(self, xy, yaw=0.0, z=0.20):
        self._robot_xy = (float(xy[0]), float(xy[1]))
        if self._camera.tracking:
            self._camera.tick((float(xy[0]), float(xy[1]), float(z)))
        if self._robot_handle is None:
            return
        half = yaw * 0.5
        self._robot_handle.position = (float(xy[0]), float(xy[1]), float(z))
        self._robot_handle.wxyz = (math.cos(half), 0.0, 0.0, math.sin(half))

    def update_robot_from_env(self, env):
        try:
            pos = env.base_pos[0].detach().cpu()
            self._robot_xy = (float(pos[0]), float(pos[1]))
            if self._camera.tracking:
                self._camera.tick((float(pos[0]), float(pos[1]), float(pos[2])))
        except Exception:
            pass
        if not self._body_handles:
            pos = env.base_pos[0].detach().cpu()
            self.update_robot(
                (float(pos[0]), float(pos[1])),
                yaw=self._m._world_yaw(env),
                z=float(pos[2]),
            )
            return

        try:
            links = env.robot.links
            if self._link_name_to_idx is None:
                self._link_name_to_idx = [lnk.name for lnk in links]
            link_pos = env.robot.get_links_pos()
            link_quat = env.robot.get_links_quat()
            if link_pos.ndim == 3:
                link_pos = link_pos[0]
                link_quat = link_quat[0]
            link_pos = link_pos.detach().cpu().numpy()
            link_quat = link_quat.detach().cpu().numpy()
        except Exception as exc:
            pos = env.base_pos[0].detach().cpu()
            self.update_robot(
                (float(pos[0]), float(pos[1])),
                yaw=self._m._world_yaw(env),
                z=float(pos[2]),
            )
            if not getattr(self, "_link_warn", False):
                print(f"  [viser] 连杆位姿读取失败，回退：{exc}")
                self._link_warn = True
            return

        with self.server.atomic():
            for i, name in enumerate(self._link_name_to_idx):
                handle = self._body_handles.get(name)
                if handle is None:
                    continue
                p = link_pos[i]
                q = link_quat[i]
                handle.position = (float(p[0]), float(p[1]), float(p[2]))
                handle.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def update_path(self, path):
        if self._path_line is not None:
            try:
                self._path_line.remove()
            except Exception:
                pass
            self._path_line = None
        if not path or len(path) < 2 or not self._show_path:
            return
        pts = np.array([[float(p[0]), float(p[1]), 0.07] for p in path], dtype=np.float32)
        self._path_line = self.server.scene.add_spline_catmull_rom(
            "/planned_path",
            points=pts,
            color=(40, 200, 255),
            line_width=3.0,
            segments=max(16, len(pts) * 2),
        )

    def update_lidar(self, scan_hits: Optional[Sequence]) -> None:
        """将 Lidar 命中画成点云。"""
        if not self._show_lidar:
            if self._lidar_cloud is not None:
                try:
                    self._lidar_cloud.visible = False
                except Exception:
                    pass
            return
        if not scan_hits:
            if self._lidar_cloud is not None:
                try:
                    self._lidar_cloud.visible = False
                except Exception:
                    pass
            return
        pts = []
        for hit in scan_hits:
            if hit is None:
                continue
            if len(hit) >= 3:
                pts.append([float(hit[0]), float(hit[1]), float(hit[2])])
            elif len(hit) >= 2:
                pts.append([float(hit[0]), float(hit[1]), 0.15])
        if not pts:
            return
        arr = np.asarray(pts, dtype=np.float32)
        colors = np.tile(np.array([[1.0, 0.55, 0.15]], dtype=np.float32), (len(arr), 1))
        if self._lidar_cloud is None:
            self._lidar_cloud = self.server.scene.add_point_cloud(
                "/lidar_hits",
                points=arr,
                colors=colors,
                point_size=0.04,
            )
        else:
            self._lidar_cloud.points = arr
            self._lidar_cloud.colors = colors
            self._lidar_cloud.visible = True

    def update_occupancy(self, grid) -> None:
        """占据栅格半透明点（采样，避免过密）。"""
        if not self._show_occupancy or grid is None:
            if self._occ_cloud is not None:
                try:
                    self._occ_cloud.visible = False
                except Exception:
                    pass
            return
        m = self._m
        pts = []
        try:
            width, height = m._grid_shape()
        except Exception:
            width, height = len(grid), len(grid[0]) if grid else (0, 0)
        step = max(1, min(width, height) // 40)
        for ix in range(0, width, step):
            for iy in range(0, height, step):
                cell = grid[ix][iy]
                if cell is not True:
                    continue
                try:
                    wx, wy = m._grid_to_world((ix, iy))
                except Exception:
                    continue
                pts.append([float(wx), float(wy), 0.06])
        if not pts:
            return
        arr = np.asarray(pts, dtype=np.float32)
        colors = np.tile(np.array([[0.55, 0.2, 0.85]], dtype=np.float32), (len(arr), 1))
        if self._occ_cloud is None:
            self._occ_cloud = self.server.scene.add_point_cloud(
                "/occupancy",
                points=arr,
                colors=colors,
                point_size=0.06,
            )
        else:
            self._occ_cloud.points = arr
            self._occ_cloud.colors = colors
            self._occ_cloud.visible = True

    def stop(self):
        try:
            self.server.stop()
        except Exception:
            pass
