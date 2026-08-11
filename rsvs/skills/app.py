"""技能页：Viser 按钮触发 newtest/vendor 内 combo 的高动态动作。

自包含：权重 / MJCF / plane / combo 脚本均在 newtest/ 内（相对路径），不索引外部仓库。

前脚直立：对齐专用 eval `go2_eval_handstand_gym_policy`（全程策略，无 combo 链式 HOLD/脚本恢复）。
其它技能：沿用 combo 的 run_*_phase + 短 HOLD 交接。

前摇来自原版默认：hs_startup/hs_ramp、jump_startup/jump_frame。

用法:
  ./newtest/run.sh skills --port 8081
"""

from __future__ import annotations

import newtest.bootstrap  # noqa: F401 — 设置 sys.path

import argparse
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import torch

from newtest.common.import_utils import ensure_sys_path, load_module_from_path
from newtest.common.viser_robot import RobotViserView
from newtest.paths import (
    DEFAULT_BACKFLIP,
    DEFAULT_BACKFLIP_DOUBLE,
    DEFAULT_HANDSTAND,
    DEFAULT_LEGSTAND_CYCLE,
    DEFAULT_SPRING_JUMP,
    GO2_SKILL_MJCF,
    LOCOMOTION_DIR,
    PLANE_URDF,
    VENDOR_COMBO_SCRIPT,
)
from newtest.skills import SKILL_BY_KEY, SKILLS


class SkillPhase(Enum):
    """UI 层阶段；实际力矩计算一律调用 combo.run_* / 原版 Phase。"""

    IDLE = "idle"
    # 原版 HOLD_QUAD / HOLD_QUAD_2 微交接（~hold_quad_s）
    HANDOFF_CYCLE = "handoff_cycle"
    HANDOFF_BACKFLIP = "handoff_backflip"
    HANDOFF_DOUBLE = "handoff_double"
    HANDOFF_JUMP = "handoff_jump"
    HANDSTAND = "handstand"
    # 按钮演示按专用 eval：全程策略，不做 combo 链式 HOLD/脚本趴下恢复
    LEGSTAND_CYCLE = "legstand_cycle"
    BACKFLIP = "backflip"
    SPRING_JUMP = "spring_jump"
    BACKFLIP_DOUBLE = "backflip_double"
    # 空翻落地后先用 flip PD（Genesis 关节序）站稳，再回软站立待机
    SETTLE_FLIP = "settle_flip"
    SETTLE = "settle"


@dataclass
class UIState:
    pending: Optional[str] = None
    reset_requested: bool = False
    abort_requested: bool = False
    status: str = "待机：点击按钮或按数字键触发技能"
    lock: threading.Lock = field(default_factory=threading.Lock)
    revision: int = 0
    panel: object = None  # StatusPanel，由 build_skill_gui 注入


def _resolve_model(path_str: str, combo) -> Path:
    p = Path(path_str).expanduser()
    if p.is_file():
        return p.resolve()
    for root in (LOCOMOTION_DIR, Path.cwd()):
        candidate = combo.resolve_path(path_str, root)
        if candidate.exists():
            return candidate
    return p.resolve()


def _load_go2_meshes_for_robot(robot):
    """加载与仿真机器人连杆对齐的可视化网格。

    仿真用 GO2_SKILL_MJCF；Viser 用轻量 STL。只保留 robot.links 中存在的连杆，
    避免 LeggedGym 的 Head_* / 多余 base 停在原点像「埋地狗头」。
    """
    from newtest.common.go2_meshes import load_go2_body_meshes
    from newtest.paths import GO2_MJCF

    meshes = load_go2_body_meshes(xml_path=GO2_MJCF)
    if "base" in meshes and "base_link" not in meshes:
        meshes["base_link"] = meshes.pop("base")
    else:
        meshes.pop("base", None)

    link_names = {lnk.name for lnk in robot.links}
    filtered = {name: mesh for name, mesh in meshes.items() if name in link_names}
    if not filtered:
        raise RuntimeError(
            f"网格与机器人连杆无交集。meshes={sorted(meshes)} links={sorted(link_names)}"
        )
    dropped = sorted(set(meshes) - set(filtered))
    if dropped:
        print(f"[mesh] 已丢弃无对应连杆的网格: {dropped}", flush=True)
    return filtered


def build_skill_gui(server, ui: UIState) -> None:
    from collections import deque

    from newtest.common.viser_theme import apply_product_theme
    from newtest.common.viser_hotkeys import HotkeySpec, register_hotkeys
    from newtest.common.viser_camera import CameraController
    from newtest.common.viser_hud import (
        action_hint,
        card_head,
        gauge_svg,
        log_box,
        meter_html,
    )

    apply_product_theme(
        server,
        module_label="技能演示",
        module_id="skills",
        dark_mode=True,
        scene_info="skill-arena",
        status_text="STATUS: IDLE [待机]",
    )
    ui.panel = None
    cam = CameraController(server, folder_label="VIEW PRESETS", tracking=True, add_gui=False)
    ui._last_skill_key = None  # type: ignore[attr-defined]
    ui._run_count = 0  # type: ignore[attr-defined]
    ui._queue_repeat = True  # type: ignore[attr-defined]
    ui._busy = False  # type: ignore[attr-defined]
    ui._progress = 0.0  # type: ignore[attr-defined]
    ui._logs = deque(maxlen=8)  # type: ignore[attr-defined]

    def _refresh_widgets() -> None:
        busy = bool(getattr(ui, "_busy", False))
        prog = float(getattr(ui, "_progress", 0.0))
        count = int(getattr(ui, "_run_count", 0))
        try:
            ui._gauge.content = gauge_svg(  # type: ignore[attr-defined]
                value=prog / 100.0,
                label="PROG",
                accent="#ff8c28",
            )
        except Exception:
            pass
        try:
            lvl = max(1, min(16, int(round(prog / 100.0 * 15)) + (1 if busy else 0)))
            ui._meter.content = meter_html(level=lvl)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            lines = [
                f"> {ui.status}",
                f"> lock={'BUSY' if busy else 'IDLE'}  runs={count}",
                f"> last={getattr(ui, '_last_skill_key', None) or '—'}",
                "> keys: 1-5 / R / Esc / F · cam WASD",
            ]
            for item in list(getattr(ui, "_logs", []))[-3:]:
                lines.append(f"> {item}")
            ui._log.content = log_box(lines)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            ui._status_md.content = (  # type: ignore[attr-defined]
                f"**{ui.status}**  ·  {'忙碌' if busy else '空闲'}  ·  累计 `{count}`"
            )
        except Exception:
            pass

    def set_status(text: str, *, busy: bool = False, progress: float = 0.0, log: Optional[str] = None) -> None:
        ui.status = text
        ui._busy = busy  # type: ignore[attr-defined]
        ui._progress = float(progress)  # type: ignore[attr-defined]
        if log:
            ui._logs.append(log)  # type: ignore[attr-defined]
        _refresh_widgets()

    ui._set_status = set_status  # type: ignore[attr-defined]
    ui._camera = cam  # type: ignore[attr-defined]

    def request_skill(key: str, label: str) -> None:
        with ui.lock:
            if ui.pending is not None:
                set_status(f"忙碌中，忽略：{label}", busy=True, log=f"忽略 {label}")
                return
            ui.pending = key
            ui.revision += 1
            ui._last_skill_key = key  # type: ignore[attr-defined]
            ui._run_count = int(getattr(ui, "_run_count", 0)) + 1  # type: ignore[attr-defined]
        set_status(f"已请求：{label}", busy=True, progress=8.0, log=f"请求 {label}")

    def request_reset() -> None:
        with ui.lock:
            ui.reset_requested = True
            ui.abort_requested = False
            ui.pending = None
            ui.revision += 1
        set_status("已请求复位", busy=True, progress=5.0, log="复位")

    def request_abort() -> None:
        with ui.lock:
            ui.abort_requested = True
            ui.pending = None
            ui.revision += 1
        set_status("急停：取消待执行并请求中断", busy=True, progress=0.0, log="急停 Esc")

    def repeat_last() -> None:
        key = getattr(ui, "_last_skill_key", None)
        if not key or key not in SKILL_BY_KEY:
            set_status("尚无历史技能可重放", busy=False, log="重放失败")
            return
        skill = SKILL_BY_KEY[key]
        request_skill(skill.key, skill.label)

    def take_snapshot() -> None:
        clients = list(server.get_clients().values())
        if not clients:
            set_status("无浏览器连接，无法截图", log="截图失败")
            return
        try:
            import numpy as np
            from PIL import Image
            import io

            arr = clients[0].get_render(height=480, width=640)
            if arr is None:
                set_status("截图为空", log="截图失败")
                return
            img = Image.fromarray(np.asarray(arr))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            clients[0].send_file_download("skill_snapshot.png", buf.getvalue())
            set_status("已导出场景截图 skill_snapshot.png", log="截图成功")
        except Exception as exc:
            try:
                arr = clients[0].get_render(height=480, width=640)
                clients[0].send_file_download("skill_snapshot.raw.txt", str(arr.shape).encode())
                set_status(f"截图降级导出（需 pillow）：{exc}", log="截图降级")
            except Exception as exc2:
                set_status(f"截图失败：{exc2}", log="截图失败")

    # ── 底坞：五技能横向一排 + 指令/状态 ────────────────
    with server.gui.add_folder("1-5 SKILLS", expand_by_default=True):
        for i, skill in enumerate(SKILLS):
            # Stable English ids for custom-dock click proxy (S1..S5)
            btn = server.gui.add_button(f"S{i+1}", color="#ff8c28")

            def _make_handler(key: str, label: str, button=btn):
                @button.on_click
                def _(_e, k=key, lab=label) -> None:
                    request_skill(k, lab)

            _make_handler(skill.key, skill.label)

    with server.gui.add_folder("ACTION COMMANDS", expand_by_default=True):
        server.gui.add_html(card_head("Commands"))
        btn_reset = server.gui.add_button("RESET", color="orange")
        btn_abort = server.gui.add_button("STOP", color="red")
        btn_repeat = server.gui.add_button("REPLAY", color="green")
        btn_shot = server.gui.add_button("SHOT", color="cyan")

        @btn_reset.on_click
        def _(_e) -> None:
            request_reset()

        @btn_abort.on_click
        def _(_e) -> None:
            request_abort()

        @btn_repeat.on_click
        def _(_e) -> None:
            repeat_last()

        @btn_shot.on_click
        def _(_e) -> None:
            take_snapshot()

    with server.gui.add_folder("VIEW", expand_by_default=True):
        server.gui.add_html(card_head("View"))
        cb_follow = server.gui.add_checkbox("Follow", initial_value=True)
        fov = server.gui.add_slider("FOV", min=30.0, max=110.0, step=1.0, initial_value=55.0)
        preset = server.gui.add_dropdown(
            "Preset",
            options=("第三人称", "俯视", "侧面"),
            initial_value="第三人称",
        )
        btn_cam = server.gui.add_button("RESET VIEW", color="cyan")
        btn_cam_lock = server.gui.add_button("CAM LOCK", color="green")
        btn_cam_free = server.gui.add_button("CAM FREE", color="#3aa0a8")

        @cb_follow.on_update
        def _(_e) -> None:
            cam.tracking = bool(cb_follow.value)

        @fov.on_update
        def _(_e) -> None:
            cam.fov_deg = float(fov.value)
            cam._apply_fov()

        @preset.on_update
        def _(_e) -> None:
            cam.apply_preset(str(preset.value))

        @btn_cam.on_click
        def _(_e) -> None:
            cam.apply_preset("第三人称")
            cam.tracking = True
            cb_follow.value = True

        @btn_cam_lock.on_click
        def _(_e) -> None:
            cam.tracking = True
            cb_follow.value = True
            set_status("相机已锁定（跟随机器人）", busy=False, log="CAM LOCK")

        @btn_cam_free.on_click
        def _(_e) -> None:
            cam.tracking = False
            cb_follow.value = False
            set_status("相机已解锁（自由视角 WASD/鼠标）", busy=False, log="CAM FREE")

    with server.gui.add_folder("METER", expand_by_default=True):
        server.gui.add_html(card_head("Meter"))
        ui._gauge = server.gui.add_html(gauge_svg(value=0.0, label="PROG", accent="#ff8c28"))  # type: ignore[attr-defined]
        ui._meter = server.gui.add_html(meter_html(level=1))  # type: ignore[attr-defined]
        ui._progress_bar = server.gui.add_progress_bar(value=0.0, animated=False)  # type: ignore[attr-defined]

    with server.gui.add_folder("STATUS LOG", expand_by_default=True):
        server.gui.add_html(card_head("Status"))
        ui._status_md = server.gui.add_markdown("**IDLE**")  # type: ignore[attr-defined]
        ui._log = server.gui.add_html(  # type: ignore[attr-defined]
            log_box(["> skill stage online", "> keys 1-5 / R / Esc / F", "> cam: WASD / arrows"])
        )

    # 进度条联动
    _orig_set = set_status

    def set_status_linked(text: str, *, busy: bool = False, progress: float = 0.0, log: Optional[str] = None) -> None:
        _orig_set(text, busy=busy, progress=progress, log=log)
        try:
            ui._progress_bar.value = float(progress)  # type: ignore[attr-defined]
            ui._progress_bar.animated = bool(busy)  # type: ignore[attr-defined]
        except Exception:
            pass

    set_status = set_status_linked  # type: ignore[assignment]
    ui._set_status = set_status  # type: ignore[attr-defined]

    hotkey_specs = [
        HotkeySpec("触发：前脚直立", lambda: request_skill("handstand", "前脚直立"), "1", description="技能 1"),
        HotkeySpec("触发：后腿站立循环", lambda: request_skill("legstand_cycle", "后腿站立循环"), "2", description="技能 2"),
        HotkeySpec("触发：后空翻", lambda: request_skill("backflip", "后空翻"), "3", description="技能 3"),
        HotkeySpec("触发：弹跳", lambda: request_skill("spring_jump", "弹跳"), "4", description="技能 4"),
        HotkeySpec("触发：连续后空翻", lambda: request_skill("backflip_double", "连续后空翻"), "5", description="技能 5"),
        HotkeySpec("复位到四足站立", request_reset, "R", description="硬复位"),
        HotkeySpec("急停", request_abort, "escape", description="中断到待机"),
        HotkeySpec("重放上一技能", repeat_last, "F", description="重放"),
    ]
    register_hotkeys(server, hotkey_specs)
    set_status(ui.status, busy=False)


def parse_args():
    p = argparse.ArgumentParser(description="技能可视化演示（Viser 按钮）")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--handstand_model", type=str, default=DEFAULT_HANDSTAND)
    p.add_argument("--legstand_cycle_ckpt", type=str, default=DEFAULT_LEGSTAND_CYCLE)
    p.add_argument("--backflip_model", type=str, default=DEFAULT_BACKFLIP)
    p.add_argument("--backflip_double_model", type=str, default=DEFAULT_BACKFLIP_DOUBLE)
    p.add_argument("--spring_jump_model", type=str, default=DEFAULT_SPRING_JUMP)
    p.add_argument("--jump_distance", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    # 再次确保路径（防止被其它 import 抢先加载错误的 genesis）
    from newtest.paths import setup_runtime_paths

    setup_runtime_paths()
    ensure_sys_path([LOCOMOTION_DIR])

    if not VENDOR_COMBO_SCRIPT.is_file():
        raise FileNotFoundError(f"缺少内置 combo 脚本: {VENDOR_COMBO_SCRIPT}")

    combo = load_module_from_path(
        "newtest_ext_combo",
        VENDOR_COMBO_SCRIPT,
        extra_sys_paths=[LOCOMOTION_DIR],
    )

    import genesis as gs
    from genesis.utils.geom import inv_quat, transform_by_quat

    if not hasattr(gs, "init"):
        raise ImportError(
            f"genesis 导入异常: file={getattr(gs, '__file__', None)}。"
            "请用 ./newtest/run.sh skills 启动（勿在会遮挡包路径的目录结构下裸跑）。"
        )
    combo.gs = gs
    combo.inv_quat = inv_quat
    combo.transform_by_quat = transform_by_quat

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")
    device = gs.device

    # 对齐 go2_backflip / go2_env 训练域：dt=0.02、无自碰、站立高度 0.35。
    # 实测 dt=0.005（combo 默认）下连续后空翻无法站稳；dt=0.02 + 恰好 3.0s 可站稳。
    _combo_dt_train = 0.005
    combo.CONTROL_DT = 0.02
    combo.SIM_SUBSTEPS = 2
    combo.DECIMATION = 1
    combo.POLICY_DT = combo.CONTROL_DT * combo.DECIMATION
    # 默认生成高度；各技能 hard_reset 再切到训练域高度
    combo.INIT_HEIGHT = 0.42
    _step_scale = _combo_dt_train / combo.CONTROL_DT  # 0.25

    SKILL_RESET = {
        "handstand": ("isaac", 0.42),
        "legstand_cycle": ("cycle", 0.42),
        "backflip": ("flip", 0.35),
        "backflip_double": ("flip", 0.35),
        "spring_jump": ("jump", 0.39),
    }

    # ── 策略 ──────────────────────────────────────────────
    handstand_policy = combo.load_policy(
        _resolve_model(args.handstand_model, combo), None, device, combo.OBS_STAND
    )
    cycle_policy = combo.load_policy(
        _resolve_model(args.legstand_cycle_ckpt, combo), None, device, combo.OBS_CYCLE
    )
    backflip_policy = combo.load_policy(
        _resolve_model(args.backflip_model, combo), None, device, combo.OBS_BACKFLIP
    )
    backflip_double_policy = combo.load_policy(
        _resolve_model(args.backflip_double_model, combo), None, device, combo.OBS_BACKFLIP
    )
    spring_policy = combo.load_policy(
        _resolve_model(args.spring_jump_model, combo), None, device, combo.OBS_JUMP
    )

    # 用 combo 的 argparse 默认构造一份 cfg 命名空间（阈值等）
    cfg = _make_combo_cfg(combo, args, step_scale=_step_scale)

    # ── 场景 ──────────────────────────────────────────────
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=combo.CONTROL_DT, substeps=cfg.sim_substeps),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_self_collision=False,  # 对齐 go2_env
            enable_joint_limit=True,
            tolerance=1e-5,
            max_collision_pairs=80,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, -2.8, 1.3),
            camera_lookat=(0.3, 0.0, 0.45),
            camera_fov=45,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.URDF(file=str(PLANE_URDF), fixed=True))
    if not PLANE_URDF.is_file():
        raise FileNotFoundError(f"缺少地面 URDF: {PLANE_URDF}")
    if not GO2_SKILL_MJCF.is_file():
        raise FileNotFoundError(f"缺少技能用 Go2 MJCF: {GO2_SKILL_MJCF}")
    print(f"[skills] combo 脚本(仅 newtest/vendor): {VENDOR_COMBO_SCRIPT}", flush=True)
    print(f"[skills] MJCF(与 My_unitree/combo 同源): {GO2_SKILL_MJCF}", flush=True)
    print(f"[skills] plane: {PLANE_URDF}", flush=True)
    print(f"[skills] handstand: {DEFAULT_HANDSTAND}", flush=True)
    print(f"[skills] legstand:  {DEFAULT_LEGSTAND_CYCLE}", flush=True)
    print(f"[skills] backflip:  {DEFAULT_BACKFLIP}", flush=True)
    print(f"[skills] jump:      {DEFAULT_SPRING_JUMP}", flush=True)
    print(f"[skills] flip2:     {DEFAULT_BACKFLIP_DOUBLE}", flush=True)
    print(
        f"[skills] 仿真: dt={combo.CONTROL_DT}, decimation={combo.DECIMATION}, "
        f"self_collision=False（对齐 cycle/backflip 训练域）",
        flush=True,
    )
    print(
        "[skills] 空翻/后腿站立：位置控制+动作延迟；弹跳/前脚直立：按原策略力矩 PD。",
        flush=True,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=str(GO2_SKILL_MJCF),
            pos=[0.0, 0.0, combo.INIT_HEIGHT],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
    )
    scene.build(n_envs=1)

    isaac_dofs = [robot.get_joint(name).dof_start for name in combo.JOINT_NAMES]
    genesis_dofs = [robot.get_joint(name).dof_start for name in combo.GENESIS_JOINT_NAMES]
    genesis_dofs_t = torch.tensor(genesis_dofs, dtype=torch.long, device=device)
    # go2_env: control_dofs_position(target[:, argsort(motors)], slice(6, 18))
    actions_dof_idx = torch.argsort(genesis_dofs_t)

    quad_default = combo.DEFAULT_QUAD_DOF.to(device)
    cycle_default = combo.CYCLE_QUAD_DOF.to(device)
    backflip_default = combo.BACKFLIP_QUAD_DOF.to(device)
    spring_default = combo.SPRING_JUMP_DOF.to(device)
    hs_desired_delta = (combo.HANDSTAND_DESIRED - combo.DEFAULT_QUAD_DOF).to(device).unsqueeze(0)
    hs_joints = combo.HANDSTAND_DESIRED.to(device)
    prone_joints = combo.PRONE_FORWARD_DOF.to(device)
    low_joints = combo.QUAD_LOW_DOF.to(device)
    quad_joints = combo.DEFAULT_QUAD_DOF.to(device)

    rear_kp_boost = torch.ones((1, combo.NUM_ACTIONS), dtype=torch.float32, device=device)
    rear_kp_boost[0, 6:] = cfg.recover_rear_kp_boost

    stand_kp = torch.full((1, combo.NUM_ACTIONS), combo.STAND_KP, device=device)
    stand_kd = torch.full((1, combo.NUM_ACTIONS), combo.STAND_KD * cfg.kd_scale, device=device)
    stand_tau = torch.full((1, combo.NUM_ACTIONS), combo.STAND_TAU, device=device)
    cycle_kp = torch.full((1, combo.NUM_ACTIONS), combo.CYCLE_KP, device=device)
    cycle_kd = torch.full((1, combo.NUM_ACTIONS), combo.CYCLE_KD, device=device)
    cycle_tau = torch.full((1, combo.NUM_ACTIONS), combo.CYCLE_TAU, device=device)
    flip_kp = torch.full((1, combo.NUM_ACTIONS), combo.BACKFLIP_KP, device=device)
    flip_kd = torch.full((1, combo.NUM_ACTIONS), combo.BACKFLIP_KD, device=device)
    flip_tau = torch.full((1, combo.NUM_ACTIONS), combo.BACKFLIP_TAU, device=device)
    jump_kp = torch.full((1, combo.NUM_ACTIONS), combo.JUMP_KP, device=device)
    jump_kd = torch.full((1, combo.NUM_ACTIONS), combo.JUMP_KD, device=device)
    jump_tau = torch.full((1, combo.NUM_ACTIONS), combo.JUMP_TAU, device=device)

    zero_cmd = torch.zeros(3, dtype=torch.float32, device=device)
    jump_distance = float(args.jump_distance if args.jump_distance is not None else combo.JUMP_DISTANCE)

    def hard_reset(skill_key: Optional[str] = None):
        """按技能训练域复位高度与默认关节。"""
        mode, height = SKILL_RESET.get(skill_key or "", ("isaac", 0.42))
        if mode == "flip":
            robot.set_dofs_position(
                backflip_default.unsqueeze(0), dofs_idx_local=genesis_dofs, zero_velocity=True
            )
        elif mode == "cycle":
            robot.set_dofs_position(
                cycle_default.unsqueeze(0), dofs_idx_local=genesis_dofs, zero_velocity=True
            )
        elif mode == "jump":
            robot.set_dofs_position(
                spring_default.unsqueeze(0), dofs_idx_local=isaac_dofs, zero_velocity=True
            )
        else:
            robot.set_dofs_position(
                quad_default.unsqueeze(0), dofs_idx_local=isaac_dofs, zero_velocity=True
            )
        robot.set_pos(
            torch.tensor([[0.0, 0.0, height]], dtype=gs.tc_float, device=device),
            zero_velocity=True,
        )
        robot.set_quat(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=gs.tc_float, device=device),
            zero_velocity=True,
        )

    hard_reset()

    # ── Viser ─────────────────────────────────────────────
    import viser

    server = viser.ViserServer(port=args.port)
    from newtest.common.viser_lifecycle import attach_exit_when_browser_closed

    attach_exit_when_browser_closed(server, grace_sec=8.0, label="skills")
    ui = UIState()
    build_skill_gui(server, ui)
    view = RobotViserView(
        server,
        body_meshes=_load_go2_meshes_for_robot(robot),
        add_camera_gui=False,
    )
    if hasattr(ui, "_camera") and ui._camera is not None:  # type: ignore[attr-defined]
        cam = ui._camera  # type: ignore[attr-defined]

        def _maybe_track(base) -> None:
            if base is None:
                return
            cam.tick(base)

        view._maybe_track_camera = _maybe_track  # type: ignore[method-assign]

    print(f"[skills] Viser: http://localhost:{args.port}", flush=True)
    print(
        "[skills] 前摇: handstand≈hs_startup+hs_ramp；弹跳≈jump_startup+jump_frame（专用/combo 原版默认）",
        flush=True,
    )
    print("[skills] 按钮或数字键 1–5 触发；R 复位；Esc 急停；关闭网页自动结束进程", flush=True)

    rt = combo.PhaseRuntime()
    rt.reset_stand_policy(device)
    rt.reset_genesis_policy(device)
    rt.reset_jump_policy(device)

    phase = SkillPhase.IDLE
    active_skill: Optional[str] = None
    settle_steps_left = 0
    settle_flip_max_steps = int(2.5 / combo.CONTROL_DT)
    last_policy_torques = torch.zeros((1, combo.NUM_ACTIONS), device=device)
    last_control_dofs = isaac_dofs
    last_pos_target = None
    use_pos_control = False
    flip_pd_ready = False
    cycle_pd_ready = False
    global_step = 0
    anchor_pg = None
    anchor_prog = None
    anchor_ang = None

    def begin_flip_settle(max_hold_s: float = 2.5) -> None:
        """空翻策略结束后：先用 flip KP/KD + backflip 默认姿态稳住，避免直接切软站立栽倒。"""
        nonlocal phase, settle_flip_max_steps
        phase = SkillPhase.SETTLE_FLIP
        settle_flip_max_steps = max(1, int(max_hold_s / combo.CONTROL_DT))
        rt.reset_phase_counters()
        rt.stable_steps = 0

    def go_idle(msg: str):
        nonlocal phase, active_skill, settle_steps_left, anchor_pg, anchor_prog, anchor_ang
        nonlocal use_pos_control, last_pos_target, flip_pd_ready, cycle_pd_ready
        phase = SkillPhase.IDLE
        active_skill = None
        settle_steps_left = 0
        anchor_pg = anchor_prog = anchor_ang = None
        use_pos_control = False
        last_pos_target = None
        flip_pd_ready = False
        cycle_pd_ready = False
        rt.reset_stand_policy(device)
        rt.reset_genesis_policy(device)
        rt.reset_jump_policy(device)
        rt.reset_phase_counters()
        if hasattr(ui, "_set_status"):
            ui._set_status(msg, busy=False, progress=0.0)  # type: ignore[attr-defined]
        print(f"[skills] {msg}", flush=True)

    from genesis.utils.geom import quat_to_xyz as _quat_to_xyz

    def start_skill(key: str):
        """切入技能前先 hard_reset，动作函数仍用 vendor combo / 与专用 eval 对齐的参数。"""
        nonlocal phase, active_skill, use_pos_control, last_pos_target, flip_pd_ready, cycle_pd_ready
        active_skill = key
        hard_reset(key)
        use_pos_control = False
        last_pos_target = None
        flip_pd_ready = False
        cycle_pd_ready = False
        rt.reset_phase_counters()
        if key == "handstand":
            # 对齐 go2_eval_handstand_gym_policy：全程 HANDSTAND，不进 HOLD/RECOVER
            rt.reset_stand_policy(device)
            rt.phase = combo.Phase.HANDSTAND
            phase = SkillPhase.HANDSTAND
        elif key == "legstand_cycle":
            rt.phase = combo.Phase.HOLD_QUAD
            phase = SkillPhase.HANDOFF_CYCLE
        elif key == "backflip":
            rt.phase = combo.Phase.HOLD_QUAD_2
            phase = SkillPhase.HANDOFF_BACKFLIP
        elif key == "spring_jump":
            rt.phase = combo.Phase.HOLD_BRIDGE
            phase = SkillPhase.HANDOFF_JUMP
        elif key == "backflip_double":
            rt.phase = combo.Phase.HOLD_QUAD_2
            phase = SkillPhase.HANDOFF_DOUBLE
        else:
            go_idle(f"未知技能: {key}")
            return
        skill_label = SKILL_BY_KEY[key].label if key in SKILL_BY_KEY else key
        if hasattr(ui, "_set_status"):
            ui._set_status(f"执行中：{skill_label}", busy=True, progress=25.0, log=f"执行 {skill_label}")  # type: ignore[attr-defined]
        print(f"[skills] start {key} (combo phase={rt.phase.value})", flush=True)

    try:
        with torch.no_grad():
            while True:
                # UI 事件
                with ui.lock:
                    reset_req = ui.reset_requested
                    abort_req = ui.abort_requested
                    pending = ui.pending
                    if reset_req:
                        ui.reset_requested = False
                    if abort_req:
                        ui.abort_requested = False
                    if pending is not None and phase == SkillPhase.IDLE:
                        ui.pending = None

                if reset_req:
                    hard_reset()
                    go_idle("已复位，待机")
                    pending = None

                if abort_req and not reset_req:
                    # 急停：清 pending，若在执行则硬复位回待机（不改 vendor 相位内部）
                    hard_reset()
                    go_idle("急停完成，待机")
                    pending = None

                if pending is not None and phase == SkillPhase.IDLE:
                    try:
                        start_skill(pending)
                    except Exception as exc:
                        print(f"[skills] 启动技能失败: {exc}")
                        go_idle(f"启动失败: {exc}")

                rt.phase_step += 1
                torques = last_policy_torques
                control_dofs = last_control_dofs
                use_pos_control = False
                anchor_pg = anchor_prog = anchor_ang = None

                if phase == SkillPhase.IDLE or phase == SkillPhase.SETTLE:
                    control_dofs = isaac_dofs
                    torques, _, _, _ = combo.run_hold_pose(
                        rt,
                        robot,
                        isaac_dofs,
                        quad_default,
                        quad_default.unsqueeze(0),
                        device,
                        stand_kp,
                        stand_kd,
                        stand_tau,
                        zero_cmd,
                    )
                    last_policy_torques = torques
                    if phase == SkillPhase.SETTLE:
                        settle_steps_left -= 1
                        if settle_steps_left <= 0:
                            go_idle("技能完成，待机")

                elif phase == SkillPhase.SETTLE_FLIP:
                    # 对齐 combo 单次空翻后 HOLD_QUAD_3：继续用 flip PD / Genesis 序站稳
                    control_dofs = genesis_dofs
                    torques, pg, ang, dof_pos = combo.run_hold_pose(
                        rt,
                        robot,
                        genesis_dofs,
                        backflip_default,
                        backflip_default.unsqueeze(0),
                        device,
                        flip_kp,
                        flip_kd,
                        flip_tau,
                        zero_cmd,
                    )
                    last_policy_torques = torques
                    ready = combo.is_cycle_landed(robot, pg, ang, cfg) or combo.is_transition_ready(
                        pg, dof_pos, backflip_default, ang, cfg
                    )
                    rt.stable_steps = rt.stable_steps + 1 if ready else 0
                    landed = rt.stable_steps >= cfg.transition_steps_req
                    timed_out = rt.phase_step >= settle_flip_max_steps
                    if landed or timed_out:
                        phase = SkillPhase.SETTLE
                        settle_steps_left = int(0.8 / combo.CONTROL_DT)
                        rt.reset_phase_counters()

                elif phase == SkillPhase.HANDOFF_CYCLE:
                    # 原版 HOLD_QUAD → LEGSTAND_CYCLE
                    control_dofs = genesis_dofs
                    torques, pg, ang, dof_pos = combo.run_hold_pose(
                        rt,
                        robot,
                        genesis_dofs,
                        cycle_default,
                        cycle_default.unsqueeze(0),
                        device,
                        cycle_kp,
                        cycle_kd,
                        cycle_tau,
                        zero_cmd,
                    )
                    last_policy_torques = torques
                    ready = combo.is_cycle_landed(robot, pg, ang, cfg) or combo.is_transition_ready(
                        pg, dof_pos, cycle_default, ang, cfg
                    )
                    rt.stable_steps = rt.stable_steps + 1 if ready else 0
                    if rt.stable_steps >= cfg.transition_steps_req or rt.phase_step >= cfg.hold_quad_steps:
                        # 对齐 cycle 训练：零 action 起步
                        combo.advance_phase_with_robot(
                            rt,
                            combo.Phase.LEGSTAND_CYCLE,
                            device,
                            "handoff to cycle",
                            robot,
                        )
                        phase = SkillPhase.LEGSTAND_CYCLE

                elif phase == SkillPhase.HANDOFF_BACKFLIP:
                    # 原版 HOLD_QUAD_2 → BACKFLIP
                    control_dofs = genesis_dofs
                    torques, pg, ang, dof_pos = combo.run_hold_pose(
                        rt,
                        robot,
                        genesis_dofs,
                        backflip_default,
                        backflip_default.unsqueeze(0),
                        device,
                        flip_kp,
                        flip_kd,
                        flip_tau,
                        zero_cmd,
                    )
                    last_policy_torques = torques
                    ready = combo.is_cycle_landed(robot, pg, ang, cfg) or combo.is_transition_ready(
                        pg, dof_pos, backflip_default, ang, cfg
                    )
                    rt.stable_steps = rt.stable_steps + 1 if ready else 0
                    if rt.stable_steps >= cfg.transition_steps_req or rt.phase_step >= cfg.hold_quad_steps:
                        # 对齐 go2_backflip：从零 action 起步（勿 warm_start）
                        combo.advance_phase_with_robot(
                            rt,
                            combo.Phase.BACKFLIP,
                            device,
                            "handoff to backflip",
                            robot,
                        )
                        phase = SkillPhase.BACKFLIP

                elif phase == SkillPhase.HANDOFF_DOUBLE:
                    control_dofs = genesis_dofs
                    torques, pg, ang, dof_pos = combo.run_hold_pose(
                        rt,
                        robot,
                        genesis_dofs,
                        backflip_default,
                        backflip_default.unsqueeze(0),
                        device,
                        flip_kp,
                        flip_kd,
                        flip_tau,
                        zero_cmd,
                    )
                    last_policy_torques = torques
                    ready = combo.is_cycle_landed(robot, pg, ang, cfg) or combo.is_transition_ready(
                        pg, dof_pos, backflip_default, ang, cfg
                    )
                    rt.stable_steps = rt.stable_steps + 1 if ready else 0
                    if rt.stable_steps >= cfg.transition_steps_req or rt.phase_step >= cfg.hold_quad_steps:
                        combo.advance_phase_with_robot(
                            rt,
                            combo.Phase.BACKFLIP_FINALE,
                            device,
                            "handoff to backflip_double",
                            robot,
                        )
                        phase = SkillPhase.BACKFLIP_DOUBLE

                elif phase == SkillPhase.HANDOFF_JUMP:
                    control_dofs = isaac_dofs
                    torques, pg, ang, dof_pos = combo.run_hold_pose(
                        rt,
                        robot,
                        isaac_dofs,
                        spring_default,
                        spring_default.unsqueeze(0),
                        device,
                        jump_kp,
                        jump_kd,
                        jump_tau,
                        zero_cmd,
                    )
                    last_policy_torques = torques
                    ready = combo.is_cycle_landed(robot, pg, ang, cfg) or combo.is_transition_ready(
                        pg, dof_pos, spring_default, ang, cfg
                    )
                    rt.stable_steps = rt.stable_steps + 1 if ready else 0
                    if rt.stable_steps >= cfg.transition_steps_req or rt.phase_step >= cfg.hold_quad_steps:
                        rt.reset_jump_policy(device)
                        combo.prefill_jump_history(
                            rt,
                            robot,
                            isaac_dofs,
                            spring_default,
                            device,
                            _quat_to_xyz,
                            jump_distance,
                        )
                        rt.phase = combo.Phase.SPRING_JUMP
                        rt.reset_phase_counters()
                        phase = SkillPhase.SPRING_JUMP

                elif phase == SkillPhase.HANDSTAND:
                    # 对齐 go2_eval_handstand_gym_policy：始终 Phase.HANDSTAND，无 HOLD/脚本恢复
                    rt.phase = combo.Phase.HANDSTAND
                    control_dofs = isaac_dofs
                    torques, pg, stand_prog, ang = combo.run_stand_phase(
                        rt,
                        handstand_policy,
                        robot,
                        isaac_dofs,
                        quad_default.unsqueeze(0),
                        hs_desired_delta,
                        zero_cmd,
                        device,
                        stand_kp,
                        stand_kd,
                        stand_tau,
                        global_step,
                        cfg,
                    )
                    last_policy_torques = torques
                    if rt.phase_step >= cfg.handstand_max_steps:
                        phase = SkillPhase.SETTLE
                        settle_steps_left = int(0.6 / combo.CONTROL_DT)
                        rt.reset_phase_counters()

                elif phase == SkillPhase.LEGSTAND_CYCLE:
                    control_dofs = genesis_dofs
                    out, pg, ang, dof_pos, stand_cmd, use_pos_control = combo.run_legstand_cycle_phase(
                        rt,
                        cycle_policy,
                        robot,
                        genesis_dofs,
                        cycle_default.unsqueeze(0),
                        zero_cmd,
                        device,
                        cycle_kp,
                        cycle_kd,
                        cycle_tau,
                        global_step,
                        cfg.cycle_hold_end_s,
                        use_position=True,
                    )
                    if use_pos_control:
                        if not cycle_pd_ready:
                            robot.set_dofs_kp([combo.CYCLE_KP] * combo.NUM_ACTIONS, genesis_dofs)
                            robot.set_dofs_kv([combo.CYCLE_KD] * combo.NUM_ACTIONS, genesis_dofs)
                            cycle_pd_ready = True
                        last_pos_target = out
                    else:
                        torques = out
                        last_policy_torques = torques
                    in_recover = stand_cmd > 0.5
                    if in_recover and combo.is_cycle_landed(robot, pg, ang, cfg):
                        rt.stable_steps += 1
                    else:
                        rt.stable_steps = 0
                    leave = False
                    if in_recover and rt.stable_steps >= cfg.transition_steps_req:
                        leave = True
                    elif in_recover and rt.phase_step >= int(
                        (cfg.cycle_hold_end_s + cfg.cycle_recover_max_s) / combo.CONTROL_DT
                    ):
                        leave = True
                    elif rt.phase_step >= cfg.cycle_max_steps:
                        leave = True
                    if leave:
                        phase = SkillPhase.SETTLE
                        settle_steps_left = int(0.8 / combo.CONTROL_DT)
                        rt.reset_phase_counters()
                        use_pos_control = False
                        last_pos_target = None

                elif phase == SkillPhase.BACKFLIP:
                    control_dofs = genesis_dofs
                    out, pg, ang, dof_pos, use_pos_control = combo.run_backflip_phase(
                        rt,
                        backflip_policy,
                        robot,
                        genesis_dofs,
                        backflip_default.unsqueeze(0),
                        device,
                        flip_kp,
                        flip_kd,
                        flip_tau,
                        global_step,
                        cfg.backflip_max_policy_steps,
                        use_position=True,
                    )
                    if use_pos_control:
                        if not flip_pd_ready:
                            robot.set_dofs_kp([combo.BACKFLIP_KP] * combo.NUM_ACTIONS, genesis_dofs)
                            robot.set_dofs_kv([combo.BACKFLIP_KD] * combo.NUM_ACTIONS, genesis_dofs)
                            flip_pd_ready = True
                        last_pos_target = out
                    else:
                        torques = out
                        last_policy_torques = torques
                    if rt.phase_step >= cfg.backflip_max_steps:
                        begin_flip_settle(max_hold_s=1.5)

                elif phase == SkillPhase.BACKFLIP_DOUBLE:
                    control_dofs = genesis_dofs
                    out, pg, ang, dof_pos, use_pos_control = combo.run_backflip_phase(
                        rt,
                        backflip_double_policy,
                        robot,
                        genesis_dofs,
                        backflip_default.unsqueeze(0),
                        device,
                        flip_kp,
                        flip_kd,
                        flip_tau,
                        global_step,
                        cfg.backflip_double_max_policy_steps,
                        use_position=True,
                    )
                    if use_pos_control:
                        if not flip_pd_ready:
                            robot.set_dofs_kp([combo.BACKFLIP_KP] * combo.NUM_ACTIONS, genesis_dofs)
                            robot.set_dofs_kv([combo.BACKFLIP_KD] * combo.NUM_ACTIONS, genesis_dofs)
                            flip_pd_ready = True
                        last_pos_target = out
                    else:
                        torques = out
                        last_policy_torques = torques
                    if rt.phase_step >= cfg.backflip_double_max_steps:
                        begin_flip_settle(max_hold_s=2.5)

                elif phase == SkillPhase.SPRING_JUMP:
                    control_dofs = isaac_dofs
                    torques, dof_pos, _ = combo.run_spring_jump_phase(
                        rt,
                        spring_policy,
                        robot,
                        isaac_dofs,
                        spring_default.unsqueeze(0),
                        device,
                        jump_kp,
                        jump_kd,
                        jump_tau,
                        global_step,
                        jump_distance,
                        cfg.jump_frame,
                        cfg.jump_startup_steps,
                        _quat_to_xyz,
                    )
                    last_policy_torques = torques
                    # 与原版 combo SPRING_JUMP 结束条件一致
                    inv_q = inv_quat(robot.get_quat())
                    pg = transform_by_quat(
                        torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=device),
                        inv_q,
                    )
                    ang = transform_by_quat(robot.get_ang(), inv_q)
                    after_takeoff = rt.jump_policy_step >= cfg.jump_frame + 30
                    if after_takeoff and (
                        combo.is_cycle_landed(robot, pg, ang, cfg)
                        or combo.is_transition_ready(pg, dof_pos, spring_default, ang, cfg)
                    ):
                        rt.stable_steps += 1
                    else:
                        rt.stable_steps = 0
                    jump_done = (
                        after_takeoff and rt.stable_steps >= cfg.transition_steps_req
                    ) or (rt.phase_step >= cfg.spring_jump_max_steps)
                    if jump_done:
                        phase = SkillPhase.SETTLE
                        settle_steps_left = int(0.8 / combo.CONTROL_DT)
                        rt.reset_phase_counters()

                # 原版：稳定 hold 时施加 XY/yaw anchor
                if combo.should_apply_anchor(rt, rt.phase, cfg, anchor_pg, anchor_prog, anchor_ang):
                    combo.apply_anchor_drift_control(robot, scene, rt, cfg, device)

                if use_pos_control and last_pos_target is not None:
                    # 对齐 go2_env: target 按 motors 序重排后写入 dof slice(6,18)
                    robot.control_dofs_position(last_pos_target[:, actions_dof_idx], slice(6, 18))
                else:
                    robot.control_dofs_force(torques, control_dofs)
                scene.step()
                last_control_dofs = control_dofs
                global_step += 1

                if global_step % 2 == 0:
                    view.update_from_robot(robot)
                if global_step % 5 == 0:
                    server.flush()

    except KeyboardInterrupt:
        print("\n[skills] 退出")
    finally:
        try:
            server.stop()
        except Exception:
            pass


def _make_combo_cfg(combo, args, step_scale: float = 1.0):
    """用 combo 脚本里的 argparse 默认值构造 cfg，避免漏字段。

    ``step_scale``：当技能页改用 go2_backflip 的 dt=0.02 时，把原先按 0.005
    计的整数步数（hs_startup / hs_ramp / jump_startup）等比缩小。
    """
    import ast
    from pathlib import Path as _Path

    src = _Path(combo.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        name = None
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--"):
                name = a.value[2:].replace("-", "_")
                break
        if not name:
            continue
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            try:
                defaults[name] = ast.literal_eval(kw.value)
            except Exception:
                pass

    # BooleanOptionalAction / 常量默认值可能解析不到，用 combo 常量补齐
    defaults.setdefault("anchor_enabled", True)
    defaults.setdefault("cycle_land_pg_z", 0.55)
    defaults.setdefault("cycle_land_h_min", 0.18)
    defaults.setdefault("cycle_land_h_max", 0.55)
    defaults.setdefault("cycle_land_ang_vel", 2.5)
    defaults.setdefault("sim_substeps", getattr(combo, "SIM_SUBSTEPS", 2))
    defaults.setdefault("spring_jump_s", getattr(combo, "JUMP_DURATION_S", 4.0))
    defaults.setdefault("cycle_episode_s", getattr(combo, "CYCLE_EPISODE_S", 30.0))
    defaults.setdefault("cycle_hold_end_s", getattr(combo, "CYCLE_HOLD_END_S", 14.0))
    defaults.setdefault("cycle_recover_max_s", 3.0)
    defaults.setdefault("backflip_episode_s", combo.BACKFLIP_EPISODE_S)
    # 关键：多跑 BACKFLIP_EXTRA_S 会把已站稳的连续空翻弄倒（go2_env 到点会 reset）
    defaults.setdefault("backflip_extra_s", 0.0)
    defaults.setdefault("backflip_double_episode_s", combo.BACKFLIP_DOUBLE_EPISODE_S)
    defaults.setdefault("hold_handstand_s", 2.0)
    defaults.setdefault("hold_quad_s", 0.35)
    defaults.setdefault("recover_quad_s", 8.0)
    defaults.setdefault("handstand_max_s", 12.0)
    defaults.setdefault("stand_stable_s", 1.2)
    defaults.setdefault("transition_stable_s", 0.18)
    defaults.setdefault("hs_startup", 300)
    defaults.setdefault("hs_ramp", 150)
    defaults.setdefault("jump_startup_steps", combo.JUMP_STARTUP_STEPS)
    defaults.setdefault("jump_frame", combo.JUMP_FRAME)

    ns = argparse.Namespace(**defaults)
    ns.mjcf_model = str(GO2_SKILL_MJCF)

    # 按控制周期换算：时间类阈值用 CONTROL_DT；原 0.005 步数类再乘 step_scale
    ns.hs_startup = max(1, int(round(ns.hs_startup * step_scale)))
    ns.hs_ramp = max(1, int(round(ns.hs_ramp * step_scale)))
    ns.jump_startup_steps = max(1, int(round(int(ns.jump_startup_steps) * step_scale)))
    # jump_frame 按政策步计（50Hz），与 dt 换算后 decimation=1 时数值不变
    ns.jump_frame = int(ns.jump_frame)

    ns.stable_steps_req = max(1, int(ns.stand_stable_s / combo.CONTROL_DT))
    ns.transition_steps_req = max(1, int(ns.transition_stable_s / combo.CONTROL_DT))
    ns.hold_handstand_steps = int(ns.hold_handstand_s / combo.CONTROL_DT)
    ns.hold_quad_steps = int(ns.hold_quad_s / combo.CONTROL_DT)
    ns.recover_quad_max = int(ns.recover_quad_s / combo.CONTROL_DT)
    ns.handstand_max_steps = max(
        int(ns.handstand_max_s / combo.CONTROL_DT),
        ns.hs_startup + ns.hs_ramp + ns.stable_steps_req,
    )
    ns.cycle_max_steps = int(ns.cycle_episode_s / combo.CONTROL_DT)
    ns.backflip_max_policy_steps = max(1, int(ns.backflip_episode_s / combo.POLICY_DT))
    ns.backflip_max_steps = int((ns.backflip_episode_s + ns.backflip_extra_s) / combo.CONTROL_DT)
    ns.backflip_double_max_policy_steps = max(
        1, int(ns.backflip_double_episode_s / combo.POLICY_DT)
    )
    ns.backflip_double_max_steps = int(
        (ns.backflip_double_episode_s + ns.backflip_extra_s) / combo.CONTROL_DT
    )
    ns.spring_jump_max_steps = int(ns.spring_jump_s / combo.CONTROL_DT)
    return ns


if __name__ == "__main__":
    main()
