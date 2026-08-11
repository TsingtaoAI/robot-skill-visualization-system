"""运动遥控自研交互层：R.S.V.I.S. 底坞 + WASD 遥控（IJKL 留给相机）。"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Optional

from newtest.common.viser_hotkeys import HotkeySpec, register_hotkeys
from newtest.common.viser_hud import dual_gauge, log_box, meter_html
from newtest.common.viser_theme import apply_product_theme


# 与底坞 GAIT_1..4 及 ViserViewer.GAIT_PRESETS 键一致（勿用中文，否则 dropdown 设值失败）
GAIT_ORDER = ("trot", "bound", "pace", "pronk")
GAIT_LABELS = {
    "trot": "对角小跑",
    "bound": "前后跳跃",
    "pace": "对侧跑步",
    "pronk": "四足齐跳",
}

# 回退默认（若滑条范围读不到时使用）；正常会按指令范围上限的 ~80% 取值
FIXED_VX = 0.40
FIXED_VY = 0.45
FIXED_YAW = 0.60


def _clamp_slider(slider, value: float) -> float:
    lo = float(getattr(slider, "min", -10.0))
    hi = float(getattr(slider, "max", 10.0))
    step = float(getattr(slider, "step", 0.05) or 0.05)
    v = max(lo, min(hi, float(value)))
    if step > 0:
        v = round(v / step) * step
    return float(v)


def _set_cmd(viewer, key: str, value: float) -> None:
    """将指令滑条设为绝对值（经 clamp）。"""
    sliders = getattr(viewer, "_command_sliders", None) or {}
    if key not in sliders:
        return
    s = sliders[key]
    s.value = _clamp_slider(s, float(value))


def _set_slider_ratio(viewer, key: str, ratio: float) -> None:
    """将命名滑条设到 [min,max] 上的比例位置。"""
    sliders = getattr(viewer, "_command_sliders", None) or {}
    if key not in sliders:
        return
    s = sliders[key]
    lo = float(getattr(s, "min", 0.0))
    hi = float(getattr(s, "max", 1.0))
    s.value = _clamp_slider(s, lo + (hi - lo) * float(ratio))


def _zero_velocity(viewer) -> None:
    sliders = getattr(viewer, "_command_sliders", None) or {}
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading"):
        if key in sliders:
            sliders[key].value = 0.0


def _gait_presets(viewer) -> dict:
    presets = getattr(viewer, "GAIT_PRESETS", None)
    if isinstance(presets, dict) and presets:
        return presets
    # 与官方 ViserViewer 默认一致
    return {
        "trot": (0.0, 0.5, 0.5, 0.0),
        "pronk": (0.0, 0.0, 0.0, 0.0),
        "pace": (0.5, 0.0, 0.5, 0.0),
        "bound": (0.0, 0.0, 0.5, 0.5),
    }


def _set_gait(viewer, index: int) -> None:
    if index < 0 or index >= len(GAIT_ORDER):
        return
    name = GAIT_ORDER[index]
    viewer._sel_gait_name = name  # type: ignore[attr-defined]
    viewer._sel_gait = str(index + 1)  # type: ignore[attr-defined]
    dd = getattr(viewer, "_gait_dropdown", None)
    if dd is not None:
        try:
            dd.value = name
        except Exception as exc:
            print(f"[play] gait dropdown 设值失败 ({name}): {exc}", flush=True)
    # 立刻写入 env.theta（不等下一帧）
    env = getattr(viewer, "_play_env", None)
    presets = _gait_presets(viewer)
    theta = presets.get(name)
    if env is not None and theta is not None and hasattr(env, "theta"):
        try:
            env.theta[:, 0] = theta[0]
            env.theta[:, 1] = theta[1]
            env.theta[:, 2] = theta[2]
            env.theta[:, 3] = theta[3]
        except Exception as exc:
            print(f"[play] 写入 theta 失败: {exc}", flush=True)


def _read_cmd(viewer) -> tuple[float, float, float, str]:
    tele = getattr(viewer, "_teleop_cmd", None)
    if tele is not None and len(tele) >= 3:
        vx, vy, yaw = float(tele[0]), float(tele[1]), float(tele[2])
    else:
        sliders = getattr(viewer, "_command_sliders", None) or {}
        vx = float(sliders["lin_vel_x"].value) if "lin_vel_x" in sliders else 0.0
        vy = float(sliders["lin_vel_y"].value) if "lin_vel_y" in sliders else 0.0
        if "ang_vel_yaw" in sliders:
            yaw = float(sliders["ang_vel_yaw"].value)
        elif "heading" in sliders:
            yaw = float(sliders["heading"].value)
        else:
            yaw = 0.0
    gait_key = str(getattr(viewer, "_sel_gait_name", "") or "")
    if not gait_key:
        dd = getattr(viewer, "_gait_dropdown", None)
        if dd is not None:
            gait_key = str(dd.value)
    gait = GAIT_LABELS.get(gait_key, gait_key or "—")
    return vx, vy, yaw, gait


def _norm_vel(v: float, scale: float = 1.5) -> float:
    return max(0.0, min(1.0, abs(float(v)) / scale))


def attach_play_console(viewer, env=None) -> Any:
    """挂载遥控甲板：WASD 移动，Q/E 偏航；IJKL/UO 由 HUD 脚本映射为相机。"""
    server = viewer.server
    apply_product_theme(
        server,
        module_label="运动遥控",
        module_id="play",
        dark_mode=True,
        scene_info="teleop-deck",
        status_text="STATUS: IDLE",
    )

    viewer._invert_forward = False  # type: ignore[attr-defined]
    viewer._cruise_hold = False  # type: ignore[attr-defined]
    viewer._cmd_history = deque(maxlen=12)  # type: ignore[attr-defined]
    viewer._speed_cap = 1.0  # type: ignore[attr-defined]
    viewer._play_note = ""  # type: ignore[attr-defined]
    viewer._play_env = env  # type: ignore[attr-defined]
    # 权威遥控指令（每帧直接写入 env.commands，不依赖滑条点击是否成功）
    viewer._teleop_cmd = [0.0, 0.0, 0.0]  # type: ignore[attr-defined]  # vx, vy, yaw_rate
    # 单选状态（供底坞高亮同步）；与 GAIT_ORDER / HUD 一致
    viewer._sel_gait = "1"  # type: ignore[attr-defined]
    viewer._sel_gait_name = GAIT_ORDER[0]  # type: ignore[attr-defined]
    viewer._sel_cap = "1.0"  # type: ignore[attr-defined]
    viewer._sel_period = "mid"  # type: ignore[attr-defined]
    viewer._sel_height = "mid"  # type: ignore[attr-defined]
    viewer._sel_cam = "lock"  # type: ignore[attr-defined]
    if not hasattr(viewer, "_camera_tracking_enabled"):
        viewer._camera_tracking_enabled = True  # type: ignore[attr-defined]

    # 从滑条范围推导「默认行走速度」= 范围上限的 80%（中等偏快、有明显反应）
    def _default_from_slider(key: str, fallback: float) -> float:
        sliders = getattr(viewer, "_command_sliders", None) or {}
        if key not in sliders:
            return float(fallback)
        s = sliders[key]
        lo = abs(float(getattr(s, "min", 0.0)))
        hi = abs(float(getattr(s, "max", fallback)))
        peak = max(lo, hi, 1e-3)
        return float(peak * 0.8)

    viewer._default_vx = _default_from_slider("lin_vel_x", FIXED_VX)  # type: ignore[attr-defined]
    viewer._default_vy = _default_from_slider("lin_vel_y", FIXED_VY)  # type: ignore[attr-defined]
    yaw_key0 = "ang_vel_yaw" if "ang_vel_yaw" in (getattr(viewer, "_command_sliders", None) or {}) else "heading"
    # heading 模式也按角速度滑条范围估；没有则用 FIXED_YAW
    viewer._default_yaw = _default_from_slider(  # type: ignore[attr-defined]
        "ang_vel_yaw" if yaw_key0 == "ang_vel_yaw" else "heading",
        FIXED_YAW,
    )
    # heading 绝对值范围是 ±π，不适合当角速度；强制用合理 yaw rate
    if yaw_key0 == "heading" or viewer._default_yaw > 1.5:  # type: ignore[attr-defined]
        viewer._default_yaw = FIXED_YAW  # type: ignore[attr-defined]

    print(
        f"[play] 默认遥控速度 vx={viewer._default_vx:.2f} vy={viewer._default_vy:.2f} "
        f"yaw={viewer._default_yaw:.2f}（按住 WASD/QE 直接写入，松手清零）",
        flush=True,
    )

    # 关闭 heading 重算：否则会持续把 commands[:,2] 改成朝向误差 → 不停转圈
    try:
        viewer._heading_command = False  # type: ignore[attr-defined]
        if env is not None and hasattr(env, "cfg"):
            env.cfg.commands.heading_command = False
            if hasattr(env, "commands"):
                env.commands[:, :3] = 0.0
                if env.commands.shape[-1] > 3:
                    env.commands[:, 3] = 0.0
        _set_gait(viewer, 0)
        print("[play] 已关闭 heading_command，偏航仅由 Q/E / TELEOP 控制", flush=True)
    except Exception as exc:
        print(f"[play] 关闭 heading_command 失败: {exc}", flush=True)

    def _sel() -> dict:
        return {
            "gait": str(getattr(viewer, "_sel_gait", "1")),
            "cap": str(getattr(viewer, "_sel_cap", "1.0")),
            "period": str(getattr(viewer, "_sel_period", "mid")),
            "height": str(getattr(viewer, "_sel_height", "mid")),
            "cam": str(getattr(viewer, "_sel_cam", "lock")),
            "defvx": f"{float(getattr(viewer, '_default_vx', FIXED_VX)):.3f}",
            "defvy": f"{float(getattr(viewer, '_default_vy', FIXED_VY)):.3f}",
            "defyaw": f"{float(getattr(viewer, '_default_yaw', FIXED_YAW)):.3f}",
        }

    def _refresh() -> None:
        vx, vy, yaw, gait = _read_cmd(viewer)
        note = str(getattr(viewer, "_play_note", "") or "")
        invert = "ON" if getattr(viewer, "_invert_forward", False) else "OFF"
        cruise = "ON" if getattr(viewer, "_cruise_hold", False) else "OFF"
        cap = float(getattr(viewer, "_speed_cap", 1.0))
        cam = "LOCK" if getattr(viewer, "_camera_tracking_enabled", True) else "FREE"
        hist: Deque[str] = getattr(viewer, "_cmd_history", deque())
        try:
            viewer._gauge_html.content = dual_gauge(  # type: ignore[attr-defined]
                left_value=_norm_vel(vx),
                right_value=_norm_vel(vy),
                left_label="VX",
                right_label="VY",
                accent="#64e650",
            )
        except Exception:
            pass
        try:
            spd = _norm_vel((vx * vx + vy * vy) ** 0.5, 2.0)
            viewer._meter_html.content = meter_html(level=max(1, int(spd * 15) + 1))  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            viewer._status_md.content = (  # type: ignore[attr-defined]
                f"**vx** `{vx:.2f}`  **vy** `{vy:.2f}`  **yaw** `{yaw:.2f}`  ·  {gait}  ·  cam={cam}"
            )
        except Exception:
            pass
        try:
            lines = [
                f"> {note}" if note else "> teleop online",
                f"> vx={vx:.2f} vy={vy:.2f} yaw={yaw:.2f}",
                f"> gait={gait}  cap×{cap:.1f}  cam={cam}",
                f"> cruise={cruise} invert={invert}",
                "> WASD/QE hold = fixed speed · release = stop",
                "> cam free: IJKL / UO / arrows (this page)",
            ]
            for h in list(hist)[-3:]:
                lines.append(f"> {h}")
            viewer._log_html.content = log_box(lines, sel=_sel())  # type: ignore[attr-defined]
        except Exception:
            pass

    def _note(text: str) -> None:
        viewer._play_note = text  # type: ignore[attr-defined]
        _refresh()

    def _fixed_speed(axis: str) -> float:
        scale = float(getattr(viewer, "_speed_cap", 1.0))
        if axis == "x":
            return float(getattr(viewer, "_default_vx", FIXED_VX)) * scale
        if axis == "y":
            return float(getattr(viewer, "_default_vy", FIXED_VY)) * scale
        return float(getattr(viewer, "_default_yaw", FIXED_YAW)) * scale

    def _sync_teleop_text() -> None:
        box = getattr(viewer, "_teleop_text", None)
        if box is None:
            return
        vx, vy, yw = viewer._teleop_cmd  # type: ignore[attr-defined]
        try:
            box.value = f"{vx:.3f},{vy:.3f},{yw:.3f}"
        except Exception:
            pass

    def _set_teleop(vx: Optional[float] = None, vy: Optional[float] = None, yaw: Optional[float] = None) -> None:
        """更新权威遥控指令；None 表示该轴不变。"""
        cmd = list(getattr(viewer, "_teleop_cmd", [0.0, 0.0, 0.0]))
        if vx is not None:
            cmd[0] = float(vx)
        if vy is not None:
            cmd[1] = float(vy)
        if yaw is not None:
            cmd[2] = float(yaw)
        viewer._teleop_cmd = cmd  # type: ignore[attr-defined]
        # 同步滑条显示（非权威）
        _set_cmd(viewer, "lin_vel_x", cmd[0])
        _set_cmd(viewer, "lin_vel_y", cmd[1])
        if "ang_vel_yaw" in (getattr(viewer, "_command_sliders", None) or {}):
            _set_cmd(viewer, "ang_vel_yaw", cmd[2])
        _sync_teleop_text()
        _refresh()

    def _apply_vx(sign: int) -> None:
        v = _fixed_speed("x") * float(sign)
        if sign != 0 and getattr(viewer, "_invert_forward", False):
            v = -v
        _set_teleop(vx=v)

    def _apply_vy(sign: int) -> None:
        _set_teleop(vy=_fixed_speed("y") * float(sign))

    def _apply_yaw(sign: int) -> None:
        _set_teleop(yaw=_fixed_speed("yaw") * float(sign))

    def _brake() -> None:
        viewer._teleop_cmd = [0.0, 0.0, 0.0]  # type: ignore[attr-defined]
        _zero_velocity(viewer)
        _sync_teleop_text()
        viewer._cmd_history.append("急停")  # type: ignore[attr-defined]
        _note("已急停")

    def _yaw_key() -> str:
        return "ang_vel_yaw"

    def _cam_lock() -> None:
        viewer._camera_tracking_enabled = True  # type: ignore[attr-defined]
        viewer._sel_cam = "lock"  # type: ignore[attr-defined]
        cb = getattr(viewer, "_play_cam_checkbox", None)
        if cb is not None:
            try:
                cb.value = True
            except Exception:
                pass
        _note("相机已锁定（跟随）")

    def _cam_free() -> None:
        viewer._camera_tracking_enabled = False  # type: ignore[attr-defined]
        viewer._sel_cam = "free"  # type: ignore[attr-defined]
        cb = getattr(viewer, "_play_cam_checkbox", None)
        if cb is not None:
            try:
                cb.value = False
            except Exception:
                pass
        _note("自由相机（IJKL/鼠标）")

    # ── 隐藏 GUI（供底坞/键盘代理点击）────────────────
    with server.gui.add_folder("COMMAND READOUT", expand_by_default=True):
        viewer._gauge_html = server.gui.add_html(  # type: ignore[attr-defined]
            dual_gauge(left_value=0.0, right_value=0.0)
        )
        viewer._meter_html = server.gui.add_html(meter_html(level=1))  # type: ignore[attr-defined]
        viewer._status_md = server.gui.add_markdown("**vx** `0.00`")  # type: ignore[attr-defined]

    with server.gui.add_folder("MOVE PROXY", expand_by_default=True):
        # TELEOP_CMD: JS 写入 "vx,vy,yaw"；每帧 apply 以此为准
        teleop_text = server.gui.add_text("TELEOP_CMD", initial_value="0.000,0.000,0.000")
        viewer._teleop_text = teleop_text  # type: ignore[attr-defined]

        @teleop_text.on_update
        def _(_e) -> None:
            raw = str(teleop_text.value or "0,0,0").strip()
            try:
                parts = [float(x) for x in raw.replace(" ", "").split(",")]
                if len(parts) >= 3:
                    viewer._teleop_cmd = [parts[0], parts[1], parts[2]]  # type: ignore[attr-defined]
                    _set_cmd(viewer, "lin_vel_x", parts[0])
                    _set_cmd(viewer, "lin_vel_y", parts[1])
                    if "ang_vel_yaw" in (getattr(viewer, "_command_sliders", None) or {}):
                        _set_cmd(viewer, "ang_vel_yaw", parts[2])
                    _refresh()
            except Exception:
                pass

        move_btns = [
            ("MOVE_FWD", lambda: _apply_vx(+1)),
            ("MOVE_BACK", lambda: _apply_vx(-1)),
            ("MOVE_LEFT", lambda: _apply_vy(+1)),
            ("MOVE_RIGHT", lambda: _apply_vy(-1)),
            ("YAW_L", lambda: _apply_yaw(+1)),
            ("YAW_R", lambda: _apply_yaw(-1)),
            ("ZERO_VX", lambda: _apply_vx(0)),
            ("ZERO_VY", lambda: _apply_vy(0)),
            ("ZERO_YAW", lambda: _apply_yaw(0)),
            ("ZERO_ALL", lambda: _brake()),
        ]
        for label, cb in move_btns:
            btn = server.gui.add_button(label)

            def _bind(callback=cb, button=btn):
                @button.on_click
                def _(_e, fn=callback) -> None:
                    fn()

            _bind()

    with server.gui.add_folder("TELEOP CAP", expand_by_default=True):
        cap = server.gui.add_slider("指令幅度上限", min=0.3, max=1.0, step=0.1, initial_value=1.0)
        cb_cruise = server.gui.add_checkbox("巡航保持", initial_value=False)
        btn_cap30 = server.gui.add_button("×0.3", color="#3aa0a8")
        btn_cap70 = server.gui.add_button("×0.7", color="cyan")
        btn_cap100 = server.gui.add_button("×1.0", color="green")

        @cb_cruise.on_update
        def _(_e) -> None:
            viewer._cruise_hold = bool(cb_cruise.value)  # type: ignore[attr-defined]
            _note("巡航保持已更新")

        @cap.on_update
        def _(_e) -> None:
            viewer._speed_cap = float(cap.value)  # type: ignore[attr-defined]
            v = float(cap.value)
            if abs(v - 0.3) < 0.05:
                viewer._sel_cap = "0.3"  # type: ignore[attr-defined]
            elif abs(v - 0.7) < 0.05:
                viewer._sel_cap = "0.7"  # type: ignore[attr-defined]
            else:
                viewer._sel_cap = "1.0"  # type: ignore[attr-defined]
            _note(f"幅度上限 ×{cap.value:.1f}")

        @btn_cap30.on_click
        def _(_e) -> None:
            cap.value = 0.3

        @btn_cap70.on_click
        def _(_e) -> None:
            cap.value = 0.7

        @btn_cap100.on_click
        def _(_e) -> None:
            cap.value = 1.0

    with server.gui.add_folder("GAIT QUICK", expand_by_default=True):
        for i, name in enumerate(GAIT_ORDER):
            btn = server.gui.add_button(f"GAIT_{i+1}", color="#64e650" if i == 0 else "#3aa0a8")

            def _make(idx: int, button=btn):
                @button.on_click
                def _(_e, j=idx) -> None:
                    _set_gait(viewer, j)
                    name = GAIT_ORDER[j]
                    label = GAIT_LABELS.get(name, name)
                    viewer._cmd_history.append(f"gait:{label}")  # type: ignore[attr-defined]
                    _note(f"步态 → {label}")

            _make(i)

        btn_ps = server.gui.add_button("PERIOD_SLOW")
        btn_pm = server.gui.add_button("PERIOD_MID")
        btn_pf = server.gui.add_button("PERIOD_FAST")

        @btn_ps.on_click
        def _(_e) -> None:
            _set_slider_ratio(viewer, "gait_period", 0.85)
            viewer._sel_period = "slow"  # type: ignore[attr-defined]
            _note("步态周期：慢")

        @btn_pm.on_click
        def _(_e) -> None:
            _set_slider_ratio(viewer, "gait_period", 0.50)
            viewer._sel_period = "mid"  # type: ignore[attr-defined]
            _note("步态周期：中")

        @btn_pf.on_click
        def _(_e) -> None:
            _set_slider_ratio(viewer, "gait_period", 0.20)
            viewer._sel_period = "fast"  # type: ignore[attr-defined]
            _note("步态周期：快")

        btn_hl = server.gui.add_button("HEIGHT_LOW")
        btn_hm = server.gui.add_button("HEIGHT_MID")
        btn_hh = server.gui.add_button("HEIGHT_HIGH")

        @btn_hl.on_click
        def _(_e) -> None:
            _set_slider_ratio(viewer, "base_height_target", 0.20)
            viewer._sel_height = "low"  # type: ignore[attr-defined]
            _note("机身高度：低")

        @btn_hm.on_click
        def _(_e) -> None:
            _set_slider_ratio(viewer, "base_height_target", 0.50)
            viewer._sel_height = "mid"  # type: ignore[attr-defined]
            _note("机身高度：中")

        @btn_hh.on_click
        def _(_e) -> None:
            _set_slider_ratio(viewer, "base_height_target", 0.85)
            viewer._sel_height = "high"  # type: ignore[attr-defined]
            _note("机身高度：高")

    with server.gui.add_folder("CAMERA PROXY", expand_by_default=True):
        btn_cam_lock = server.gui.add_button("CAM LOCK", color="green")
        btn_cam_free = server.gui.add_button("CAM FREE", color="#3aa0a8")

        @btn_cam_lock.on_click
        def _(_e) -> None:
            _cam_lock()

        @btn_cam_free.on_click
        def _(_e) -> None:
            _cam_free()

    with server.gui.add_folder("ACTION COMMANDS", expand_by_default=True):
        btn_stop = server.gui.add_button("STOP / BRAKE", color="red")
        btn_reset = server.gui.add_button("RESET STAND", color="orange")
        btn_flip = server.gui.add_button("FLIP FORWARD", color="cyan")
        btn_shot = server.gui.add_button("SNAPSHOT", color="#3d7dff")

        @btn_stop.on_click
        def _(_e) -> None:
            _brake()

        @btn_reset.on_click
        def _(_e) -> None:
            _zero_velocity(viewer)
            if env is not None and hasattr(env, "reset"):
                try:
                    env.reset()
                    viewer._cmd_history.append("复位")  # type: ignore[attr-defined]
                except Exception as exc:
                    print(f"[play] reset 失败: {exc}", flush=True)
            _note("已请求复位站立")

        @btn_flip.on_click
        def _(_e) -> None:
            viewer._invert_forward = not bool(getattr(viewer, "_invert_forward", False))  # type: ignore[attr-defined]
            _note("已切换前进方向映射")

        @btn_shot.on_click
        def _(_e) -> None:
            clients = list(server.get_clients().values())
            if not clients:
                _note("无连接，无法截图")
                return
            try:
                import io
                import numpy as np
                from PIL import Image

                arr = clients[0].get_render(height=480, width=640)
                img = Image.fromarray(np.asarray(arr))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                clients[0].send_file_download("play_snapshot.png", buf.getvalue())
                _note("已导出 play_snapshot.png")
            except Exception as exc:
                _note(f"截图失败：{exc}")

    with server.gui.add_folder("SYSTEM STATUS LOG", expand_by_default=True):
        viewer._log_html = server.gui.add_html(  # type: ignore[attr-defined]
            log_box(
                [
                    "> R.S.V.I.S. teleop deck online",
                    "> WASD/QE hold fixed speed, release stops",
                    "> cam lock/free · free uses IJKL/UO",
                ],
                sel=_sel(),
            )
        )

    # 尝试挂上官方「跟随机器人」复选框，便于双向同步
    try:
        # 遍历已有 GUI 较难；在 checkbox 更新时我们只写 _camera_tracking_enabled
        # 若 vender 侧 checkbox 被用户点，会直接改属性，我们的按钮也改同一属性
        pass
    except Exception:
        pass

    def _cap_set(v: float) -> None:
        cap.value = float(v)
        _note(f"幅度 ×{v:.1f}")

    def _period(r: float, label: str, tag: str):
        def _() -> None:
            _set_slider_ratio(viewer, "gait_period", r)
            viewer._sel_period = tag  # type: ignore[attr-defined]
            _note(label)

        return _

    def _height(r: float, label: str, tag: str):
        def _() -> None:
            _set_slider_ratio(viewer, "base_height_target", r)
            viewer._sel_height = tag  # type: ignore[attr-defined]
            _note(label)

        return _

    def _do_reset() -> None:
        viewer._teleop_cmd = [0.0, 0.0, 0.0]  # type: ignore[attr-defined]
        _zero_velocity(viewer)
        _sync_teleop_text()
        if env is not None and hasattr(env, "reset"):
            try:
                env.reset()
            except Exception:
                pass
        _note("已请求复位站立")

    def _do_flip() -> None:
        viewer._invert_forward = not bool(getattr(viewer, "_invert_forward", False))  # type: ignore[attr-defined]
        _note("已切换前进方向映射")

    def _gait_hot(i: int):
        def _() -> None:
            _set_gait(viewer, i)
            name = GAIT_ORDER[i]
            _note(f"步态 → {GAIT_LABELS.get(name, name)}")

        return _

    # 服务端热键作备份；主路径是 HUD 客户端 JS 代理点击
    specs = [
        HotkeySpec("急停 B", _brake, "B", description="急停"),
        HotkeySpec("急停 Space", _brake, "space", description="急停"),
        HotkeySpec("步态1", _gait_hot(0), "1"),
        HotkeySpec("步态2", _gait_hot(1), "2"),
        HotkeySpec("步态3", _gait_hot(2), "3"),
        HotkeySpec("步态4", _gait_hot(3), "4"),
        HotkeySpec("幅度0.3", lambda: _cap_set(0.3), "5"),
        HotkeySpec("幅度0.7", lambda: _cap_set(0.7), "6"),
        HotkeySpec("幅度1.0", lambda: _cap_set(1.0), "7"),
        HotkeySpec("周期慢", _period(0.85, "周期慢", "slow"), "Z"),
        HotkeySpec("周期中", _period(0.50, "周期中", "mid"), "X"),
        HotkeySpec("周期快", _period(0.20, "周期快", "fast"), "C"),
        HotkeySpec("高度低", _height(0.20, "高度低", "low"), "V"),
        HotkeySpec("高度中", _height(0.50, "高度中", "mid"), "N"),
        HotkeySpec("高度高", _height(0.85, "高度高", "high"), "M"),
        HotkeySpec("复位", _do_reset, "R", description="复位站立"),
        HotkeySpec("前进反向", _do_flip, "F", description="前进反向"),
        HotkeySpec("锁定相机", _cam_lock, None, description="跟随"),
        HotkeySpec("自由相机", _cam_free, None, description="自由"),
    ]
    register_hotkeys(server, specs)

    _orig_apply = viewer.apply_commands_to_env

    def apply_and_refresh(env_arg) -> None:
        """每帧把按住键对应的默认速度强制写入 env（覆盖随机/旧指令）。"""
        try:
            # 始终禁止 heading 重算（防止配置被别处改回）
            try:
                env_arg.cfg.commands.heading_command = False
            except Exception:
                pass
            viewer._heading_command = False  # type: ignore[attr-defined]

            sliders = getattr(viewer, "_command_sliders", None) or {}
            if getattr(viewer, "_has_wtw_controls", False):
                for key, attr in (
                    ("gait_period", "gait_period"),
                    ("base_height_target", "base_height_target"),
                    ("foot_clearance_target", "foot_clearance_target"),
                    ("pitch_target", "pitch_target"),
                ):
                    if key in sliders and hasattr(env_arg, attr):
                        getattr(env_arg, attr)[:] = sliders[key].value

                presets = _gait_presets(viewer)
                gait_name = str(getattr(viewer, "_sel_gait_name", "") or "")
                if not gait_name:
                    dd = getattr(viewer, "_gait_dropdown", None)
                    if dd is not None:
                        gait_name = str(dd.value)
                theta = presets.get(gait_name)
                if theta is not None and hasattr(env_arg, "theta"):
                    env_arg.theta[:, 0] = theta[0]
                    env_arg.theta[:, 1] = theta[1]
                    env_arg.theta[:, 2] = theta[2]
                    env_arg.theta[:, 3] = theta[3]

            vx, vy, yw = viewer._teleop_cmd  # type: ignore[attr-defined]
            # 松手时应严格为零，避免浮点残留驱动转圈
            if abs(float(yw)) < 1e-4:
                yw = 0.0
            if abs(float(vx)) < 1e-4:
                vx = 0.0
            if abs(float(vy)) < 1e-4:
                vy = 0.0
            env_arg.commands[:, 0] = float(vx)
            env_arg.commands[:, 1] = float(vy)
            env_arg.commands[:, 2] = float(yw)
            # heading 通道置零（heading_command 已关；双保险避免残留目标角）
            if getattr(env_arg.commands, "shape", [0, 0])[-1] > 3:
                env_arg.commands[:, 3] = 0.0
        except Exception as exc:
            try:
                _orig_apply(env_arg)
            except Exception:
                print(f"[play] apply teleop failed: {exc}", flush=True)
        try:
            _refresh()
        except Exception:
            pass

    viewer.apply_commands_to_env = apply_and_refresh  # type: ignore[method-assign]
    viewer._play_refresh = _refresh  # type: ignore[attr-defined]
    _refresh()
    print(
        "[play] 遥控甲板：按住 WASD/QE=默认固定速度并覆盖当前指令，松手停止；"
        "CAM LOCK/FREE；自由后 IJKL·UO 相机（仅本页）",
        flush=True,
    )
    return viewer
