"""R.S.V.I.S. HUD：隐藏 Viser 自带浮窗，自建底部横向控制坞（点击代理到隐藏 GUI）。"""

from __future__ import annotations

import base64
import html as html_lib
from typing import Optional, Tuple

SOFTWARE_NAME = "机器人技能可视化演示交互系统"
SOFTWARE_SHORT = "R.S.V.I.S."
SOFTWARE_VERSION = "V1.0"
PORTAL_DEFAULT_URL = "http://127.0.0.1:8090"

MODULE_LOOK = {
    "skills": {
        "label": "SKILL-DEMO",
        "label_cn": "技能演示",
        "accent": (0, 200, 255),
        "hex": "#00c8ff",
        "accent2": (255, 140, 40),
        "hex2": "#ff8c28",
        "panel": " ",
    },
    "nav": {
        "label": "NAV-DEMO",
        "label_cn": "导航演示",
        "accent": (0, 210, 230),
        "hex": "#00d2e6",
        "accent2": (80, 255, 140),
        "hex2": "#50ff8c",
        "panel": " ",
    },
    "play": {
        "label": "TELEOP-DEMO",
        "label_cn": "运动遥控",
        "accent": (100, 230, 80),
        "hex": "#64e650",
        "accent2": (0, 200, 255),
        "hex2": "#00c8ff",
        "panel": " ",
    },
}


def _esc(text: str) -> str:
    return html_lib.escape(str(text), quote=True)


def _css(module_id: str, accent_hex: str, accent2_hex: str) -> str:
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Share+Tech+Mono&display=swap');

:root {{
  --rsv-accent: {accent_hex};
  --rsv-accent2: {accent2_hex};
  --rsv-bg: #070a0f;
  --rsv-glass: rgba(12, 18, 28, 0.88);
  --rsv-line: rgba(160, 210, 240, 0.32);
  --rsv-text: #d8e6f2;
  --rsv-muted: #7f93a8;
}}

html, body, #root {{
  background: radial-gradient(ellipse at 50% 28%, #101820 0%, var(--rsv-bg) 55%, #04060a 100%) !important;
}}

/* Hide native Viser floating GUI (keep in DOM for click proxy) */
.rsvis-hidden-gui,
.rsvis-hidden-gui-wrap {{
  position: fixed !important;
  left: -14000px !important;
  top: 0 !important;
  width: 520px !important;
  height: 980px !important;
  max-width: 520px !important;
  max-height: 980px !important;
  opacity: 0 !important;
  pointer-events: none !important;
  z-index: -5 !important;
  overflow: hidden !important;
  transform: none !important;
}}

button[aria-label*="Share"], a[href*="share"] {{ opacity: 0.2; }}

/* ===== Top chrome (does not block 3D: pointer-events none except chips) ===== */
#rsvis-chrome {{
  pointer-events: none;
  position: fixed;
  inset: 0;
  z-index: 60;
  font-family: 'Rajdhani', 'Segoe UI', sans-serif;
  color: var(--rsv-text);
}}
#rsvis-chrome * {{ box-sizing: border-box; }}
#rsvis-top {{
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 64px;
  display: grid;
  grid-template-columns: 1fr minmax(280px, 520px) 1fr;
  gap: 10px;
  padding: 8px 14px 0;
  background: linear-gradient(180deg, rgba(0,0,0,0.72) 0%, transparent 100%);
  pointer-events: none;
}}
#rsvis-top a.rsvis-chip {{ pointer-events: auto; cursor: pointer; }}

.rsvis-side {{ display: flex; align-items: center; gap: 8px; }}
.rsvis-side.right {{ justify-content: flex-end; }}

.rsvis-chip {{
  display: inline-flex; align-items: center; height: 32px; padding: 0 12px;
  border: 1px solid var(--rsv-line); background: var(--rsv-glass); color: var(--rsv-text);
  font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  text-decoration: none; backdrop-filter: blur(10px);
  clip-path: polygon(6px 0, 100% 0, 100% calc(100% - 6px), calc(100% - 6px) 100%, 0 100%, 0 6px);
}}
.rsvis-chip:hover {{ border-color: var(--rsv-accent); color: var(--rsv-accent); }}

.rsvis-title-trap {{
  position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center;
  padding: 6px 24px 8px;
  background: linear-gradient(180deg, rgba(20,30,44,0.94), rgba(8,12,18,0.9));
  border: 1px solid var(--rsv-line); border-bottom: 2px solid var(--rsv-accent);
  clip-path: polygon(8% 0, 92% 0, 100% 100%, 0 100%);
  box-shadow: 0 0 24px rgba(0, 200, 255, 0.14);
}}
.rsvis-title-trap::before {{
  content: ''; position: absolute; left: 16%; right: 16%; top: 100%; height: 110px;
  background: radial-gradient(ellipse at 50% 0%, rgba(0,200,255,0.22) 0%, transparent 70%);
  pointer-events: none; z-index: -1;
}}
.rsvis-title-trap .code {{
  font-family: 'Share Tech Mono', monospace; font-size: 14px; font-weight: 700;
  letter-spacing: 0.06em; color: #f2f7fc;
}}
.rsvis-title-trap .sub {{ margin-top: 2px; font-size: 10px; letter-spacing: 0.14em; color: var(--rsv-muted); }}
.rsvis-title-trap .status {{
  margin-top: 2px; font-size: 11px; font-weight: 700; letter-spacing: 0.12em; color: var(--rsv-accent2);
}}
.rsvis-scene-box {{
  min-width: 140px; padding: 4px 10px; border: 1px solid var(--rsv-line); background: rgba(0,0,0,0.45);
  font-size: 10px; letter-spacing: 0.08em;
}}
.rsvis-scene-box b {{ display: block; color: var(--rsv-accent); font-size: 11px; margin-top: 2px; }}
.rsvis-sig {{ display: flex; gap: 6px; align-items: center; font-family: 'Share Tech Mono', monospace; font-size: 10px; color: var(--rsv-muted); }}
.rsvis-sig span {{ width: 14px; height: 10px; border: 1px solid var(--rsv-line); display: inline-block; position: relative; }}
.rsvis-sig span::after {{ content: ''; position: absolute; inset: 2px; background: linear-gradient(90deg, var(--rsv-accent), transparent); }}

/* ===== Custom bottom dock: hover to expand ===== */
#rsvis-dock-shell {{
  position: fixed;
  left: 0; right: 0; bottom: 0;
  z-index: 55;
  pointer-events: none;
  font-family: 'Rajdhani', 'Segoe UI', sans-serif;
  color: var(--rsv-text);
}}
#rsvis-dock-hit {{
  pointer-events: auto;
  position: relative;
  margin: 0 12px 8px;
  padding-top: 18px; /* invisible hover strip above the tab */
}}
#rsvis-dock-tab {{
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  border: 1px solid rgba(0, 210, 255, 0.35);
  border-bottom: none;
  border-radius: 10px 10px 0 0;
  background: linear-gradient(180deg, rgba(20,30,44,0.95), rgba(8,12,18,0.92));
  box-shadow: 0 -6px 20px rgba(0,0,0,0.35);
  cursor: default;
  user-select: none;
  letter-spacing: 0.14em;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  color: var(--rsv-accent);
}}
#rsvis-dock-tab .chev {{
  display: inline-block;
  transition: transform 0.28s ease;
  color: var(--rsv-accent2);
}}
#rsvis-dock-body {{
  max-height: 0;
  opacity: 0;
  overflow: hidden;
  transform: translateY(10px);
  transition: max-height 0.35s ease, opacity 0.25s ease, transform 0.35s ease;
  border: 1px solid rgba(0, 210, 255, 0.28);
  border-top: none;
  border-radius: 0 0 10px 10px;
  background: rgba(6, 10, 16, 0.55);
  backdrop-filter: blur(10px);
  padding: 0 8px;
}}
#rsvis-dock-hit:hover #rsvis-dock-body,
#rsvis-dock-hit.rsvis-dock-open #rsvis-dock-body {{
  max-height: 42vh;
  opacity: 1;
  transform: translateY(0);
  overflow: auto;
  padding: 8px;
}}
#rsvis-dock-hit:hover #rsvis-dock-tab .chev,
#rsvis-dock-hit.rsvis-dock-open #rsvis-dock-tab .chev {{
  transform: rotate(180deg);
}}

#rsvis-dock {{
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
#rsvis-dock * {{ box-sizing: border-box; }}

.rsvis-dock-row {{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
}}
.rsvis-dock-row.cols-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.rsvis-dock-row.cols-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.rsvis-dock-row.auto {{
  display: flex; flex-wrap: wrap; gap: 8px;
}}
.rsvis-dock-row.auto > .rsvis-btn {{ flex: 1 1 120px; }}

.rsvis-card {{
  background:
    linear-gradient(165deg, rgba(28, 40, 58, 0.78) 0%, rgba(8, 12, 20, 0.9) 100%),
    repeating-linear-gradient(135deg, rgba(255,255,255,0.02) 0 1px, transparent 1px 3px);
  border: 1px solid rgba(0, 210, 255, 0.34);
  border-radius: 10px;
  padding: 10px 12px;
  backdrop-filter: blur(12px) saturate(1.1);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 8px 24px rgba(0,0,0,0.4);
  min-width: 0;
}}
.rsvis-card h3 {{
  margin: 0 0 8px;
  font-size: 11px; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--rsv-accent);
  border-bottom: 1px solid rgba(0,200,255,0.18);
  padding-bottom: 6px;
}}
.rsvis-card .hint {{
  font-family: 'Share Tech Mono', monospace; font-size: 9px; color: var(--rsv-muted);
  margin-bottom: 6px; letter-spacing: 0.06em;
}}
.rsvis-card .btns {{
  display: flex; flex-wrap: wrap; gap: 6px;
}}
.rsvis-card .btns.grid5 {{
  display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px;
}}
.rsvis-card .btns.grid2 {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
}}

.rsvis-btn {{
  appearance: none; cursor: pointer;
  min-height: 40px; padding: 8px 10px;
  border-radius: 6px;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(0,0,0,0.35);
  color: var(--rsv-text);
  font-family: 'Rajdhani', sans-serif;
  font-size: 13px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  box-shadow: 0 0 10px rgba(0,0,0,0.25);
}}
.rsvis-btn:hover {{ border-color: var(--rsv-accent); color: #fff; }}
.rsvis-btn:active {{ transform: translateY(1px); }}
.rsvis-btn.accent {{ border-color: #ff8c28; box-shadow: 0 0 12px rgba(255,140,40,0.25); }}
.rsvis-btn.green {{ border-color: #3dff8a; }}
.rsvis-btn.red {{ border-color: #ff4d4d; }}
.rsvis-btn.cyan {{ border-color: #00d2e6; }}
.rsvis-btn.purple {{ border-color: #7c5cff; }}
.rsvis-btn.block {{ width: 100%; }}
.rsvis-btn.is-on {{
  outline: 2px solid var(--rsv-accent2);
  outline-offset: 1px;
  background: rgba(0, 210, 230, 0.18);
  box-shadow: inset 0 0 0 1px var(--rsv-accent), 0 0 14px rgba(0, 210, 230, 0.35);
  position: relative;
}}
.rsvis-btn.is-on::after {{
  content: 'ON';
  position: absolute;
  top: 4px; right: 6px;
  font-size: 9px;
  letter-spacing: 0.08em;
  color: var(--rsv-accent2);
  font-family: 'Share Tech Mono', monospace;
}}

.rsvis-log {{
  font-family: 'Share Tech Mono', monospace; font-size: 10px; line-height: 1.45; color: #9ec9c0;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,180,0.03) 2px, rgba(0,255,180,0.03) 4px), rgba(0,0,0,0.35);
  border: 1px solid var(--rsv-line); padding: 8px; min-height: 64px; max-height: 90px; overflow: auto;
}}

#rsvis-boot {{ display: none !important; width: 0 !important; height: 0 !important; overflow: hidden !important; }}

@media (max-width: 1100px) {{
  .rsvis-dock-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .rsvis-card .btns.grid5 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
}}
""".strip()


def _top_chrome_html(*, module_id: str, portal: str, scene_info: str, status_text: str) -> str:
    look = MODULE_LOOK[module_id]
    return f"""
<div id="rsvis-chrome">
  <div id="rsvis-top">
    <div class="rsvis-side">
      <a class="rsvis-chip" href="{_esc(portal)}">RETURN PORTAL</a>
    </div>
    <div class="rsvis-title-trap">
      <div class="code" id="rsvis-title-code">{SOFTWARE_SHORT} // {SOFTWARE_VERSION}</div>
      <div class="sub" id="rsvis-title-sub">MODULE: {look['label']}</div>
      <div class="status" id="rsvis-status-led">{_esc(status_text)}</div>
    </div>
    <div class="rsvis-side right">
      <div class="rsvis-scene-box">SCENE INFO<b id="rsvis-scene-info">{_esc(scene_info)}</b></div>
      <div class="rsvis-sig"><span></span><span></span><span></span> LINK</div>
    </div>
  </div>
</div>
""".strip()


def _wrap_dock_shell(inner_html: str, *, title: str) -> str:
    """Hover strip + small tab; body expands on mouse enter (no click)."""
    return f"""
<div id="rsvis-dock-shell">
  <div id="rsvis-dock-hit">
    <div id="rsvis-dock-tab"><span class="chev">▴</span>&nbsp; {_esc(title)} · HOVER TO EXPAND&nbsp; <span class="chev">▴</span></div>
    <div id="rsvis-dock-body">
{inner_html}
    </div>
  </div>
</div>
""".strip()


def _skills_dock_html() -> str:
    try:
        from newtest.skills import SKILLS
    except Exception:
        SKILLS = []  # type: ignore
    skill_btns = []
    for i, s in enumerate(SKILLS):
        skill_btns.append(
            f'<button type="button" class="rsvis-btn accent block" data-click="S{i+1}">'
            f"{i+1}. {_esc(s.label)}</button>"
        )
    skills_grid = "".join(skill_btns) or '<span class="hint">no skills</span>'
    inner = f"""
<div id="rsvis-dock" data-module="skills">
  <div class="rsvis-card">
    <h3>Skill Actions</h3>
    <div class="hint">CLICK / KEYS 1-5 · CAM: WASD / ARROWS</div>
    <div class="btns grid5">{skills_grid}</div>
  </div>
  <div class="rsvis-dock-row cols-4">
    <div class="rsvis-card">
      <h3>Commands</h3>
      <div class="btns grid2">
        <button type="button" class="rsvis-btn" data-click="RESET" style="border-color:#ff8c28">RESET</button>
        <button type="button" class="rsvis-btn red" data-click="STOP">STOP</button>
        <button type="button" class="rsvis-btn green" data-click="REPLAY">REPLAY</button>
        <button type="button" class="rsvis-btn cyan" data-click="SHOT">SHOT</button>
      </div>
    </div>
    <div class="rsvis-card">
      <h3>Camera</h3>
      <div class="hint">锁定=跟随机器人 · 自由=WASD/鼠标 · 当前项带 ON 标记</div>
      <div class="btns grid2" id="rsvis-cam-mode">
        <button type="button" class="rsvis-btn green is-on" data-click="CAM LOCK" data-cam-mode="lock">锁定相机</button>
        <button type="button" class="rsvis-btn cyan" data-click="CAM FREE" data-cam-mode="free">自由相机</button>
        <button type="button" class="rsvis-btn block" data-click="RESET VIEW" style="grid-column:1/-1">RESET VIEW</button>
      </div>
    </div>
    <div class="rsvis-card" style="grid-column: span 2">
      <h3>Status Log</h3>
      <div class="rsvis-log" id="rsvis-dock-log">&gt; skill stage online<br/>&gt; await command</div>
    </div>
  </div>
</div>
""".strip()
    return _wrap_dock_shell(inner, title="CONTROL DOCK / 技能控制")


def _nav_dock_html() -> str:
    inner = """
<div id="rsvis-dock" data-module="nav">
  <div class="rsvis-dock-row">
    <div class="rsvis-card">
      <h3>Scene Selection</h3>
      <div class="hint">PICK THEN LOAD</div>
      <div class="btns">
        <button type="button" class="rsvis-btn block" data-click="SCENE corridor">CORRIDOR</button>
        <button type="button" class="rsvis-btn block" data-click="SCENE fence">FENCE</button>
        <button type="button" class="rsvis-btn block" data-click="SCENE open8">OPEN8</button>
        <button type="button" class="rsvis-btn purple block" data-click="LOAD SCENE">LOAD SCENE</button>
        <button type="button" class="rsvis-btn block" data-click="RESET" style="border-color:#ff8c28">RESET</button>
      </div>
    </div>
    <div class="rsvis-card">
      <h3>Velocity</h3>
      <div class="hint">T/Y KEYS ALSO WORK</div>
      <div class="btns">
        <button type="button" class="rsvis-btn cyan block" data-click="SLOW">SLOW</button>
        <button type="button" class="rsvis-btn cyan block" data-click="NORMAL">NORMAL</button>
        <button type="button" class="rsvis-btn green block" data-click="FAST">FAST</button>
      </div>
    </div>
    <div class="rsvis-card">
      <h3>Target · 终点</h3>
      <div class="hint">也可在 3D 里点地面设终点</div>
      <div class="btns">
        <button type="button" class="rsvis-btn cyan block" data-click="PREVIEW ENDPOINT">预览终点</button>
        <button type="button" class="rsvis-btn block" data-click="BOOKMARK">收藏当前终点</button>
        <button type="button" class="rsvis-btn purple block" data-click="GOTO BOOKMARK">前往收藏终点</button>
      </div>
    </div>
    <div class="rsvis-card">
      <h3>Action Commands</h3>
      <div class="btns grid2">
        <button type="button" class="rsvis-btn green" data-click="START NAV">START NAV</button>
        <button type="button" class="rsvis-btn red" data-click="STOP">STOP</button>
        <button type="button" class="rsvis-btn cyan" data-click="CLEAR ENDPOINT">CLEAR</button>
        <button type="button" class="rsvis-btn" data-click="PREVIEW ENDPOINT">PREVIEW</button>
      </div>
    </div>
    <div class="rsvis-card">
      <h3>Status Log</h3>
      <div class="rsvis-log" id="rsvis-dock-log">&gt; nav online<br/>&gt; Space start/stop · click ground</div>
      <div class="btns" style="margin-top:6px">
        <button type="button" class="rsvis-btn block" data-click="TOGGLE LIDAR">TOGGLE LIDAR</button>
      </div>
    </div>
  </div>
</div>
""".strip()
    return _wrap_dock_shell(inner, title="CONTROL DOCK / 导航控制")


def _play_dock_html() -> str:
    inner = """
<div id="rsvis-dock" data-module="play">
  <div class="rsvis-dock-row">
    <div class="rsvis-card">
      <h3>Move · 遥控</h3>
      <div class="hint">HOLD WASD/QE = FIXED SPEED · RELEASE = STOP</div>
      <div class="btns grid2">
        <button type="button" class="rsvis-btn green" data-click="MOVE_FWD">W FWD</button>
        <button type="button" class="rsvis-btn green" data-click="MOVE_BACK">S BACK</button>
        <button type="button" class="rsvis-btn cyan" data-click="MOVE_LEFT">A LEFT</button>
        <button type="button" class="rsvis-btn cyan" data-click="MOVE_RIGHT">D RIGHT</button>
        <button type="button" class="rsvis-btn" data-click="YAW_L">Q YAW-</button>
        <button type="button" class="rsvis-btn" data-click="YAW_R">E YAW+</button>
      </div>
    </div>
    <div class="rsvis-card">
      <h3>Gait · 步态</h3>
      <div class="hint">KEYS 1-4 · 单选高亮</div>
      <div class="btns">
        <button type="button" class="rsvis-btn green block is-on" data-click="GAIT_1" data-radio="gait" data-radio-val="1">1 TROT</button>
        <button type="button" class="rsvis-btn block" data-click="GAIT_2" data-radio="gait" data-radio-val="2">2 BOUND</button>
        <button type="button" class="rsvis-btn block" data-click="GAIT_3" data-radio="gait" data-radio-val="3">3 PACE</button>
        <button type="button" class="rsvis-btn block" data-click="GAIT_4" data-radio="gait" data-radio-val="4">4 PRONK</button>
      </div>
    </div>
    <div class="rsvis-card">
      <h3>Tune · 档位</h3>
      <div class="hint">5-7 CAP · ZXC PERIOD · VNM HEIGHT</div>
      <div class="btns grid2">
        <button type="button" class="rsvis-btn cyan" data-click="×0.3" data-radio="cap" data-radio-val="0.3">CAP 0.3</button>
        <button type="button" class="rsvis-btn cyan" data-click="×0.7" data-radio="cap" data-radio-val="0.7">CAP 0.7</button>
        <button type="button" class="rsvis-btn green is-on" data-click="×1.0" data-radio="cap" data-radio-val="1.0">CAP 1.0</button>
        <button type="button" class="rsvis-btn" data-click="PERIOD_SLOW" data-radio="period" data-radio-val="slow">PERIOD SLOW</button>
        <button type="button" class="rsvis-btn is-on" data-click="PERIOD_MID" data-radio="period" data-radio-val="mid">PERIOD MID</button>
        <button type="button" class="rsvis-btn" data-click="PERIOD_FAST" data-radio="period" data-radio-val="fast">PERIOD FAST</button>
        <button type="button" class="rsvis-btn" data-click="HEIGHT_LOW" data-radio="height" data-radio-val="low">HEIGHT LOW</button>
        <button type="button" class="rsvis-btn is-on" data-click="HEIGHT_MID" data-radio="height" data-radio-val="mid">HEIGHT MID</button>
        <button type="button" class="rsvis-btn" data-click="HEIGHT_HIGH" data-radio="height" data-radio-val="high">HEIGHT HIGH</button>
      </div>
    </div>
    <div class="rsvis-card">
      <h3>Camera · Action</h3>
      <div class="hint">自由相机后可用 IJKL/UO · 方向键</div>
      <div class="btns grid2" id="rsvis-cam-mode">
        <button type="button" class="rsvis-btn green is-on" data-click="CAM LOCK" data-radio="cam" data-radio-val="lock" data-cam-mode="lock">锁定相机</button>
        <button type="button" class="rsvis-btn cyan" data-click="CAM FREE" data-radio="cam" data-radio-val="free" data-cam-mode="free">自由相机</button>
        <button type="button" class="rsvis-btn red" data-click="STOP / BRAKE">BRAKE</button>
        <button type="button" class="rsvis-btn" data-click="RESET STAND" style="border-color:#ff8c28">RESET</button>
        <button type="button" class="rsvis-btn cyan" data-click="FLIP FORWARD">FLIP</button>
        <button type="button" class="rsvis-btn" data-click="SNAPSHOT">SHOT</button>
      </div>
    </div>
    <div class="rsvis-card">
      <h3>Status Log</h3>
      <div class="rsvis-log" id="rsvis-dock-log">&gt; teleop deck online<br/>&gt; WASD dog · IJKL cam (free)</div>
    </div>
  </div>
</div>
""".strip()
    return _wrap_dock_shell(inner, title="CONTROL DOCK / 遥控甲板")


def gauge_svg(*, value: float = 0.35, label: str = "EST", accent: str = "#00d2e6", gid: str = "g") -> str:
    import math

    v = max(0.0, min(1.0, float(value)))
    ang = -120 + 240 * v
    rad = math.radians(ang)
    x = 36 + 26 * math.sin(rad)
    y = 40 - 26 * math.cos(rad)
    return f"""
<svg class="rsvis-gauge" viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg" style="width:72px;height:72px;display:block;margin:4px auto">
  <defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="{accent}" stop-opacity="0.9"/>
    <stop offset="100%" stop-color="#ffffff" stop-opacity="0.2"/>
  </linearGradient></defs>
  <circle cx="36" cy="40" r="30" fill="rgba(0,0,0,0.35)" stroke="rgba(180,210,230,0.25)" stroke-width="1"/>
  <path d="M12 48 A28 28 0 0 1 60 48" fill="none" stroke="rgba(0,200,255,0.2)" stroke-width="4" stroke-linecap="round"/>
  <path d="M12 48 A28 28 0 0 1 {x:.1f} {y:.1f}" fill="none" stroke="url(#{gid})" stroke-width="4" stroke-linecap="round"/>
  <line x1="36" y1="40" x2="{x:.1f}" y2="{y:.1f}" stroke="{accent}" stroke-width="2"/>
  <circle cx="36" cy="40" r="3" fill="{accent}"/>
  <text x="36" y="62" text-anchor="middle" fill="{accent}" font-size="8" font-family="Share Tech Mono, monospace">{label}</text>
</svg>
""".strip()


def meter_html(*, level: int = 7, total: int = 16) -> str:
    bars = []
    for i in range(total):
        cls = "on" if i < level else ""
        h = 4 + int(14 * (0.3 + 0.7 * (i / max(1, total - 1))))
        bars.append(f'<i class="{cls}" style="display:inline-block;width:4px;height:{h}px;margin-right:2px;background:linear-gradient(180deg,var(--rsv-accent),transparent);opacity:{0.95 if cls else 0.3}"></i>')
    return f'<div style="display:flex;align-items:flex-end;height:18px">{"".join(bars)}</div>'


def card_head(title: str) -> str:
    return f'<div style="display:none">{_esc(title)}</div>'


def log_box(lines: Tuple[str, ...] | list, *, sel: Optional[dict] = None) -> str:
    body = "<br/>".join(_esc(x) for x in lines)
    extra = ""
    if sel:
        for k, v in sel.items():
            extra += f' data-sel-{_esc(k)}="{_esc(v)}"'
    return f'<div class="rsvis-log" data-rsvis-src-log="1"{extra}>{body}</div>'


def banner(title: str, detail: str) -> str:
    return f'<div style="display:none">{_esc(title)}</div>'


def action_hint(text: str) -> str:
    return f'<div style="display:none">{_esc(text)}</div>'


def pill(text: str) -> str:
    return f"<span>{_esc(text)}</span>"


def dual_gauge(
    *,
    left_value: float,
    right_value: float,
    left_label: str = "VX",
    right_label: str = "VY",
    accent: str = "#64e650",
) -> str:
    return (
        '<div style="display:flex;gap:8px;justify-content:center">'
        + gauge_svg(value=left_value, label=left_label, accent=accent, gid="gx")
        + gauge_svg(value=right_value, label=right_label, accent="#00c8ff", gid="gy")
        + "</div>"
    )


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _dock_html(module_id: str) -> str:
    if module_id == "skills":
        return _skills_dock_html()
    if module_id == "nav":
        return _nav_dock_html()
    if module_id == "play":
        return _play_dock_html()
    return _skills_dock_html()


def inject_hud(
    server,
    *,
    module_id: str,
    portal_url: Optional[str] = None,
    scene_info: str = "READY",
    status_text: str = "STATUS: IDLE",
) -> None:
    """Hide native Viser panel; inject top chrome + custom horizontal dock; proxy clicks."""
    look = MODULE_LOOK[module_id]
    portal = portal_url or PORTAL_DEFAULT_URL
    scene_ascii = "".join(ch if ord(ch) < 128 else "-" for ch in str(scene_info)) or "READY"
    status_ascii = "".join(ch if ord(ch) < 128 else " " for ch in str(status_text)).strip() or "STATUS: IDLE"

    css = _css(module_id, look["hex"], look["hex2"])
    chrome = _top_chrome_html(
        module_id=module_id,
        portal=portal,
        scene_info=scene_ascii,
        status_text=status_ascii,
    )
    dock = _dock_html(module_id)

    css_safe = css.replace("</style>", "<\\/style>")
    try:
        server.gui.add_html(f'<style id="rsvis-style">{css_safe}</style>')
    except Exception as exc:
        print(f"[hud] style inject failed: {exc}", flush=True)

    title_cn = f"{SOFTWARE_SHORT} // {SOFTWARE_VERSION} [{SOFTWARE_NAME}]"
    sub_cn = f"MODULE: {look['label']} [{look['label_cn']}]"
    status_cn = str(status_text)
    scene_cn = str(scene_info)

    boot_js = f"""
(function(){{
  if (window.__RSVIS_HUD__) return;
  window.__RSVIS_HUD__ = true;

  function b64utf8(b64) {{
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    try {{ return new TextDecoder('utf-8').decode(bytes); }}
    catch (e) {{
      try {{ return decodeURIComponent(escape(bin)); }} catch (e2) {{ return bin; }}
    }}
  }}

  function mount(html) {{
    var wrap = document.createElement('div');
    wrap.innerHTML = html;
    var node = wrap.firstElementChild;
    if (node) document.body.appendChild(node);
    return node;
  }}

  try {{
    if (!document.getElementById('rsvis-chrome')) mount(b64utf8('{_b64(chrome)}'));
    if (!document.getElementById('rsvis-dock-shell')) mount(b64utf8('{_b64(dock)}'));
    var elTitle = document.getElementById('rsvis-title-code');
    var elSub = document.getElementById('rsvis-title-sub');
    var elStatus = document.getElementById('rsvis-status-led');
    var elScene = document.getElementById('rsvis-scene-info');
    if (elTitle) elTitle.textContent = b64utf8('{_b64(title_cn)}');
    if (elSub) elSub.textContent = b64utf8('{_b64(sub_cn)}');
    if (elStatus) elStatus.textContent = b64utf8('{_b64(status_cn)}');
    if (elScene) elScene.textContent = b64utf8('{_b64(scene_cn)}');
  }} catch (e) {{ console.warn('rsvis mount', e); }}

  function forceStyle(el, props) {{
    if (!el || !el.style) return;
    Object.keys(props).forEach(function(k) {{
      try {{ el.style.setProperty(k, props[k], 'important'); }} catch (e) {{}}
    }});
  }}

  function isControlPaper(p) {{
    if (!p || (p.closest && p.closest('#rsvis-chrome,#rsvis-dock,#rsvis-dock-shell'))) return false;
    if (p.querySelector && p.querySelector('canvas')) return false;
    if (!p.querySelector('button, input, .mantine-Accordion-item, .mantine-Slider-root')) return false;
    var r = p.getBoundingClientRect();
    if (r.width > window.innerWidth * 0.92 && r.height > window.innerHeight * 0.55) return false;
    return true;
  }}

  function findPanel() {{
    var papers = Array.prototype.slice.call(document.querySelectorAll('.mantine-Paper-root'));
    var best = null, bestScore = -1;
    papers.forEach(function(p) {{
      if (!isControlPaper(p)) return;
      var n = p.querySelectorAll('button').length;
      var score = n * 1000 + (p.classList.contains('rsvis-hidden-gui') ? 1e9 : 0);
      if (score > bestScore) {{ bestScore = score; best = p; }}
    }});
    return best;
  }}

  function hideNativeGui() {{
    var paper = findPanel();
    if (!paper) return null;
    paper.classList.add('rsvis-hidden-gui');
    var wrap = paper.parentElement;
    // Only hide immediate float wrapper if it does not contain canvas
    if (wrap && !(wrap.querySelector && wrap.querySelector('canvas')) && wrap !== document.body && wrap.id !== 'root') {{
      var st = window.getComputedStyle(wrap);
      var r = wrap.getBoundingClientRect();
      if ((st.position === 'absolute' || st.position === 'fixed') &&
          !(r.width > window.innerWidth * 0.9 && r.height > window.innerHeight * 0.5)) {{
        wrap.classList.add('rsvis-hidden-gui-wrap');
        forceStyle(wrap, {{
          'position': 'fixed', 'left': '-14000px', 'top': '0',
          'width': '520px', 'height': '980px', 'opacity': '0',
          'pointer-events': 'none', 'z-index': '-5', 'transform': 'none'
        }});
      }}
    }}
    forceStyle(paper, {{
      'position': 'fixed', 'left': '-14000px', 'top': '0',
      'width': '520px', 'height': '980px', 'opacity': '0',
      'pointer-events': 'none', 'z-index': '-5', 'transform': 'none'
    }});
    return paper;
  }}

  function clickHidden(label) {{
    var want = (label || '').trim();
    if (!want) return false;
    var root = document.querySelector('.rsvis-hidden-gui') || findPanel();
    if (!root) root = document.body;
    var buttons = root.querySelectorAll('button');
    var exact = null, soft = null;
    for (var i = 0; i < buttons.length; i++) {{
      var t = (buttons[i].textContent || '').replace(/\\s+/g, ' ').trim();
      if (t === want) {{ exact = buttons[i]; break; }}
      if (!soft && t.indexOf(want) >= 0) soft = buttons[i];
    }}
    var btn = exact || soft;
    if (!btn) {{
      // fallback: whole document (still skip dock/chrome)
      buttons = document.querySelectorAll('button');
      for (var j = 0; j < buttons.length; j++) {{
        if (buttons[j].closest && buttons[j].closest('#rsvis-dock,#rsvis-dock-shell,#rsvis-chrome')) continue;
        var tt = (buttons[j].textContent || '').replace(/\\s+/g, ' ').trim();
        if (tt === want || tt.indexOf(want) >= 0) {{ btn = buttons[j]; break; }}
      }}
    }}
    if (!btn) {{ console.warn('rsvis click miss', want); return false; }}
    try {{ btn.click(); return true; }} catch (e) {{ console.warn(e); return false; }}
  }}

  function setCamMode(mode) {{
    var box = document.getElementById('rsvis-cam-mode');
    if (!box) return;
    box.querySelectorAll('[data-cam-mode]').forEach(function(b) {{
      if (b.getAttribute('data-cam-mode') === mode) b.classList.add('is-on');
      else b.classList.remove('is-on');
    }});
  }}

  function setRadio(group, val) {{
    if (!group) return;
    var dock = document.getElementById('rsvis-dock');
    if (!dock) return;
    dock.querySelectorAll('[data-radio="' + group + '"]').forEach(function(b) {{
      if (String(b.getAttribute('data-radio-val')) === String(val)) b.classList.add('is-on');
      else b.classList.remove('is-on');
    }});
    if (group === 'cam') setCamMode(val === 'free' ? 'free' : 'lock');
  }}

  function applySelFromLog(src) {{
    if (!src || !src.getAttribute) return;
    ['gait','cap','period','height','cam'].forEach(function(k) {{
      var v = src.getAttribute('data-sel-' + k);
      if (v != null && v !== '') setRadio(k, v);
    }});
    // play 默认速度（与 Python 推导一致）
    var dvx = src.getAttribute('data-sel-defvx');
    var dvy = src.getAttribute('data-sel-defvy');
    var dyaw = src.getAttribute('data-sel-defyaw');
    if (dvx) window.__RSVIS_DEF_VX = parseFloat(dvx);
    if (dvy) window.__RSVIS_DEF_VY = parseFloat(dvy);
    if (dyaw) window.__RSVIS_DEF_YAW = parseFloat(dyaw);
  }}

  function bindDock() {{
    var dock = document.getElementById('rsvis-dock');
    if (!dock || dock.__bound) return;
    dock.__bound = true;
    // default: locked follow (skills/play); nav has no cam mode
    setCamMode('lock');
    dock.addEventListener('click', function(ev) {{
      var t = ev.target;
      while (t && t !== dock && !(t.getAttribute && t.getAttribute('data-click'))) t = t.parentElement;
      if (!t || t === dock) return;
      var label = t.getAttribute('data-click');
      hideNativeGui();
      clickHidden(label);
      var rg = t.getAttribute('data-radio');
      var rv = t.getAttribute('data-radio-val');
      if (rg && rv != null) setRadio(rg, rv);
      if (label === 'CAM LOCK') setCamMode('lock');
      if (label === 'CAM FREE') setCamMode('free');
      if (label === 'RESET VIEW') setCamMode('lock');
    }});
  }}

  function syncLog() {{
    var dst = document.getElementById('rsvis-dock-log');
    if (!dst) return;
    var src = document.querySelector('.rsvis-hidden-gui [data-rsvis-src-log], .rsvis-hidden-gui .rsvis-log');
    if (src && src.innerHTML) {{
      dst.innerHTML = src.innerHTML;
      applySelFromLog(src);
    }}
  }}

  // play 页：按住 WASD/QE = 固定速度；松手清零；IJKL = 相机
  var MODULE_ID = '{module_id}';
  if (MODULE_ID === 'play') {{
    var TIMES = String.fromCharCode(0xd7); // ×
    var HOLD_KEYS = {{ KeyW:1, KeyS:1, KeyA:1, KeyD:1, KeyQ:1, KeyE:1 }};
    var heldMove = {{}};
    var ACTION_MAP = {{
      KeyB: 'STOP / BRAKE', Space: 'STOP / BRAKE',
      Digit1: 'GAIT_1', Digit2: 'GAIT_2', Digit3: 'GAIT_3', Digit4: 'GAIT_4',
      Digit5: TIMES + '0.3', Digit6: TIMES + '0.7', Digit7: TIMES + '1.0',
      KeyZ: 'PERIOD_SLOW', KeyX: 'PERIOD_MID', KeyC: 'PERIOD_FAST',
      KeyV: 'HEIGHT_LOW', KeyN: 'HEIGHT_MID', KeyM: 'HEIGHT_HIGH',
      KeyR: 'RESET STAND', KeyF: 'FLIP FORWARD'
    }};
    var ACTION_RADIO = {{
      Digit1: ['gait','1'], Digit2: ['gait','2'], Digit3: ['gait','3'], Digit4: ['gait','4'],
      Digit5: ['cap','0.3'], Digit6: ['cap','0.7'], Digit7: ['cap','1.0'],
      KeyZ: ['period','slow'], KeyX: ['period','mid'], KeyC: ['period','fast'],
      KeyV: ['height','low'], KeyN: ['height','mid'], KeyM: ['height','high']
    }};
    var CAM_MAP = {{
      KeyI: 'KeyW', KeyK: 'KeyS', KeyJ: 'KeyA', KeyL: 'KeyD',
      KeyU: 'KeyQ', KeyO: 'KeyE'
    }};
    var CAM_KEY = {{ KeyW: 'w', KeyA: 'a', KeyS: 's', KeyD: 'd', KeyQ: 'q', KeyE: 'e' }};

    function isTypingTarget(el) {{
      if (!el) return false;
      var t = (el.tagName || '').toLowerCase();
      if (t === 'input' || t === 'textarea' || t === 'select') return true;
      if (el.isContentEditable) return true;
      return !!(el.closest && el.closest('input, textarea, select, [contenteditable="true"]'));
    }}

    function fireLabel(label) {{
      hideNativeGui();
      if (!clickHidden(label)) console.warn('rsvis key miss', label);
    }}

    function setTeleopCmd(vx, vy, yaw) {{
      hideNativeGui();
      var root = document.querySelector('.rsvis-hidden-gui') || findPanel() || document.body;
      var inputs = root.querySelectorAll('input');
      var want = Number(vx).toFixed(3) + ',' + Number(vy).toFixed(3) + ',' + Number(yaw).toFixed(3);
      for (var i = 0; i < inputs.length; i++) {{
        var el = inputs[i];
        if (el.disabled || el.type === 'checkbox' || el.type === 'range') continue;
        var wrap = el.closest('[class*="InputWrapper"], [class*="input-wrapper"], label, div') || el.parentElement;
        var probe = (wrap && wrap.textContent) ? wrap.textContent : '';
        // 向上多找两层，匹配 TELEOP_CMD 标签
        var p = el.parentElement, hit = false;
        for (var k = 0; k < 5 && p; k++) {{
          var t = (p.textContent || '');
          if (t.indexOf('TELEOP_CMD') >= 0) {{ hit = true; break; }}
          p = p.parentElement;
        }}
        if (!hit && probe.indexOf('TELEOP') < 0) continue;
        try {{
          var desc = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
          if (desc && desc.set) desc.set.call(el, want);
          else el.value = want;
          el.dispatchEvent(new Event('input', {{ bubbles: true }}));
          el.dispatchEvent(new Event('change', {{ bubbles: true }}));
          return true;
        }} catch (e) {{ console.warn('teleop set', e); }}
      }}
      // 回退：逐轴点按钮
      if (Math.abs(vx) < 1e-6) fireLabel('ZERO_VX');
      else if (vx > 0) fireLabel('MOVE_FWD');
      else fireLabel('MOVE_BACK');
      if (Math.abs(vy) < 1e-6) fireLabel('ZERO_VY');
      else if (vy > 0) fireLabel('MOVE_LEFT');
      else fireLabel('MOVE_RIGHT');
      if (Math.abs(yaw) < 1e-6) fireLabel('ZERO_YAW');
      else if (yaw > 0) fireLabel('YAW_L');
      else fireLabel('YAW_R');
      return false;
    }}

    // 默认速度：优先用服务端下发，否则中等偏快常量
    function defVx() {{ return (window.__RSVIS_DEF_VX > 0 ? window.__RSVIS_DEF_VX : 0.40); }}
    function defVy() {{ return (window.__RSVIS_DEF_VY > 0 ? window.__RSVIS_DEF_VY : 0.45); }}
    function defYaw() {{ return (window.__RSVIS_DEF_YAW > 0 ? window.__RSVIS_DEF_YAW : 0.60); }}

    function syncHoldMove() {{
      var vx = 0, vy = 0, yaw = 0;
      if (heldMove.KeyW && !heldMove.KeyS) vx = defVx();
      else if (heldMove.KeyS && !heldMove.KeyW) vx = -defVx();
      if (heldMove.KeyA && !heldMove.KeyD) vy = defVy();
      else if (heldMove.KeyD && !heldMove.KeyA) vy = -defVy();
      if (heldMove.KeyQ && !heldMove.KeyE) yaw = defYaw();
      else if (heldMove.KeyE && !heldMove.KeyQ) yaw = -defYaw();
      setTeleopCmd(vx, vy, yaw);
    }}

    function clearHoldMove() {{
      heldMove = {{}};
      setTeleopCmd(0, 0, 0);
    }}

    function synthCam(type, code) {{
      window.__rsvisCamSynth = true;
      try {{
        var ev = new KeyboardEvent(type, {{
          key: CAM_KEY[code] || '',
          code: code,
          bubbles: true,
          cancelable: true,
          view: window
        }});
        document.dispatchEvent(ev);
      }} catch (e) {{}}
      window.__rsvisCamSynth = false;
    }}

    function eat(ev) {{
      ev.preventDefault();
      ev.stopPropagation();
      if (typeof ev.stopImmediatePropagation === 'function') ev.stopImmediatePropagation();
    }}

    window.addEventListener('keydown', function(ev) {{
      if (window.__rsvisCamSynth) return;
      if (isTypingTarget(ev.target)) return;
      if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
      if (HOLD_KEYS[ev.code]) {{
        eat(ev);
        if (ev.repeat) return;
        heldMove[ev.code] = 1;
        syncHoldMove();
        return;
      }}
      var act = ACTION_MAP[ev.code];
      if (act) {{
        eat(ev);
        if (ev.repeat) return;
        if (ev.code === 'KeyB' || ev.code === 'Space') clearHoldMove();
        fireLabel(act);
        var rad = ACTION_RADIO[ev.code];
        if (rad) setRadio(rad[0], rad[1]);
        return;
      }}
      var camCode = CAM_MAP[ev.code];
      if (camCode) {{
        eat(ev);
        var freeBtn = document.querySelector('#rsvis-cam-mode [data-cam-mode="free"]');
        var locked = freeBtn && !freeBtn.classList.contains('is-on');
        if (locked) {{
          fireLabel('CAM FREE');
          setRadio('cam', 'free');
        }}
        if (!ev.repeat) synthCam('keydown', camCode);
      }}
    }}, true);

    window.addEventListener('keyup', function(ev) {{
      if (window.__rsvisCamSynth) return;
      if (isTypingTarget(ev.target)) return;
      if (HOLD_KEYS[ev.code]) {{
        eat(ev);
        delete heldMove[ev.code];
        syncHoldMove();
        return;
      }}
      var camCode = CAM_MAP[ev.code];
      if (camCode) {{
        eat(ev);
        synthCam('keyup', camCode);
      }}
    }}, true);

    window.addEventListener('blur', function() {{
      clearHoldMove();
      ['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE'].forEach(function(c) {{ synthCam('keyup', c); }});
    }});
  }}

  hideNativeGui();
  bindDock();
  setInterval(function() {{ hideNativeGui(); bindDock(); syncLog(); }}, 700);
}})();
""".strip()

    boot_b64 = _b64(boot_js)
    loader = (
        '<div id="rsvis-boot" class="rsvis-boot">'
        f'<img alt="" width="0" height="0" '
        f'src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" '
        f"onload=\"try{{(new Function(atob('{boot_b64}')))();}}catch(e){{console.warn(e);}}this.remove();\" "
        f"onerror=\"try{{(new Function(atob('{boot_b64}')))();}}catch(e){{}}this.remove();\" />"
        "</div>"
    )
    try:
        server.gui.add_html(loader)
    except Exception as exc:
        print(f"[hud] boot inject failed: {exc}", flush=True)
