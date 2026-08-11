"""三维演示窗主题：对接 R.S.V.I.S. 科幻 HUD。"""

from __future__ import annotations

from typing import Optional, Tuple

from newtest.common.viser_hud import MODULE_LOOK, SOFTWARE_NAME, SOFTWARE_VERSION, inject_hud

PORTAL_DEFAULT_URL = "http://127.0.0.1:8090"
BRAND_COLOR = (0, 200, 255)


class ModuleTheme:
    def __init__(self, module_id: str):
        look = MODULE_LOOK[module_id]
        self.module_id = module_id
        self.label = look["label_cn"]
        self.accent: Tuple[int, int, int] = look["accent"]
        self.panel_label = look["panel"]


def apply_product_theme(
    server,
    *,
    module_label: str,
    portal_url: Optional[str] = None,
    dark_mode: bool = True,
    module_id: Optional[str] = None,
    scene_info: str = "READY",
    status_text: str = "STATUS: IDLE",
) -> ModuleTheme:
    """深色科幻主题 + 顶栏 HUD + 底坞吸底。"""
    mid = module_id or "skills"
    if mid not in MODULE_LOOK:
        for k, v in MODULE_LOOK.items():
            if v["label_cn"] == module_label or module_label in v["label"]:
                mid = k
                break
        else:
            mid = "skills"

    mt = ModuleTheme(mid)
    look = MODULE_LOOK[mid]
    portal = portal_url or PORTAL_DEFAULT_URL

    try:
        server.gui.set_panel_label(mt.panel_label)
    except Exception:
        pass

    try:
        server.gui.configure_theme(
            titlebar_content=None,
            control_layout="floating",
            control_width="large",
            dark_mode=True,
            show_logo=False,
            show_share_button=False,
            brand_color=look["accent"],
        )
    except Exception as exc:
        print(f"[theme] configure_theme 失败: {exc}", flush=True)

    inject_hud(
        server,
        module_id=mid,
        portal_url=portal,
        scene_info=scene_info,
        status_text=status_text,
    )

    try:
        if mid == "nav":
            server.scene.configure_fog(14.0, 60.0, color=(6, 12, 18), enabled=True)
        server.scene.configure_default_lights(True)
    except Exception:
        pass

    return mt
