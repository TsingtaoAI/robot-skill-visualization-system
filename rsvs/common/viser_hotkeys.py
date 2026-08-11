"""Viser Command Palette 热键注册封装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence


@dataclass(frozen=True)
class HotkeySpec:
    """一条与说明书对齐的快捷键定义。"""

    label: str
    callback: Callable[[], None]
    hotkey: Optional[str] = None
    modifier: Optional[str] = None
    description: Optional[str] = None


def register_hotkeys(server, specs: Sequence[HotkeySpec]) -> List[object]:
    """批量注册 ``gui.add_command`` 热键，返回 handles。"""
    handles: List[object] = []
    gui = server.gui
    for spec in specs:
        try:
            handle = gui.add_command(
                spec.label,
                description=spec.description,
                hotkey=spec.hotkey,
                modifier=spec.modifier,
            )
        except Exception as exc:
            print(f"[hotkeys] 注册失败 {spec.label}: {exc}", flush=True)
            continue

        @handle.on_trigger
        def _(_e, cb=spec.callback) -> None:
            try:
                cb()
            except Exception as err:
                print(f"[hotkeys] 触发失败 {spec.label}: {err}", flush=True)

        handles.append(handle)
    return handles


def hotkey_help_markdown(specs: Sequence[HotkeySpec], *, title: str = "键盘快捷键") -> str:
    """生成说明书风格的快捷键 Markdown。"""
    lines = [f"**{title}**", ""]
    for s in specs:
        key = s.hotkey or "—"
        if s.modifier:
            key = f"{s.modifier}+{key}"
        desc = s.description or s.label
        lines.append(f"- `{key}` — {desc}")
    return "\n".join(lines)
