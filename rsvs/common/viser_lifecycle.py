"""Viser 生命周期：演示页全部关闭后自动结束进程。

规则：
- 至少有一个浏览器客户端连接过之后，才启用“关页退出”；
- 最后一个客户端断开后等待 grace_sec（默认 8s），避免刷新页面误杀；
- 宽限期内若重新连上则取消退出。
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional


def attach_exit_when_browser_closed(
    server: Any,
    *,
    grace_sec: float = 8.0,
    label: str = "demo",
) -> None:
    """挂到 ViserServer：关网页（无客户端）→ 自动退出本进程。"""

    state = {
        "ever_connected": False,
        "timer": None,  # type: Optional[threading.Timer]
        "lock": threading.Lock(),
        "exiting": False,
    }

    def _cancel_timer_locked() -> None:
        t = state["timer"]
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass
            state["timer"] = None

    def _do_exit() -> None:
        with state["lock"]:
            if state["exiting"]:
                return
            if not state["ever_connected"]:
                return
            try:
                clients = server.get_clients()
            except Exception:
                clients = {}
            if clients:
                return
            state["exiting"] = True

        print(
            f"[{label}] 演示网页已关闭（{grace_sec:.0f}s 内无重新连接），正在退出进程…",
            flush=True,
        )
        try:
            server.stop()
        except Exception:
            pass
        # 仿真主循环可能阻塞，强制结束本进程
        os._exit(0)

    def _schedule_exit() -> None:
        with state["lock"]:
            if state["exiting"]:
                return
            _cancel_timer_locked()
            timer = threading.Timer(float(grace_sec), _do_exit)
            timer.daemon = True
            state["timer"] = timer
            timer.start()

    @server.on_client_connect
    def _on_connect(_client) -> None:
        with state["lock"]:
            state["ever_connected"] = True
            _cancel_timer_locked()
        print(f"[{label}] 演示页已连接", flush=True)

    @server.on_client_disconnect
    def _on_disconnect(_client) -> None:
        # 断开瞬间客户端表可能尚未更新，稍后再由 Timer 复查
        print(
            f"[{label}] 演示页断开，{grace_sec:.0f}s 内无重连将自动退出",
            flush=True,
        )
        _schedule_exit()
