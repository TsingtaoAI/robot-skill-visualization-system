"""状态 Markdown / 进度条 / 操作日志面板。"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional


class StatusPanel:
    """统一状态区：标题文案 + 可选进度条 + 最近操作日志。"""

    def __init__(
        self,
        server,
        *,
        folder_label: str = "运行状态",
        log_size: int = 8,
        show_progress: bool = True,
    ):
        self._server = server
        self._logs: Deque[str] = deque(maxlen=max(3, int(log_size)))
        self._headline = "待机"
        self._detail = ""
        self._busy = False
        self._progress_value = 0.0

        with server.gui.add_folder(folder_label, expand_by_default=True):
            self._md = server.gui.add_markdown(self._render())
            self._progress = None
            if show_progress:
                self._progress = server.gui.add_progress_bar(
                    value=0.0,
                    animated=False,
                    visible=True,
                )

    def _render(self) -> str:
        lock = "忙碌" if self._busy else "空闲"
        lines = [
            f"**状态**：{self._headline}  |  **锁**：{lock}",
            "",
        ]
        if self._detail:
            lines.append(self._detail)
            lines.append("")
        if self._logs:
            lines.append("**最近操作**")
            for item in list(self._logs)[-6:]:
                lines.append(f"- {item}")
        return "\n".join(lines)

    def _flush(self) -> None:
        self._md.content = self._render()
        if self._progress is not None:
            try:
                self._progress.value = float(self._progress_value)
            except Exception:
                pass
            try:
                self._progress.animated = bool(self._busy)
            except Exception:
                pass

    def set_status(
        self,
        headline: str,
        *,
        detail: str = "",
        busy: Optional[bool] = None,
        progress: Optional[float] = None,
        log: Optional[str] = None,
    ) -> None:
        self._headline = headline
        if detail is not None:
            self._detail = detail
        if busy is not None:
            self._busy = bool(busy)
        if progress is not None:
            self._progress_value = max(0.0, min(100.0, float(progress)))
        if log:
            self._logs.append(log)
        self._flush()

    def log(self, text: str) -> None:
        self._logs.append(text)
        self._flush()

    def set_busy(self, busy: bool, *, progress: Optional[float] = None) -> None:
        self._busy = bool(busy)
        if progress is not None:
            self._progress_value = max(0.0, min(100.0, float(progress)))
        elif not busy:
            self._progress_value = 0.0
        self._flush()
