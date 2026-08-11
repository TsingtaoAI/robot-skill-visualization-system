"""产品门户：机器人技能可视化演示交互系统 V1.0

提供统一 Web 入口、一键启动演示服务、端口在线检测。

  ./newtest/run.sh hub
  ./newtest/run.sh hub --port 8090
"""

from __future__ import annotations

import newtest.bootstrap  # noqa: F401

import argparse
import json
import mimetypes
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

VERSION = "V1.0"
SOFTWARE_NAME = "机器人技能可视化演示交互系统"
ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
GIF_DIR = ROOT_DIR / "gif"
RUN_SH = ROOT_DIR / "run.sh"
LOG_DIR = ROOT_DIR / "logs"
ALLOWED_PORTS = {8081, 8082, 8083, 8090}

DEMO_SPECS: Dict[str, Dict[str, Any]] = {
    "skills": {"port": 8081, "label": "技能演示", "cmd": "skills"},
    "nav": {"port": 8082, "label": "导航演示", "cmd": "nav"},
    "play": {"port": 8083, "label": "运动遥控", "cmd": "play"},
}


def _probe_port(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class DemoLauncher:
    """在门户进程内后台拉起 skills/nav/play（继承当前 conda/Python 环境）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: Dict[str, subprocess.Popen] = {}
        self._log_handles: Dict[str, Any] = {}
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def status(self, demo_id: str) -> dict:
        spec = DEMO_SPECS[demo_id]
        port = int(spec["port"])
        up = _probe_port("127.0.0.1", port)
        with self._lock:
            proc = self._procs.get(demo_id)
            alive = proc is not None and proc.poll() is None
            pid = proc.pid if alive else None
            if proc is not None and not alive:
                self._cleanup_locked(demo_id)
        return {
            "id": demo_id,
            "label": spec["label"],
            "port": port,
            "up": up,
            "starting": bool(alive and not up),
            "managed": bool(alive),
            "pid": pid,
        }

    def start(self, demo_id: str, *, cpu: bool = False) -> dict:
        if demo_id not in DEMO_SPECS:
            return {"ok": False, "error": f"未知模块: {demo_id}"}
        if not RUN_SH.is_file():
            return {"ok": False, "error": f"缺少启动脚本: {RUN_SH}"}

        spec = DEMO_SPECS[demo_id]
        port = int(spec["port"])
        if _probe_port("127.0.0.1", port):
            return {
                "ok": True,
                "already": True,
                "message": f"{spec['label']}已在运行",
                **self.status(demo_id),
            }

        with self._lock:
            proc = self._procs.get(demo_id)
            if proc is not None and proc.poll() is None:
                return {
                    "ok": True,
                    "already": True,
                    "message": f"{spec['label']}正在启动中",
                    **self.status(demo_id),
                }
            if proc is not None:
                self._cleanup_locked(demo_id)

            cmd = [str(RUN_SH), str(spec["cmd"]), "--port", str(port)]
            if cpu:
                cmd.append("--cpu")

            log_path = LOG_DIR / f"{demo_id}.log"
            # 每次启动覆盖写，便于前端展示最近错误
            log_f = open(log_path, "wb", buffering=0)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            # 关键：门户若用 genesisEx 启动，子进程必须用同一解释器；
            # run.sh 里裸写 python 会落到 conda base，缺 quadrants 等依赖。
            env["PYTHON"] = sys.executable
            py_dir = str(Path(sys.executable).resolve().parent)
            env["PATH"] = f"{py_dir}{os.pathsep}{env.get('PATH', '')}"
            # 与 run.sh 对齐：缓存写入 newtest/.cache，避免 ~/.cache 权限导致启动失败
            qd_cache = ROOT_DIR / ".cache" / "qdcache"
            try:
                (qd_cache / "python_side_cache").mkdir(parents=True, exist_ok=True)
                (qd_cache / "kernel_compilation_manager").mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            env.setdefault("QD_OFFLINE_CACHE_FILE_PATH", str(qd_cache))
            popen = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR.parent),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            self._procs[demo_id] = popen
            self._log_handles[demo_id] = log_f
            print(
                f"[hub] 已启动 {demo_id} pid={popen.pid} python={sys.executable} → :{port} 日志 {log_path}",
                flush=True,
            )

        # 快速失败检测：依赖错误通常在 1–2 秒内退出
        time.sleep(1.2)
        st = self.status(demo_id)
        if not st["managed"] and not st["up"]:
            err = _tail_log(LOG_DIR / f"{demo_id}.log")
            return {
                "ok": False,
                "error": f"{spec['label']}启动失败（进程已退出）",
                "log": str(LOG_DIR / f"{demo_id}.log"),
                "log_tail": err,
                **st,
            }

        return {
            "ok": True,
            "started": True,
            "message": f"正在启动{spec['label']}，首次加载仿真可能需要数十秒",
            "log": str(LOG_DIR / f"{demo_id}.log"),
            **st,
        }

    def log_tail(self, demo_id: str, n: int = 40) -> str:
        return _tail_log(LOG_DIR / f"{demo_id}.log", n=n)

    def stop(self, demo_id: str) -> dict:
        if demo_id not in DEMO_SPECS:
            return {"ok": False, "error": f"未知模块: {demo_id}"}
        with self._lock:
            proc = self._procs.get(demo_id)
            if proc is None or proc.poll() is not None:
                self._cleanup_locked(demo_id)
                return {
                    "ok": True,
                    "stopped": False,
                    "message": "没有由门户托管的进程（若端口仍被占用，请在终端手动结束）",
                    **self.status(demo_id),
                }
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=3)
            self._cleanup_locked(demo_id)
        return {
            "ok": True,
            "stopped": True,
            "message": f"已停止{DEMO_SPECS[demo_id]['label']}",
            **self.status(demo_id),
        }

    def _cleanup_locked(self, demo_id: str) -> None:
        self._procs.pop(demo_id, None)
        handle = self._log_handles.pop(demo_id, None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass


LAUNCHER = DemoLauncher()


def _tail_log(path: Path, n: int = 40) -> str:
    try:
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = text.strip("\n").splitlines()
    return "\n".join(lines[-n:])


def _safe_under_dir(base: Path, rel: str) -> Optional[Path]:
    if not rel or ".." in rel or rel.startswith("/") or rel.startswith("\\"):
        return None
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


def _safe_frontend_path(url_path: str) -> Optional[Path]:
    # 浏览器会对中文文件名做百分号编码，需先解码再查文件
    raw = unquote(url_path.split("?", 1)[0])
    if raw in ("", "/"):
        raw = "/index.html"
    if ".." in raw or raw.startswith("//"):
        return None
    rel = raw.lstrip("/")
    if rel.startswith("gif/"):
        return _safe_under_dir(GIF_DIR, rel[4:])
    return _safe_under_dir(FRONTEND_DIR, rel)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


class Handler(BaseHTTPRequestHandler):
    server_version = f"SkillDemoHub/{VERSION}"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._handle_status(parsed)
            return
        if parsed.path == "/api/demos":
            self._send_json(
                200,
                {
                    "ok": True,
                    "demos": [LAUNCHER.status(i) for i in DEMO_SPECS],
                },
            )
            return
        if parsed.path == "/api/log":
            qs = parse_qs(parsed.query)
            demo_id = (qs.get("id") or [""])[0]
            if demo_id not in DEMO_SPECS:
                self._send_json(400, {"ok": False, "error": "非法 id"})
                return
            self._send_json(
                200,
                {
                    "ok": True,
                    "id": demo_id,
                    "log": str(LOG_DIR / f"{demo_id}.log"),
                    "log_tail": LAUNCHER.log_tail(demo_id),
                },
            )
            return
        if parsed.path == "/api/info":
            self._send_json(
                200,
                {
                    "name": SOFTWARE_NAME,
                    "version": VERSION,
                    "modules": ["portal", "skills", "nav", "play", "help"],
                    "can_launch": True,
                },
            )
            return

        path = _safe_frontend_path(parsed.path)
        if path is None:
            self.send_error(404, "Not Found")
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".html":
            ctype = "text/html; charset=utf-8"
        elif path.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        elif path.suffix == ".js":
            ctype = "application/javascript; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        parsed = urlparse(self.path)
        body = _read_json_body(self)
        qs = parse_qs(parsed.query)
        demo_id = str(body.get("id") or (qs.get("id") or [""])[0]).strip()
        cpu = bool(body.get("cpu")) or ((qs.get("cpu") or ["0"])[0] in ("1", "true", "True"))

        if parsed.path == "/api/start":
            if demo_id not in DEMO_SPECS:
                self._send_json(400, {"ok": False, "error": "缺少或非法的 id"})
                return
            self._send_json(200, LAUNCHER.start(demo_id, cpu=cpu))
            return
        if parsed.path == "/api/stop":
            if demo_id not in DEMO_SPECS:
                self._send_json(400, {"ok": False, "error": "缺少或非法的 id"})
                return
            self._send_json(200, LAUNCHER.stop(demo_id))
            return
        self.send_error(404, "Not Found")

    def _handle_status(self, parsed):
        qs = parse_qs(parsed.query)
        demo_id = (qs.get("id") or [""])[0]
        if demo_id in DEMO_SPECS:
            self._send_json(200, {"ok": True, **LAUNCHER.status(demo_id)})
            return
        try:
            port = int((qs.get("port") or ["0"])[0])
        except ValueError:
            port = 0
        if port not in ALLOWED_PORTS:
            self._send_json(400, {"ok": False, "error": "port not allowed", "up": False})
            return
        up = _probe_port("127.0.0.1", port)
        matched = next((k for k, v in DEMO_SPECS.items() if int(v["port"]) == port), None)
        payload = {"ok": True, "port": port, "up": up}
        if matched:
            payload.update(LAUNCHER.status(matched))
        self._send_json(200, payload)

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[hub] {self.address_string()} - {fmt % args}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=f"{SOFTWARE_NAME} {VERSION} 门户")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    if not FRONTEND_DIR.is_dir():
        raise FileNotFoundError(f"缺少前端目录: {FRONTEND_DIR}")

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        if getattr(e, "errno", None) == 98:  # Address already in use
            print(
                f"[hub] 端口 {args.port} 已被占用：门户可能已在运行。\n"
                f"  请直接打开 http://localhost:{args.port}\n"
                f"  若需重启，先结束旧进程：  fuser -k {args.port}/tcp\n"
                f"  然后再执行：  ./newtest/run.sh hub --port {args.port}",
                flush=True,
            )
            raise SystemExit(1) from e
        raise
    print(f"[hub] {SOFTWARE_NAME} {VERSION}", flush=True)
    print(f"[hub] http://localhost:{args.port}", flush=True)
    print(f"[hub] 前端目录: {FRONTEND_DIR}", flush=True)
    print("[hub] 支持网页一键启动 skills/nav/play（需在已安装 Genesis 的环境中启动本门户）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[hub] 退出", flush=True)
        for demo_id in list(DEMO_SPECS):
            LAUNCHER.stop(demo_id)


if __name__ == "__main__":
    main()
