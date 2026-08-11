#!/usr/bin/env bash
# 一键启动（请先按 README：conda lr_gen + pip install -e ".[genesis]"）
# 用法:
#   ./run.sh hub
#   ./run.sh skills [--cpu] [--port 8081]
#   ./run.sh nav [--cpu] [--port 8082]     # Lidar 导航（multivel）
#   ./run.sh play [--cpu] [--port 8083]    # 官方对齐 go2_wtw 行走
#
# 只依赖本目录（newtest/）内的脚本与资源；Genesis 本体来自当前 conda 环境。
# 默认 export SIMULATOR=genesis，无需用户再手动设置。
# legged_gym / rsl_rl / resources 已放在 newtest 根目录（与 LeggedGym-Ex 布局一致）。
#
# 可用环境变量 PYTHON 指定解释器（门户一键启动会注入，避免落到错误的 base python）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PARENT="$(dirname "$ROOT")"
PYTHON_BIN="${PYTHON:-python}"

export SIMULATOR="${SIMULATOR:-genesis}"
export PYTHONUNBUFFERED=1

# Quadrants 离线缓存放到 newtest 内，避免 ~/.cache/quadrants 权限/锁导致 scene.build 失败
QD_CACHE_DIR="${QD_OFFLINE_CACHE_FILE_PATH:-${ROOT}/.cache/qdcache}"
mkdir -p "${QD_CACHE_DIR}/python_side_cache" "${QD_CACHE_DIR}/kernel_compilation_manager" 2>/dev/null || true
export QD_OFFLINE_CACHE_FILE_PATH="${QD_CACHE_DIR}"
find "${QD_CACHE_DIR}" -name '*.lock' -delete 2>/dev/null || true
find "${QD_CACHE_DIR}" -name 'ticache.lock' -delete 2>/dev/null || true

# newtest + 内置 LeggedGym；可选：若仍放在 monorepo 且存在源码包，则附加（不强制）
EXTRA=""
if [[ -f "${PARENT}/genesis/genesis/__init__.py" ]]; then
  EXTRA="${PARENT}/genesis:"
elif [[ -f "${PARENT}/genesis/__init__.py" ]]; then
  EXTRA="${PARENT}:"
fi

# ROOT 优先（含 legged_gym / rsl_rl / resources）；vendor 仅作回退
LG_PATH="${ROOT}"
if [[ ! -d "${ROOT}/legged_gym" ]]; then
  LG_PATH="${ROOT}/vendor/LeggedGym-Ex"
fi
# PARENT 仅用于 `python -m newtest.*` 能找到包名；资源路径一律相对 ROOT
export PYTHONPATH="${EXTRA}${PARENT}:${LG_PATH}${PYTHONPATH:+:$PYTHONPATH}"

cmd="${1:-hub}"
shift || true

cd "$ROOT"

case "$cmd" in
  hub)
    exec "$PYTHON_BIN" -m newtest.hub "$@"
    ;;
  skills)
    exec "$PYTHON_BIN" -m newtest.skills.app "$@"
    ;;
  nav)
    exec "$PYTHON_BIN" -m newtest.nav.app "$@"
    ;;
  play)
    exec "$PYTHON_BIN" -m newtest.play.app "$@"
    ;;
  *)
    echo "用法: $0 {hub|skills|nav|play} [额外参数...]"
    echo "示例: $0 skills --port 8081"
    echo "      $0 nav --port 8082"
    echo "      $0 play --port 8083"
    exit 1
    ;;
esac
