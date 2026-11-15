#!/usr/bin/env bash
# 清理工程构建产物、日志和临时文件。默认不删除 Pinocchio 安装结果。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CLEAN_ALL=0
CLEAN_PINOCCHIO=0
DRY_RUN=0

usage() {
  cat <<USAGE
用法: $0 [--all] [--pinocchio] [--dry-run]

默认清理:
  build_*、cmake 临时文件、python_output、sdk_output、bin/log、bin/generated、__pycache__、pytest 缓存、compile_commands.json

选项:
  --pinocchio   同时删除 thirdparty/build/pinocchio-4.0.0 和 thirdparty/install/pinocchio-4.0.0
  --all         删除默认产物、Pinocchio 构建/安装产物、本地编译出的 revo2 可执行文件和 librevo2 动态库
  --dry-run     只打印将删除的路径，不实际删除
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) CLEAN_ALL=1; CLEAN_PINOCCHIO=1; shift ;;
    --pinocchio) CLEAN_PINOCCHIO=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[clean][ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

remove_path() {
  local path="$1"
  [[ -e "$path" || -L "$path" ]] || return 0
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[clean][DRY] rm -rf $path"
  else
    rm -rf "$path"
    echo "[clean][OK] removed $path"
  fi
}

remove_glob() {
  local pattern="$1"
  shopt -s nullglob dotglob
  local matches=( $pattern )
  shopt -u nullglob dotglob
  local item
  for item in "${matches[@]}"; do
    remove_path "$item"
  done
}

remove_glob "build*"
remove_path "python_output"
remove_path "sdk_output"
remove_path "bin/log"
remove_path "bin/generated"
remove_path ".pytest_cache"
remove_path "compile_commands.json"
remove_glob "*.tar.xz"
remove_glob "core"

find . -type d \( -name '__pycache__' -o -name '.mypy_cache' -o -name '.ruff_cache' \) -prune -print | while IFS= read -r path; do
  remove_path "$path"
done
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -print | while IFS= read -r path; do
  remove_path "$path"
done

if [[ $CLEAN_PINOCCHIO -eq 1 ]]; then
  remove_path "thirdparty/build/pinocchio-4.0.0"
  remove_path "thirdparty/install/pinocchio-4.0.0"
fi

if [[ $CLEAN_ALL -eq 1 ]]; then
  remove_glob "lib/librevo2_ik_sdk.so*"
  remove_glob "bin/revo2_ik_*"
fi

echo "[clean][DONE]"
