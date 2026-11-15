#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${VR_SIM_ENV:-$repo_dir/.venv}"
python_bin="${VR_SIM_PYTHON:-python3.12}"
if [ -z "${VR_SIM_PYTHON:-}" ] && [ -x "$env_dir/bin/python" ]; then
  python_bin="$env_dir/bin/python"
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python 3.11-3.12 is required (3.12 tested); set VR_SIM_PYTHON to its executable."
  exit 1
fi
"$python_bin" - <<'PY'
import sys
if not (3, 11) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit("Use Python 3.11-3.12; this Pinocchio/PyBullet environment was tested on 3.12.")
PY
if [ ! -x "$env_dir/bin/python" ]; then
  "$python_bin" -m venv "$env_dir"
fi
"$env_dir/bin/python" -m pip install -r "$repo_dir/requirements_simulation.txt"
echo "Ready: $env_dir/bin/python $repo_dir/run.py simulate --realtime"
