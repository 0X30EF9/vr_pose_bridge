#!/usr/bin/env python3
"""Repository entry point; the launcher performs all argument processing."""
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    entry = root / "app" / "run_ik.py"
    if not entry.is_file():
        raise FileNotFoundError("找不到 app/run_ik.py 启动入口")
    runpy.run_path(str(entry), run_name="__main__")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n操作已停止。")
        raise SystemExit(130) from None
