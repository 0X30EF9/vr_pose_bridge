from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Revo2 双手 IK 本机 sidecar worker")
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--sdk-config", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--response-fd", type=int, required=True)
    return parser.parse_args()


def emit(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def solve_request(sdk: Any, request: dict[str, Any]) -> dict[str, Any]:
    sequence = int(request["sequence"])
    timestamp_ms = int(request["timestamp_ms"])
    result = sdk.solve(
        left_points=request["left_points"],
        right_points=request["right_points"],
        left_quaternions_wxyz=request.get("left_quaternions_wxyz"),
        right_quaternions_wxyz=request.get("right_quaternions_wxyz"),
        sequence=sequence,
        timestamp_ms=timestamp_ms,
    )
    return {"type": "result", "sequence": sequence, "result": result}


def local_only_sdk_config(source_path: Path) -> tempfile.NamedTemporaryFile:
    source = source_path.expanduser().resolve()
    root = json.loads(source.read_text(encoding="utf-8"))
    for side in ("left_hand", "right_hand"):
        hand = root.get(side)
        if not isinstance(hand, dict) or not isinstance(hand.get("sdk_config"), str):
            raise ValueError(f"Revo2 配置缺少 {side}.sdk_config")
        hand["sdk_config"] = str((source.parent / hand["sdk_config"]).resolve())
    root["output_record"] = {"enabled": False}
    root["diagnostic_record"] = {"enabled": False}
    temporary = tempfile.NamedTemporaryFile(mode="w+", suffix=".json", encoding="utf-8")
    json.dump(root, temporary, ensure_ascii=False)
    temporary.flush()
    return temporary


def main() -> int:
    args = parse_args()
    sdk_root = args.sdk_root.expanduser().resolve()
    sdk_python = sdk_root / "python"
    if not (sdk_python / "revo2_hand_ik").is_dir():
        raise FileNotFoundError(f"Revo2 Python package 不存在: {sdk_python}")
    sys.path.insert(0, str(sdk_python))

    from revo2_hand_ik.api import Revo2DualHandIkSdk

    with open(args.response_fd, "w", encoding="utf-8", buffering=1, closefd=True) as response:
        try:
            with local_only_sdk_config(args.sdk_config) as local_config:
                with Revo2DualHandIkSdk(local_config.name, library_path=args.library) as sdk:
                    emit(response, {"type": "ready"})
                    for line in sys.stdin:
                        text = line.strip()
                        if not text:
                            continue
                        request: dict[str, Any] | None = None
                        try:
                            request = json.loads(text)
                            if request.get("type") == "close":
                                emit(response, {"type": "closed"})
                                return 0
                            if request.get("type") == "reset":
                                sdk.reset()
                                emit(response, {"type": "reset"})
                                continue
                            if request.get("type") != "solve":
                                raise ValueError(f"未知请求类型: {request.get('type')}")
                            emit(response, solve_request(sdk, request))
                        except Exception as exc:
                            emit(
                                response,
                                {
                                    "type": "error",
                                    "sequence": None if request is None else request.get("sequence"),
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                            )
        except Exception as exc:
            emit(response, {"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())